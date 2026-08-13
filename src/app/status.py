"""Background status probing and capability gating.

The UI must never wait on a dependency to paint. Measured on Windows, a TCP
connect to a closed local port costs **2,045 ms** - the SYN is dropped and
retransmitted rather than refused - so four synchronous status checks turned a
dashboard render into 8.4 seconds. Every probe here therefore runs on a
background thread and the UI reads whatever the last completed probe left behind.

Three rules this module exists to enforce:

**Never show a confident lie.** Before the first probe completes the state is
`UNKNOWN`, not "Active". A stale-but-plausible indicator is worse than an honest
"Checking..." because an operator will act on it.

**Never hammer a dead dependency.** A circuit breaker opens after repeated
failures and stops probing for a cooldown. Without it the 2.5 s poll would launch
a 2 s connect every cycle and never catch up.

**A database outage is not an AI outage.** Without the AI server you can still
browse extracted data and export CSVs. Without the database almost nothing works.
The two produce different capability sets, and the UI disables what would fail
rather than letting it fail.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Localhost answers in microseconds or not at all; a long timeout only buys
#: Windows' SYN retransmission delay.
CONNECT_TIMEOUT_S = 0.4
READ_TIMEOUT_S = 3.0
#: How long a successful probe is trusted before it is refreshed.
DEFAULT_TTL_S = 5.0
#: A result older than this is shown as stale in the UI.
STALE_AFTER_S = 20.0


class Availability(str, Enum):
    UNKNOWN = "unknown"     # never probed - first paint
    CHECKING = "checking"   # probe in flight, nothing cached yet
    UP = "up"
    LOADING = "loading"     # reachable but not ready (model still loading)
    DEGRADED = "degraded"   # reachable, refusing work (resource pressure)
    DOWN = "down"

    @property
    def label(self) -> str:
        return {
            "unknown": "Checking…", "checking": "Checking…",
            "up": "Active", "loading": "Loading model…",
            "degraded": "Busy", "down": "Offline",
        }[self.value]

    @property
    def dot(self) -> str:
        return {"up": "on", "loading": "busy", "degraded": "busy",
                "down": "off", "unknown": "off", "checking": "busy"}[self.value]

    @property
    def usable(self) -> bool:
        return self in (Availability.UP, Availability.DEGRADED)


@dataclass
class ProbeResult:
    availability: Availability = Availability.UNKNOWN
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    checked_at: float | None = None
    error: str = ""

    @property
    def age_s(self) -> float:
        return 0.0 if self.checked_at is None else time.monotonic() - self.checked_at

    @property
    def stale(self) -> bool:
        """True when the value is old enough that the UI should say so."""
        return self.checked_at is not None and self.age_s > STALE_AFTER_S

    @property
    def never_checked(self) -> bool:
        return self.checked_at is None

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.availability.value,
            "label": self.availability.label,
            "dot": self.availability.dot,
            "detail": self.detail,
            "stale": self.stale,
            "age_s": round(self.age_s, 1) if self.checked_at else None,
            "error": self.error,
        }


class CircuitBreaker:
    """Stops probing a dependency that keeps failing.

    Without this, a dead AI server costs 2 s on every 2.5 s poll - the poll never
    completes inside its interval and the UI feels locked up.
    """

    def __init__(self, threshold: int = 2, cooldown_s: float = 15.0) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def open(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return False
            if time.monotonic() - self._opened_at >= self.cooldown_s:
                # Half-open: allow one attempt through to test recovery.
                self._opened_at = None
                self._failures = self.threshold - 1
                return False
            return True

    def record(self, ok: bool) -> None:
        with self._lock:
            if ok:
                self._failures = 0
                self._opened_at = None
            else:
                self._failures += 1
                if self._failures >= self.threshold:
                    self._opened_at = time.monotonic()

    @property
    def retry_in_s(self) -> float:
        with self._lock:
            if self._opened_at is None:
                return 0.0
            return max(0.0, self.cooldown_s - (time.monotonic() - self._opened_at))


def http_json(url: str, *, connect_timeout: float = CONNECT_TIMEOUT_S,
              read_timeout: float = READ_TIMEOUT_S) -> dict[str, Any]:
    """GET JSON with a short, explicit timeout. Raises on failure."""
    # urllib applies one timeout to both connect and read; the connect budget is
    # what matters here, so the larger of the two is used and kept small.
    timeout = max(connect_timeout, min(read_timeout, 3.0))
    request = urllib.request.Request(url, headers={"Connection": "close"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


@dataclass
class Probe:
    """One dependency, refreshed on its own schedule."""

    name: str
    check: Callable[[], ProbeResult]
    ttl_s: float = DEFAULT_TTL_S
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    result: ProbeResult = field(default_factory=ProbeResult)
    _running: bool = False

    def due(self) -> bool:
        if self._running or self.breaker.open:
            return False
        return self.result.never_checked or self.result.age_s >= self.ttl_s


#: Below this fraction of RAM free, the governor stops admitting work. Kept here
#: only to phrase the message; the decision belongs to the AI server.
_LOW_RAM_FRACTION = 0.10
_GIB = 1024 ** 3


def _pressure_reason(health: dict[str, Any]) -> str:
    """Explain, in the operator's terms, why processing is being refused.

    "AI server offline" was the previous wording and it was wrong twice over:
    the server is running with the model loaded, and the thing that needs fixing
    is on the operator's desktop, not in the service. Naming the actual resource
    is the difference between a user restarting something that is already
    healthy and closing a browser.
    """
    resources = health.get("resources") or {}
    pressure = str(health.get("pressure") or "pressure")

    ram_free = resources.get("ram_available_bytes") or 0
    ram_total = resources.get("ram_total_bytes") or 0
    if ram_total and ram_free / ram_total < _LOW_RAM_FRACTION:
        return (f"Not enough free memory - {ram_free / _GIB:.1f} GB of "
                f"{ram_total / _GIB:.1f} GB available. Close other applications "
                "and processing will resume automatically.")

    vram_free = resources.get("vram_free_bytes") or 0
    vram_total = resources.get("vram_total_bytes") or 0
    if vram_total and vram_free / vram_total < _LOW_RAM_FRACTION:
        return (f"Graphics memory is full - {vram_free / _GIB:.1f} GB of "
                f"{vram_total / _GIB:.1f} GB free. Processing resumes when the "
                "current work releases it.")

    disk_free = resources.get("disk_free_bytes") or 0
    if disk_free and disk_free < 2 * _GIB:
        return (f"Only {disk_free / _GIB:.1f} GB of disk space is free. "
                "Free some space before processing.")

    return (f"The AI server is running but is not accepting work "
            f"({pressure} resource pressure). It resumes on its own.")


@dataclass
class Capabilities:
    """What the operator may do right now, and why not otherwise.

    Deliberately asymmetric. The database holds every batch, document and
    extraction, so without it browsing, exporting and uploading are all
    impossible. The AI server only produces new extractions - without it, work
    already done remains fully usable.
    """

    can_browse: bool = False
    can_export: bool = False
    can_upload: bool = False
    can_process: bool = False
    reasons: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "can_browse": self.can_browse, "can_export": self.can_export,
            "can_upload": self.can_upload, "can_process": self.can_process,
            "reasons": self.reasons,
        }


class StatusService:
    """Owns every dependency probe and answers instantly from cache."""

    def __init__(self, ai_base_url: str, db_check: Callable[[], ProbeResult],
                 *, workers: int = 4) -> None:
        self.ai_base_url = ai_base_url.rstrip("/")
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="status")
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.probes: dict[str, Probe] = {
            "database": Probe("database", db_check, ttl_s=10.0),
            "ai": Probe("ai", self._check_ai, ttl_s=5.0),
            "gpu": Probe("gpu", self._check_gpu, ttl_s=5.0),
            # Profile and hardware are constant for a process lifetime; a long
            # TTL keeps them out of the hot path entirely.
            "profile": Probe("profile", self._check_profile, ttl_s=300.0),
        }

    # -- lifecycle --------------------------------------------------------

    def start(self, interval_s: float = 2.0) -> None:
        """Begin refreshing in the background. Returns immediately."""
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, args=(interval_s,), name="status", daemon=True)
            self._thread.start()
        self.refresh_now()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        self._thread = None
        self._pool.shutdown(wait=False, cancel_futures=True)

    def refresh_now(self) -> None:
        """Kick every due probe concurrently. Does not wait."""
        for probe in self.probes.values():
            if probe.due():
                self._submit(probe)

    def _loop(self, interval_s: float) -> None:
        while not self._stop.wait(interval_s):
            self.refresh_now()

    def _submit(self, probe: Probe) -> None:
        probe._running = True

        def run() -> None:
            try:
                result = probe.check()
            except Exception as exc:  # noqa: BLE001 - a probe must never raise out
                result = ProbeResult(Availability.DOWN, error=f"{type(exc).__name__}: {exc}")
            result.checked_at = time.monotonic()
            probe.breaker.record(result.availability.usable)
            with self._lock:
                probe.result = result
            probe._running = False

        self._pool.submit(run)

    # -- individual checks -------------------------------------------------

    def _check_ai(self) -> ProbeResult:
        try:
            payload = http_json(f"{self.ai_base_url}/health")
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            return ProbeResult(Availability.DOWN, "not reachable", error=str(exc))

        engine = payload.get("engine") or {}
        pressure = str(payload.get("pressure") or "")

        if payload.get("ready"):
            state = (Availability.DEGRADED if not payload.get("admitting_work")
                     else Availability.UP)
            detail = f"{engine.get('engine', '?')} on {engine.get('device', '?')}"
            if state is Availability.DEGRADED:
                detail = f"pressure {pressure}"
            return ProbeResult(state, detail, payload)

        # Reachable but not ready. Three different situations reach here and
        # they need different words, because each sends the operator somewhere
        # different.
        if engine.get("loaded"):
            # The server answered and the model is resident. `ready` is false
            # only because the governor is refusing work - almost always host
            # RAM. Calling this "offline" is simply untrue, and it sends the
            # user to restart a service that is running perfectly well instead
            # of closing the applications that are actually short of memory.
            return ProbeResult(Availability.DEGRADED,
                               f"pressure {pressure}" if pressure else "not admitting work",
                               payload)

        # Distinguishing "loading the model" from "offline" matters too: the
        # first load takes about a minute, and showing Offline for that long
        # reads as a failure.
        loading = not engine.get("detail", "").startswith("error")
        return ProbeResult(
            Availability.LOADING if loading else Availability.DOWN,
            engine.get("detail") or f"pressure {pressure}", payload)

    def _check_gpu(self) -> ProbeResult:
        ai = self.probes["ai"].result
        payload = ai.data if ai.availability.usable else {}
        if not payload:
            return ProbeResult(Availability.UNKNOWN, "unavailable")
        resources = payload.get("resources") or {}
        total = resources.get("vram_total_bytes") or 0
        free = resources.get("vram_free_bytes") or 0
        used = (1 - free / total) if total else 0.0
        return ProbeResult(Availability.UP, f"{used:.0%} used", {
            "vram_free_bytes": free, "vram_total_bytes": total,
            "ram_available_bytes": resources.get("ram_available_bytes") or 0,
            "ram_total_bytes": resources.get("ram_total_bytes") or 0,
            "pressure": payload.get("pressure"),
        })

    def _check_profile(self) -> ProbeResult:
        try:
            profile = http_json(f"{self.ai_base_url}/profile", read_timeout=3.0)
            hardware = http_json(f"{self.ai_base_url}/hardware", read_timeout=3.0)
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
            return ProbeResult(Availability.DOWN, "unavailable", error=str(exc))
        return ProbeResult(Availability.UP, profile.get("quantisation") or "",
                           {"profile": profile, "hardware": hardware})

    # -- reading -----------------------------------------------------------

    def get(self, name: str) -> ProbeResult:
        with self._lock:
            return self.probes[name].result

    def capabilities(self) -> Capabilities:
        database = self.get("database")
        ai = self.get("ai")

        db_ok = database.availability.usable
        ai_ok = ai.availability.usable
        reasons: dict[str, str] = {}

        # Reachable is not the same as willing. The governor refuses work under
        # resource pressure while the server stays up and the model stays
        # loaded, so admission has to be checked separately from availability.
        admitting = bool((ai.data or {}).get("admitting_work", ai_ok))
        can_process = db_ok and ai_ok and admitting

        if not db_ok:
            note = (database.error or database.detail
                    or "the database is not reachable")
            for action in ("browse", "export", "upload", "process"):
                reasons[action] = f"Database unavailable - {note}"
        elif not ai_ok:
            # Everything already stored stays usable; only new work is blocked.
            reasons["process"] = (
                f"AI server {ai.availability.label.lower()} - "
                "existing data can still be browsed and exported")
        elif not admitting:
            reasons["process"] = _pressure_reason(ai.data or {})

        return Capabilities(
            can_browse=db_ok, can_export=db_ok, can_upload=db_ok,
            can_process=can_process, reasons=reasons)

    def snapshot(self) -> dict[str, Any]:
        """Everything the UI needs, from cache. Never blocks, never raises."""
        with self._lock:
            results = {name: probe.result for name, probe in self.probes.items()}
            breakers = {name: probe.breaker.retry_in_s
                        for name, probe in self.probes.items()}

        gpu = results["gpu"].data
        profile = results["profile"].data.get("profile") or {}
        total = gpu.get("vram_total_bytes") or 0
        free = gpu.get("vram_free_bytes") or 0

        return {
            "database": results["database"].as_dict(),
            "ai": results["ai"].as_dict(),
            "gpu": {
                **results["gpu"].as_dict(),
                "vram_free_bytes": free, "vram_total_bytes": total,
                "util": f"{(1 - free / total):.0%}" if total else "-",
            },
            "profile": profile,
            "hardware": results["profile"].data.get("hardware") or {},
            "capabilities": self.capabilities().as_dict(),
            "retry_in": {k: round(v, 1) for k, v in breakers.items() if v > 0},
        }
