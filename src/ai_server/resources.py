"""Runtime resource governor - continuous adaptation while work is in flight.

`profiles.py` answers a startup question: what configuration fits this machine?
This module answers a running one: given what the machine is doing *right now*,
how much work should be in flight?

They are different problems. A profile is chosen once against free VRAM at
launch. Pressure changes constantly - the operator opens a browser, a batch
renders 300 DPI page images, Windows decides to index something. Without runtime
adaptation a fixed concurrency that was safe at startup will thrash or OOM an
hour into a thousand-file batch.

Two mechanisms:

  ConcurrencyPlan  per-stage worker counts, recomputed as pressure moves, with
                   hysteresis so the pool does not oscillate at a threshold.

  GPU lease        exclusive access to the GPU. On a small card the OCR model,
                   the language model and the translation model cannot be
                   co-resident; whichever loads second fails. The lease
                   serialises them instead of letting them collide.

Standard library only, and every public call is thread-safe: stages run
concurrently by design.
"""

from __future__ import annotations

import ctypes
import gc
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum

from .hardware import GIB, MIB, HardwareInfo, detect, disk_for

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: Free-RAM fractions marking each pressure level. Entering a worse level uses
#: these; leaving one requires clearing it by HYSTERESIS_MARGIN, so a plan does
#: not flap while hovering on a boundary.
RAM_FREE_ELEVATED = 0.25
RAM_FREE_HIGH = 0.15
RAM_FREE_CRITICAL = 0.08
HYSTERESIS_MARGIN = 0.05

#: Below this much free disk, admitting more work risks failing mid-batch.
DISK_FREE_CRITICAL_BYTES = 5 * GIB
DISK_FREE_ELEVATED_BYTES = 20 * GIB

#: Sustained CPU above this is treated as elevated pressure.
CPU_BUSY_ELEVATED = 0.90

#: Total VRAM below which AI models must not be co-resident. A 4 GB card cannot
#: hold an OCR model and a 4B language model at once; a 16 GB card can.
GPU_CORESIDENCY_MIN_BYTES = 12 * GIB

#: Working RAM a single worker of each stage needs. Page rendering dominates:
#: one 300 DPI A4 page is ~26 MB raw, and a worker holds several at a time.
STAGE_RAM_COST = {
    "pdf_render": 320 * MIB,
    "ocr_postprocess": 96 * MIB,
    "validate": 48 * MIB,
    "translate_post": 48 * MIB,
    "export": 64 * MIB,
}

#: Stages that must hold the GPU exclusively on a small card.
GPU_STAGES = frozenset({"ocr", "extract", "translate"})


class Pressure(IntEnum):
    """How constrained the machine is. Higher is worse."""

    NORMAL = 0
    ELEVATED = 1
    HIGH = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return self.name.lower()


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time view of the machine."""

    ram_total_bytes: int
    ram_available_bytes: int
    cpu_busy: float
    vram_free_bytes: int
    vram_total_bytes: int
    disk_free_bytes: int
    #: Host RAM that could be returned on demand - currently the language
    #: model's own footprint, which the OCR stage releases before it runs.
    reclaimable_bytes: int = 0
    taken_at: float = field(default_factory=time.monotonic)

    @property
    def ram_free_fraction(self) -> float:
        if not self.ram_total_bytes:
            return 1.0
        return self.ram_available_bytes / self.ram_total_bytes

    @property
    def effective_free_fraction(self) -> float:
        """Free RAM plus what can be handed back on demand.

        The distinction matters because the largest single consumer on this
        machine is reclaimable: unloading the language model returns 2.65 GiB,
        measured. Judging admission on `ram_free_fraction` alone means refusing
        work because of memory that the work itself would release.
        """
        if not self.ram_total_bytes:
            return 1.0
        return min(1.0, (self.ram_available_bytes + self.reclaimable_bytes)
                   / self.ram_total_bytes)

    @property
    def vram_free_fraction(self) -> float:
        if not self.vram_total_bytes:
            return 1.0
        return self.vram_free_bytes / self.vram_total_bytes

    def describe(self) -> str:
        return (
            f"RAM {self.ram_available_bytes / GIB:.1f}/{self.ram_total_bytes / GIB:.1f} GiB free "
            f"({self.ram_free_fraction:.0%}) | CPU {self.cpu_busy:.0%} | "
            f"VRAM {self.vram_free_bytes / GIB:.2f}/{self.vram_total_bytes / GIB:.2f} GiB | "
            f"disk {self.disk_free_bytes / GIB:.0f} GiB"
        )


class _CpuSampler:
    """Interval CPU utilisation without psutil."""

    def __init__(self) -> None:
        self._prev: tuple[int, int] | None = None

    def sample(self) -> float:
        totals = self._read()
        if totals is None:
            return 0.0
        idle, total = totals
        if self._prev is None:
            self._prev = (idle, total)
            return 0.0
        prev_idle, prev_total = self._prev
        self._prev = (idle, total)
        d_total = total - prev_total
        if d_total <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (idle - prev_idle) / d_total))

    @staticmethod
    def _read() -> tuple[int, int] | None:
        if os.name == "nt":
            class _FILETIME(ctypes.Structure):
                _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

            def to_int(ft: _FILETIME) -> int:
                return (ft.high << 32) | ft.low

            idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
            ok = ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            )
            if not ok:
                return None
            # Kernel time already includes idle time.
            return to_int(idle), to_int(kernel) + to_int(user)

        try:
            with open("/proc/stat", encoding="utf-8") as fh:
                parts = [int(x) for x in fh.readline().split()[1:]]
            return parts[3], sum(parts)
        except (OSError, ValueError, IndexError):
            return None


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrencyPlan:
    """How many workers each stage may run, given current pressure."""

    pressure: Pressure
    workers: dict[str, int]
    gpu_exclusive: bool
    admit_new_work: bool
    reason: str

    def for_stage(self, stage: str) -> int:
        return self.workers.get(stage, 1)

    def describe(self) -> str:
        stages = "  ".join(f"{k}={v}" for k, v in sorted(self.workers.items()))
        gate = "admitting" if self.admit_new_work else "PAUSED"
        return (
            f"[{self.pressure.label}] {gate} | {stages} | "
            f"gpu={'exclusive' if self.gpu_exclusive else 'shared'} | {self.reason}"
        )


# ---------------------------------------------------------------------------
# Governor
# ---------------------------------------------------------------------------


class ResourceGovernor:
    """Samples the machine and publishes a concurrency plan that tracks it."""

    def __init__(
        self,
        hw: HardwareInfo | None = None,
        *,
        data_paths: tuple[str, ...] = (),
        interval_s: float = 3.0,
        min_workers: int = 1,
        reclaimable: Callable[[], int] | None = None,
    ) -> None:
        self.hw = hw or detect(disk_paths=data_paths)
        #: Returns host RAM that could be handed back on demand. Supplied by the
        #: server, which is the only thing that knows whether the language model
        #: is loaded and how large it is.
        self.reclaimable_provider = reclaimable
        self.data_paths = data_paths
        self.interval_s = interval_s
        #: Never drop below this. Zero workers means a stalled batch, which is
        #: worse than slow progress.
        self.min_workers = max(1, min_workers)

        self._cpu = _CpuSampler()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[Callable[[ConcurrencyPlan], None]] = []

        vram_total = self.hw.primary_gpu.total_bytes if self.hw.primary_gpu else 0
        #: Small cards cannot hold two models; serialise GPU stages.
        self._gpu_exclusive = vram_total < GPU_CORESIDENCY_MIN_BYTES
        self._gpu_lock = threading.RLock()
        self._gpu_holder: str | None = None

        # Tracked separately from the plan so _classify can read it during the
        # very first _compute(), before a plan exists.
        self._pressure = Pressure.NORMAL
        self._snapshot = self._sample()
        self._plan = self._compute(self._snapshot)

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._loop, name="resource-governor", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        with self._lock:
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            snapshot = self._sample()
            plan = self._compute(snapshot)
            with self._lock:
                changed = plan.pressure != self._plan.pressure or plan.workers != self._plan.workers
                self._snapshot, self._plan = snapshot, plan
                listeners = list(self._listeners)
            if plan.pressure >= Pressure.HIGH:
                self.release_memory()
            if changed:
                for listener in listeners:
                    try:
                        listener(plan)
                    except Exception:  # noqa: BLE001 - a bad listener must not kill the governor
                        pass

    # -- public state -----------------------------------------------------

    def plan(self) -> ConcurrencyPlan:
        with self._lock:
            return self._plan

    def snapshot(self) -> ResourceSnapshot:
        with self._lock:
            return self._snapshot

    def on_change(self, callback: Callable[[ConcurrencyPlan], None]) -> None:
        """Register a listener fired when the plan changes (pool resizing)."""
        with self._lock:
            self._listeners.append(callback)

    # -- sampling and planning -------------------------------------------

    def _sample(self) -> ResourceSnapshot:
        fresh = detect(probe_other_adapters=False, disk_paths=self.data_paths)
        gpu = fresh.primary_gpu
        disks = fresh.disks or disk_for(*self.data_paths) if self.data_paths else fresh.disks
        disk_free = min((d.free_bytes for d in disks), default=0)
        return ResourceSnapshot(
            ram_total_bytes=fresh.ram_total_bytes,
            ram_available_bytes=fresh.ram_available_bytes,
            cpu_busy=self._cpu.sample(),
            vram_free_bytes=gpu.free_bytes if gpu else 0,
            vram_total_bytes=gpu.total_bytes if gpu else 0,
            disk_free_bytes=disk_free,
            reclaimable_bytes=self._reclaimable(),
        )

    def _reclaimable(self) -> int:
        """Host RAM the engine would return if asked. Zero when nothing can."""
        if self.reclaimable_provider is None:
            return 0
        try:
            return max(0, int(self.reclaimable_provider()))
        except Exception:  # noqa: BLE001 - a broken probe must not block work
            return 0

    def _classify(self, snap: ResourceSnapshot) -> tuple[Pressure, str]:
        """Map a snapshot to a pressure level, biased toward staying put.

        Hysteresis is one-directional: worsening is immediate (protecting the
        machine), recovering requires clearing the threshold by a margin.
        """
        with self._lock:
            current = self._pressure

        free = snap.ram_free_fraction
        margin = HYSTERESIS_MARGIN

        if snap.disk_free_bytes and snap.disk_free_bytes < DISK_FREE_CRITICAL_BYTES:
            return Pressure.CRITICAL, f"disk below {DISK_FREE_CRITICAL_BYTES / GIB:.0f} GiB"
        if free < RAM_FREE_CRITICAL:
            return Pressure.CRITICAL, f"RAM {free:.0%} free"
        if free < RAM_FREE_HIGH:
            return Pressure.HIGH, f"RAM {free:.0%} free"
        if free < RAM_FREE_ELEVATED:
            return Pressure.ELEVATED, f"RAM {free:.0%} free"
        if snap.disk_free_bytes and snap.disk_free_bytes < DISK_FREE_ELEVATED_BYTES:
            return Pressure.ELEVATED, f"disk {snap.disk_free_bytes / GIB:.0f} GiB free"
        if snap.cpu_busy >= CPU_BUSY_ELEVATED:
            return Pressure.ELEVATED, f"CPU {snap.cpu_busy:.0%} busy"

        # Recovering: require clearing the level we are leaving, plus a margin.
        if current >= Pressure.ELEVATED and free < RAM_FREE_ELEVATED + margin:
            return Pressure.ELEVATED, f"RAM {free:.0%} free (recovering)"
        return Pressure.NORMAL, "resources available"

    def _compute(self, snap: ResourceSnapshot) -> ConcurrencyPlan:
        pressure, reason = self._classify(snap)
        with self._lock:
            self._pressure = pressure

        # Leave a core for the UI so the interface stays responsive under load.
        cpu_ceiling = max(self.min_workers, self.hw.physical_cores - 1)
        # And never plan more workers than working RAM can back.
        usable_ram = max(0, snap.ram_available_bytes - 512 * MIB)

        scale = {
            Pressure.NORMAL: 1.0,
            Pressure.ELEVATED: 0.5,
            Pressure.HIGH: 0.25,
            Pressure.CRITICAL: 0.0,
        }[pressure]

        workers: dict[str, int] = {}
        for stage, cost in STAGE_RAM_COST.items():
            by_ram = usable_ram // cost if cost else cpu_ceiling
            allowed = int(min(cpu_ceiling, by_ram) * scale)
            workers[stage] = max(self.min_workers, allowed)

        # GPU stages are paced by the lease, not by a worker count.
        for stage in sorted(GPU_STAGES):
            workers[stage] = 1 if self._gpu_exclusive else max(
                self.min_workers, int(2 * scale) or self.min_workers
            )

        # Under critical pressure, finish what is running but admit nothing
        # new. Killing in-flight work would lose a document; pausing is safe
        # because every stage is resumable.
        #
        # But "critical" is judged on free RAM, and on a small machine the
        # single largest consumer is the language model - which the OCR stage
        # releases before it starts, returning 2.65 GiB (measured). Refusing to
        # admit work because of memory that admitting the work would free is a
        # deadlock, not a safeguard: nothing else was ever going to release it,
        # so the state never cleared and the application simply stopped working.
        #
        # `pressure` therefore stays honest - it describes the machine as it is,
        # and the worker counts above are scaled from it. Only the admission
        # decision looks at what could be freed.
        admit = (pressure < Pressure.CRITICAL
                 or snap.effective_free_fraction >= RAM_FREE_CRITICAL)

        return ConcurrencyPlan(
            pressure=pressure,
            workers=workers,
            gpu_exclusive=self._gpu_exclusive,
            admit_new_work=admit,
            reason=reason if admit else f"{reason}, nothing reclaimable",
        )

    # -- GPU arbitration --------------------------------------------------

    class _Lease:
        def __init__(self, governor: ResourceGovernor, stage: str, exclusive: bool):
            self._gov, self._stage, self._exclusive = governor, stage, exclusive

        def __enter__(self) -> ResourceGovernor._Lease:
            if self._exclusive:
                self._gov._gpu_lock.acquire()
                self._gov._gpu_holder = self._stage
            return self

        def __exit__(self, *exc: object) -> None:
            if self._exclusive:
                self._gov._gpu_holder = None
                self._gov._gpu_lock.release()

    def gpu_lease(self, stage: str) -> ResourceGovernor._Lease:
        """Acquire the GPU for a stage.

        On a card too small for co-residency this serialises OCR, extraction and
        translation so their models never try to occupy VRAM simultaneously. On
        a large card it is a no-op and the stages overlap freely.
        """
        return ResourceGovernor._Lease(self, stage, self._gpu_exclusive and stage in GPU_STAGES)

    @property
    def gpu_holder(self) -> str | None:
        return self._gpu_holder

    # -- memory hygiene ---------------------------------------------------

    @staticmethod
    def release_memory() -> None:
        """Return freed memory to the OS.

        CPython's allocator holds freed arenas, and on Windows a long-running
        process accumulates a large working set that never shrinks on its own.
        Trimming keeps a multi-hour batch from looking like a leak.
        """
        gc.collect()
        if os.name == "nt":
            try:
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, -1, -1)
            except (AttributeError, OSError):
                pass
        else:
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    gov = ResourceGovernor(data_paths=(os.getcwd(),))
    print(gov.hw.summary())
    print()
    print("Snapshot")
    print(" ", gov.snapshot().describe())
    print()
    print("Concurrency plan")
    print(" ", gov.plan().describe())
    print()
    print(f"GPU co-residency: {'DISABLED' if gov.plan().gpu_exclusive else 'enabled'} "
          f"(threshold {GPU_CORESIDENCY_MIN_BYTES / GIB:.0f} GiB VRAM)")
    with gov.gpu_lease("extract"):
        print(f"  lease held by: {gov.gpu_holder}")
    print(f"  lease released: {gov.gpu_holder}")
