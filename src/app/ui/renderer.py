"""Pystache rendering and view models for the desktop UI.

Templates live in `templates/`, assets in `assets/`. Rendering is pure: a view
model dict in, HTML out, no database and no Qt. That keeps every screen testable
by asserting on the produced HTML, which is the only practical way to check a
webview UI without driving a browser.

View models are built here rather than in the templates because Mustache is
deliberately logic-less. Anything needing a decision - a percentage, a badge
class, a formatted byte count, a relative time - is computed in Python and
handed over as a plain value.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pystache

UI_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = UI_DIR / "templates"
ASSET_DIR = UI_DIR / "assets"

#: Nav entries: (key, template, title). `key` also drives the active highlight.
PAGES: tuple[tuple[str, str, str], ...] = (
    ("dashboard", "dashboard", "Dashboard"),
    ("upload", "upload", "Upload PDFs"),
    ("processing", "processing", "PDF Processing"),
    ("failed_ocr", "failed_ocr", "Failed OCR"),
    ("data", "data_view", "Data View"),
    ("watermark", "watermark", "Watermark Remover"),
    ("settings", "settings", "Settings"),
    ("validation", "validation", "Validation Rules"),
    ("help", "help", "Help"),
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def human_bytes(value: int | float | None) -> str:
    if not value:
        return "0 B"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def human_duration(seconds: float | None) -> str:
    """Compact duration. Batches here can run for hours, so hours matter."""
    if not seconds or seconds <= 0:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def local_time(value: datetime | None) -> str:
    if value is None:
        return "-"
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone().strftime("%d-%m-%Y %H:%M")


def percent(done: int, total: int) -> int:
    return int(round(100.0 * done / total)) if total else 0


def pressure_class(level: str) -> str:
    return {"normal": "ok", "elevated": "warn", "high": "warn",
            "critical": "danger"}.get(level, "")


def state_badge(state: str) -> str:
    return {
        "completed": "ok", "processed": "ok",
        "running": "accent",
        "needs_review": "review", "paused": "review",
        # A stopped batch is not a failed one - it is healthy and resumable, so
        # it must not wear the red badge that means "something went wrong".
        # `stopping` is transitional and reads as in-progress.
        "stopping": "accent", "stopped": "review",
        "failed": "danger",
    }.get(state, "")


def pager(page: int, per_page: int, total: int) -> dict[str, Any]:
    """Pagination view model. Windowed so a thousand batches stay usable."""
    pages = max(1, (total + per_page - 1) // per_page)
    page = min(max(1, page), pages)
    lo = max(1, page - 2)
    hi = min(pages, lo + 4)
    lo = max(1, hi - 4)
    first = (page - 1) * per_page + 1 if total else 0
    return {
        "page": page,
        "pages": pages,
        "total": total,
        "has_prev": page > 1,
        "has_next": page < pages,
        "prev": page - 1,
        "next": page + 1,
        "numbers": [{"n": n, "current": n == page} for n in range(lo, hi + 1)],
        "range_from": first,
        "range_to": min(total, page * per_page),
        "multi": pages > 1,
    }


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


@dataclass
class Chrome:
    """Values shown in the top bar on every page."""

    ai_ready: bool = False
    running: bool = False
    runner_state: str = "idle"
    gpu_util: str = "-"
    vram_free: str = "-"
    translation_language: str = "English"


class Renderer:
    """Renders a page into the shell.

    Templates are read from disk on every render when `reload` is set, which
    makes iterating on the UI immediate; production loads once and caches.
    """

    def __init__(self, *, reload: bool = False) -> None:
        self.reload = reload
        self._cache: dict[str, str] = {}
        self._renderer = pystache.Renderer(
            partials=self,          # `self.get` resolves {{> partial}}
            escape=lambda raw: html.escape(str(raw), quote=True),
            missing_tags="ignore",  # an absent optional field renders empty
        )

    # pystache partial resolver protocol
    def get(self, name: str) -> str:
        return self._template(name)

    def _template(self, name: str) -> str:
        if not self.reload and name in self._cache:
            return self._cache[name]
        path = TEMPLATE_DIR / f"{name}.mustache"
        if not path.is_file():
            return ""
        text = path.read_text(encoding="utf-8")
        self._cache[name] = text
        return text

    def render_page(self, key: str, model: dict[str, Any] | None = None,
                    chrome: Chrome | None = None, *, shell_html: bool = True) -> str:
        """Render one page. With `shell_html` false, returns the content only.

        Navigation replaces the content area rather than the whole document.
        Rewriting the document with `document.write` tears down the JavaScript
        context and, with it, the QWebChannel - a new one is then constructed
        over the same transport, and any reply still in flight for the old
        channel arrives at a channel that has never heard of it
        (`execCallbacks[message.id] is not a function`). One channel per session
        is the only arrangement in which an asynchronous reply can be relied on.
        """
        template_name, title = self._page_meta(key)
        context: dict[str, Any] = dict(model or {})
        inner = self._renderer.render(self._template(template_name), context)

        if not shell_html:
            # The banner lives in the shell, so a content-only render has to
            # carry it or it would vanish on the first navigation.
            if context.get("degraded"):
                inner = self._renderer.render(
                    self._template("capability_banner"), context) + inner
            return inner

        shell: dict[str, Any] = {
            "page_title": title,
            # `{{{content}}}` is intentionally unescaped: it is HTML this
            # renderer produced, and every value inside it was escaped when the
            # inner template rendered.
            "content": inner,
        }
        chrome = chrome or Chrome()
        shell.update({
            "ai_ready": chrome.ai_ready,
            "running": chrome.running,
            "runner_state": chrome.runner_state,
            "gpu_util": chrome.gpu_util,
            "vram_free": chrome.vram_free,
            "translation_language": chrome.translation_language,
        })
        # Capability keys are forwarded from the page model into the shell. The
        # banner lives in base.mustache so it appears once above the content
        # rather than being repeated on every page, but the values are computed
        # per render alongside the page model - without this the shell renders
        # `{{#degraded}}` against a context that has never heard of it, and the
        # block silently collapses to nothing.
        for key_name in ("degraded", "capability_reasons"):
            if key_name in context:
                shell[key_name] = context[key_name]
        for nav_key, _, _ in PAGES:
            shell[f"nav_{nav_key}"] = nav_key == key
        return self._renderer.render(self._template("base"), shell)

    def render_fragment(self, template_name: str,
                        model: dict[str, Any] | None = None) -> str:
        """Render a template on its own - used for modals and live updates."""
        return self._renderer.render(self._template(template_name), dict(model or {}))

    @staticmethod
    def _page_meta(key: str) -> tuple[str, str]:
        for nav_key, template_name, title in PAGES:
            if nav_key == key:
                return template_name, title
        return "dashboard", "Dashboard"

    def asset(self, name: str) -> bytes:
        path = ASSET_DIR / name
        return path.read_bytes() if path.is_file() else b""


# ---------------------------------------------------------------------------
# View model builders
# ---------------------------------------------------------------------------


def dashboard_model(
    *,
    active: dict[str, Any] | None,
    manage: list[dict[str, Any]],
    completed: list[dict[str, Any]],
    page: int = 1,
    per_page: int = 5,
    completed_total: int = 0,
    max_queued: int = 4,
    notice: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the dashboard view model.

    Stage bars are given distinct colours so an operator can tell at a glance
    which phase a batch is in without reading labels.
    """
    stage_meta = (
        ("ocr", "OCR", ""),
        ("extract", "Extraction", "accent"),
        ("translate", "Translation", "saffron"),
        ("validate", "Validation", ""),
    )

    active_model = None
    if active:
        total = active.get("total") or 0
        stages = []
        for key, label, bar in stage_meta:
            stage = (active.get("stages") or {}).get(key) or {}
            done = stage.get("done") or 0
            stages.append({"label": label, "bar_class": bar, "done": done,
                           "total": total, "percent": percent(done, total)})
        finished = ((active.get("completed") or 0) + (active.get("failed") or 0)
                    + (active.get("needs_review") or 0))
        active_model = {
            "name": active.get("name"),
            "total": total,
            "state": active.get("state"),
            "state_class": state_badge(str(active.get("state") or "")),
            "stages": stages,
            "completed": active.get("completed") or 0,
            "needs_review": active.get("needs_review") or 0,
            "failed": active.get("failed") or 0,
            "remaining": max(0, total - finished),
            "seconds_per_document": active.get("seconds_per_document") or 0,
            "eta": human_duration(active.get("eta_seconds")),
            "started_at": local_time(active.get("started_at")),
        }

    # The badge class is decided here rather than in the service, so every
    # state in the application is coloured by one function; a second mapping
    # would drift the first time a state was added.
    for row in (*manage, *completed):
        row["state_class"] = state_badge(str(row.get("state") or ""))

    return {
        "active": active_model,
        "manage": manage,
        "has_manage": bool(manage),
        "queued_count": sum(1 for r in manage if r.get("state") == "queued"),
        "max_queued": max_queued,
        "completed": completed,
        "has_completed": bool(completed),
        "notice": notice,
        "pager": pager(page, per_page, completed_total or len(completed)),
    }


def machine_model(hardware: dict[str, Any], profile: dict[str, Any],
                  health: dict[str, Any], db_backend: str) -> dict[str, Any]:
    """Machine panel: what the app detected and what it chose to run."""
    gpus = hardware.get("gpus") or []
    gpu = gpus[0] if gpus else {}
    resources = health.get("resources") or {}
    disks = hardware.get("disks") or []

    return {
        "gpu_name": (gpu.get("name") or "No CUDA GPU").replace("NVIDIA GeForce ", ""),
        "vram": (f"{human_bytes(resources.get('vram_free_bytes'))} / "
                 f"{human_bytes(resources.get('vram_total_bytes'))}"),
        "ram": (f"{human_bytes(resources.get('ram_available_bytes'))} / "
                f"{human_bytes(resources.get('ram_total_bytes'))}"),
        "pressure": (health.get("pressure") or "unknown").title(),
        "pressure_class": pressure_class(str(health.get("pressure") or "")),
        "model_name": (health.get("engine") or {}).get("model") or "-",
        "quantisation": profile.get("quantisation") or "-",
        # Surfaced so the operator knows when output is not bit-identical to the
        # trained model.
        "lossy_model": not profile.get("lossless", True),
        "disk_free": human_bytes(disks[0].get("free_bytes") if disks else 0),
        "db_backend": db_backend,
        "excluded_adapters": ", ".join(hardware.get("excluded_adapters") or []) or None,
    }
