"""Centralised logging, switched by `DEBUG` in the environment.

The specification asks that debug logging be *disabled*, not merely filtered,
when `DEBUG=false`. That distinction is the whole design here: a filtered logger
still formats the record, still walks the handler chain, and still pays the cost
of every `logger.debug(f"...")` f-string the caller evaluated before the call.
When DEBUG is off this module **does not install** the debug handlers at all and
raises the root level, so those records are discarded at the cheapest possible
point.

Two more properties that matter for a long-running batch process:

**Database logging never blocks the pipeline.** The DB handler sits behind a
`QueueHandler`/`QueueListener` pair, so a slow or unreachable database delays a
background thread and nothing else. A document must never fail because logging
was busy.

**Files rotate and expire.** A thousand-document batch at DEBUG produces a lot of
output; unbounded logs would quietly fill the disk that the OCR cache also needs.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from . import paths

#: Fields already on every LogRecord; anything else a caller passes via `extra`
#: is treated as structured context and included in the payload.
_STANDARD = frozenset(
    "name msg args levelname levelno pathname filename module exc_info exc_text "
    "stack_info lineno funcName created msecs relativeCreated thread threadName "
    "processName process taskName message asctime".split()
)

_listener: logging.handlers.QueueListener | None = None
_configured = False


class StructuredFormatter(logging.Formatter):
    """Human-readable line plus any structured context the caller attached.

    Not JSON by default: an operator reading a log during an incident benefits
    more from an aligned, scannable line than from machine syntax. Structured
    fields are appended as `key=value`, and `json_payload=True` switches the whole
    record to JSON for ingestion tooling.
    """

    def __init__(self, *, json_payload: bool = False) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.json_payload = json_payload

    def format(self, record: logging.LogRecord) -> str:
        context = {k: v for k, v in record.__dict__.items()
                   if k not in _STANDARD and not k.startswith("_")}

        if self.json_payload:
            payload: dict[str, Any] = {
                "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                **context,
            }
            if record.exc_info:
                payload["exception"] = self.formatException(record.exc_info)
            return json.dumps(payload, default=str)

        # `where` names the function, not only the module. During an incident
        # the question is almost always "which code path", and a logger name
        # alone answers that only as far as the file.
        where = f"{record.name}.{record.funcName}()"
        # formatTime ignores self.datefmt unless it is passed: the base class
        # supplies it from format(), and this formatter does not call that.
        line = (f"[{self.formatTime(record, self.datefmt)}] {record.levelname:<8} "
                f"{where:<52} {record.getMessage()}")
        if context:
            line += "  " + " ".join(f"{k}={_short(v)}" for k, v in sorted(context.items()))
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _short(value: Any, limit: int = 120) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit - 1] + "…"


class DatabaseHandler(logging.Handler):
    """Writes records to the `logs` table.

    Only installed when DEBUG is on. Failures are swallowed deliberately: if the
    database is unreachable, that is already being reported through the file and
    console handlers, and a logging handler that raises would take down the very
    code trying to report the problem.
    """

    def __init__(self, session_factory: Callable[[], Any], level: int = logging.INFO) -> None:
        super().__init__(level)
        self.session_factory = session_factory
        self._failed = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._failed:
            return
        try:
            from .db.models import LogEntry

            session = self.session_factory()
            try:
                session.add(LogEntry(
                    level=record.levelname,
                    logger=record.name[:120],
                    message=self.format(record),
                    batch_id=getattr(record, "batch_id", None),
                    document_id=getattr(record, "document_pk", None),
                ))
                session.commit()
            finally:
                session.close()
        except Exception:  # noqa: BLE001
            # Stop trying after the first failure rather than thrashing a broken
            # connection once per log line.
            self._failed = True


def _console_level(explicit: int | str | None, debug: bool) -> int:
    """Level for the terminal handler, in order of precedence.

    Explicit argument, then `SALEDEED_LOG_CONSOLE`, then DEBUG when debugging,
    then INFO. Anything unrecognised falls back to INFO rather than raising -
    a typo in an environment variable must not stop the application starting.
    """
    if explicit is not None:
        if isinstance(explicit, int):
            return explicit
        return logging.getLevelNamesMapping().get(explicit.upper(), logging.INFO)

    name = os.environ.get("SALEDEED_LOG_CONSOLE", "").strip().upper()
    if name:
        return logging.getLevelNamesMapping().get(name, logging.INFO)
    return logging.DEBUG if debug else logging.INFO


def configure(
    *,
    debug: bool | None = None,
    log_dir: str | Path | None = None,
    retention_days: int = 30,
    session_factory: Callable[[], Any] | None = None,
    console: bool = True,
    console_level: int | str | None = None,
    json_files: bool = False,
    app_name: str = "saledeed",
) -> logging.Logger:
    """Install handlers once. Returns the application root logger.

    `debug` defaults to the `SALEDEED_DEBUG` environment variable.
    """
    global _configured, _listener
    if debug is None:
        debug = os.environ.get("SALEDEED_DEBUG", "false").strip().lower() in (
            "1", "true", "yes", "on")

    root = logging.getLogger(app_name)
    if _configured:
        return root

    # `./logs` was the default and it is relative to the *working directory*,
    # so the logs landed wherever the process happened to be started from - a
    # stray `logs/` beside whatever the operator had `cd`-ed into, while the
    # application looked for them under `runtime/`. Absolute, from `paths`.
    directory = Path(log_dir or os.environ.get("SALEDEED_LOG_DIR")
                     or paths.LOG_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    # Root level gates everything. At INFO, a `logger.debug(...)` call is
    # rejected by `isEnabledFor` before any handler is consulted.
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    root.handlers.clear()
    root.propagate = False

    formatter = StructuredFormatter(json_payload=json_files)

    if console:
        stream = logging.StreamHandler(sys.stderr)
        # This used to sit at WARNING, on the reasoning that the desktop UI is
        # the status display and the terminal should stay quiet. That is right
        # for the packaged application and wrong for everything else: running a
        # service from a terminal and seeing nothing reads as a dead process,
        # and the AI server is run exactly that way while it is being worked on.
        #
        # INFO by default. `SALEDEED_LOG_CONSOLE=warning` restores the quiet
        # behaviour, and the desktop shell asks for it explicitly.
        stream.setLevel(_console_level(console_level, debug))
        stream.setFormatter(StructuredFormatter())
        root.addHandler(stream)

    info_file = logging.handlers.RotatingFileHandler(
        directory / f"{app_name}.log", maxBytes=8 * 1024 * 1024,
        backupCount=5, encoding="utf-8")
    info_file.setLevel(logging.INFO)
    info_file.setFormatter(formatter)
    root.addHandler(info_file)

    if debug:
        # Debug output is separate and time-rotated: it is voluminous, and
        # keeping it out of the main log keeps that log readable.
        debug_file = logging.handlers.TimedRotatingFileHandler(
            directory / f"{app_name}.debug.log", when="midnight",
            backupCount=7, encoding="utf-8")
        debug_file.setLevel(logging.DEBUG)
        debug_file.setFormatter(formatter)
        root.addHandler(debug_file)

        if session_factory is not None:
            # Behind a queue so a slow database cannot stall a stage worker.
            record_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=10_000)
            db_handler = DatabaseHandler(session_factory)
            db_handler.setFormatter(StructuredFormatter())
            _listener = logging.handlers.QueueListener(
                record_queue, db_handler, respect_handler_level=True)
            _listener.daemon = True
            _listener.start()
            root.addHandler(logging.handlers.QueueHandler(record_queue))

    _configured = True
    # Context propagation is on by default; a caller should not have to remember
    # a second setup step for batch/document identifiers to appear.
    install_context_filter(app_name)
    purge_old_logs(directory, retention_days)

    root.info("logging configured", extra={
        "debug": debug, "dir": str(directory.resolve()),
        "handlers": len(root.handlers),
        "db_logging": bool(debug and session_factory is not None),
    })
    return root


def shutdown() -> None:
    """Flush and stop the queue listener. Call on graceful exit."""
    global _listener, _configured
    if _listener is not None:
        try:
            _listener.stop()
        except Exception:  # noqa: BLE001
            pass
        _listener = None
    logging.shutdown()
    _configured = False


def purge_old_logs(directory: Path, retention_days: int) -> int:
    """Delete rotated logs past retention. Returns the count removed."""
    if retention_days <= 0:
        return 0
    cutoff = time.time() - retention_days * 86400
    removed = 0
    for path in directory.glob("*.log*"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def get_logger(name: str) -> logging.Logger:
    """Child logger. Always use this rather than `logging.getLogger`."""
    return logging.getLogger(f"saledeed.{name}")


class LogContext:
    """Attach batch/document identifiers to every record inside a block.

    Stage workers run concurrently, so an interleaved log is unreadable without
    knowing which document each line belongs to. Thread-local, so concurrent
    workers never see each other's context.
    """

    _local = threading.local()

    def __init__(self, **fields: Any) -> None:
        self.fields = fields
        self._previous: dict[str, Any] = {}

    def __enter__(self) -> LogContext:
        self._previous = dict(getattr(self._local, "fields", {}))
        merged = {**self._previous, **self.fields}
        self._local.fields = merged
        return self

    def __exit__(self, *exc: object) -> None:
        self._local.fields = self._previous

    @classmethod
    def current(cls) -> dict[str, Any]:
        return dict(getattr(cls._local, "fields", {}))


class _ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for key, value in LogContext.current().items():
            if not hasattr(record, key):
                setattr(record, key, value)
        return True


def install_context_filter(app_name: str = "saledeed") -> None:
    """Make LogContext fields appear on records automatically.

    Attached to every **handler**, not to the logger. A filter on a logger only
    sees records logged directly to it - records propagated up from child
    loggers such as `saledeed.pipeline` bypass it entirely. Handler filters run
    on everything that reaches the handler, which is what is wanted here.
    """
    context_filter = _ContextFilter()
    for handler in logging.getLogger(app_name).handlers:
        if not any(isinstance(f, _ContextFilter) for f in handler.filters):
            handler.addFilter(context_filter)


def timed(logger: logging.Logger, message: str, **fields: Any):  # noqa: ANN201
    """Context manager logging duration at DEBUG, and failures at ERROR."""

    class _Timer:
        def __enter__(self) -> _Timer:
            self.start = time.monotonic()
            return self

        def __exit__(self, exc_type: type | None, exc: BaseException | None,
                     tb: object) -> bool:
            elapsed = round(time.monotonic() - self.start, 3)
            if exc_type is None:
                logger.debug(message, extra={**fields, "seconds": elapsed})
            else:
                logger.error(f"{message} failed", exc_info=(exc_type, exc, tb),  # type: ignore[arg-type]
                             extra={**fields, "seconds": elapsed})
            return False

    return _Timer()


def summarise(app_name: str = "saledeed") -> dict[str, Any]:
    """What is actually installed - shown on the Settings page."""
    root = logging.getLogger(app_name)
    return {
        "level": logging.getLevelName(root.level),
        "debug_enabled": root.level <= logging.DEBUG,
        "handlers": [
            {"type": type(h).__name__,
             "level": logging.getLevelName(h.level),
             "target": getattr(h, "baseFilename", "-")}
            for h in root.handlers
        ],
    }
