"""Backup, archival and retention.

Three jobs, on different clocks:

  **Yearly**   dump the database to `/backup`, archive the previous year, then
               delete that year's rows.
  **30 days**  drop cached OCR text. It is regenerable from the PDFs and is
               never backed up, so it is the cheapest thing to expire.
  **Retention**rotate old log files.

Ordering matters and is enforced: the archive is written and *verified* before
anything is purged. A purge that runs after a failed dump destroys data with no
copy, which is the one outcome this module exists to prevent.

`pg_dump` is located from the running server rather than assumed to be on PATH -
the PostgreSQL installer does not always add it, and a backup that silently does
not run is worse than one that reports it cannot.
"""

from __future__ import annotations

import gzip
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.engine import make_url

from .logging_setup import get_logger

log = get_logger("backup")

#: Where `pg_dump` usually lives on Windows when it is not on PATH.
_WINDOWS_HINTS = (
    r"C:\Program Files\PostgreSQL\{v}\bin",
    r"C:\Program Files (x86)\PostgreSQL\{v}\bin",
)
_VERSIONS = ("18", "17", "16", "15", "14", "13")


@dataclass
class BackupResult:
    ok: bool
    path: Path | None = None
    bytes_written: int = 0
    duration_s: float = 0.0
    error: str = ""
    detail: str = ""


@dataclass
class RetentionResult:
    ocr_pages_deleted: int = 0
    log_rows_deleted: int = 0
    log_files_deleted: int = 0
    batches_purged: int = 0
    archived: Path | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def find_pg_dump() -> str | None:
    """Locate `pg_dump`, checking PATH then the standard install directories."""
    found = shutil.which("pg_dump")
    if found:
        return found
    if platform.system() == "Windows":
        for template in _WINDOWS_HINTS:
            for version in _VERSIONS:
                candidate = Path(template.format(v=version)) / "pg_dump.exe"
                if candidate.is_file():
                    return str(candidate)
    return None


def dump_database(
    dsn: str,
    output_dir: str | Path,
    *,
    label: str | None = None,
    compress: bool = True,
    timeout_s: float = 3600.0,
) -> BackupResult:
    """Write a `pg_dump` archive. Never raises; failure is reported."""
    started = datetime.now(UTC)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    binary = find_pg_dump()
    if binary is None:
        return BackupResult(
            ok=False,
            error="pg_dump not found on PATH or in the standard PostgreSQL "
                  "install directories. Backups cannot run without it.")

    url = make_url(dsn)
    if not (url.get_backend_name() or "").startswith("postgresql"):
        return BackupResult(ok=False, error=f"unsupported backend: {url.get_backend_name()}")

    stamp = label or started.strftime("%Y%m%d_%H%M%S")
    target = directory / f"saledeed_{stamp}.sql{'.gz' if compress else ''}"

    # Password via environment, never on the command line - argv is visible to
    # any other process on the machine.
    env = dict(os.environ)
    if url.password:
        env["PGPASSWORD"] = str(url.password)

    command = [
        binary,
        "--host", url.host or "localhost",
        "--port", str(url.port or 5432),
        "--username", url.username or "postgres",
        "--dbname", url.database or "saledeed",
        "--no-password",
        # Column names are written explicitly so a restore survives future
        # column reordering.
        "--column-inserts",
        "--no-owner", "--no-privileges",
    ]

    try:
        if compress:
            with subprocess.Popen(command, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=env) as proc:
                with gzip.open(target, "wb") as out:
                    assert proc.stdout is not None
                    shutil.copyfileobj(proc.stdout, out)
                _, stderr = proc.communicate(timeout=timeout_s)
                code = proc.returncode
        else:
            with target.open("wb") as out:
                proc = subprocess.run(command, stdout=out, stderr=subprocess.PIPE,
                                      env=env, timeout=timeout_s, check=False)
                stderr, code = proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        target.unlink(missing_ok=True)
        return BackupResult(ok=False, error=f"pg_dump timed out after {timeout_s:.0f}s")
    except Exception as exc:  # noqa: BLE001
        target.unlink(missing_ok=True)
        return BackupResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    duration = (datetime.now(UTC) - started).total_seconds()

    if code != 0:
        target.unlink(missing_ok=True)
        message = (stderr or b"").decode("utf-8", "replace").strip()[:300]
        return BackupResult(ok=False, duration_s=duration,
                            error=f"pg_dump exited {code}: {message}")

    size = target.stat().st_size if target.is_file() else 0
    # An empty or near-empty archive means the dump produced nothing usable.
    # Treating that as success would be the dangerous case.
    if size < 1024:
        target.unlink(missing_ok=True)
        return BackupResult(ok=False, duration_s=duration,
                            error=f"archive was only {size} bytes - treating as failed")

    log.info("database dumped", extra={
        "path": str(target), "bytes": size, "seconds": round(duration, 1)})
    return BackupResult(ok=True, path=target, bytes_written=size,
                        duration_s=duration,
                        detail=f"{size / 1024**2:.1f} MB in {duration:.0f}s")


def verify_archive(path: Path) -> tuple[bool, str]:
    """Sanity-check an archive before anything is deleted on its strength."""
    if not path.is_file():
        return False, "archive missing"
    if path.stat().st_size < 1024:
        return False, f"archive is only {path.stat().st_size} bytes"
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as fh:  # type: ignore[operator]
            head = fh.read(4096).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return False, f"archive unreadable: {type(exc).__name__}: {exc}"
    if "PostgreSQL database dump" not in head:
        return False, "archive does not look like a pg_dump output"
    return True, "archive verified"


def run_retention(
    session_factory: Any,
    *,
    dsn: str,
    backup_dir: str | Path = "./backup",
    log_dir: str | Path = "./logs",
    ocr_ttl_days: int = 30,
    log_retention_days: int = 30,
    retain_years: int = 1,
    purge_old_years: bool = False,
    now: datetime | None = None,
) -> RetentionResult:
    """Run all retention work in the safe order.

    `purge_old_years` defaults to **False**. Deleting a year of extracted deed
    data is irreversible, so it is opt-in and only ever proceeds after an archive
    has been written *and* verified.
    """
    from .db.engine import session_scope
    from .db.repositories import UnitOfWork
    from .logging_setup import purge_old_logs

    moment = now or datetime.now(UTC)
    result = RetentionResult()

    # 1. Expire the OCR cache. Cheap, safe, regenerable from the PDFs.
    try:
        with session_scope(session_factory) as session:
            result.ocr_pages_deleted = UnitOfWork(session).ocr.purge_expired(ocr_ttl_days)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"ocr purge: {type(exc).__name__}: {exc}")

    # 2. Trim logs, in the database and on disk.
    try:
        with session_scope(session_factory) as session:
            result.log_rows_deleted = UnitOfWork(session).maintenance.purge_logs(
                log_retention_days)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"log purge: {type(exc).__name__}: {exc}")

    try:
        result.log_files_deleted = purge_old_logs(Path(log_dir), log_retention_days)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"log files: {type(exc).__name__}: {exc}")

    # 3. Yearly archive, then purge - strictly in that order.
    cutoff_year = moment.year - retain_years
    try:
        with session_scope(session_factory) as session:
            older = UnitOfWork(session).maintenance.documents_before(cutoff_year)
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"year scan: {type(exc).__name__}: {exc}")
        return result

    if older == 0:
        log.debug("no data older than the retention window", extra={"year": cutoff_year})
        return result

    backup = dump_database(dsn, backup_dir, label=f"archive_{cutoff_year}")
    if not backup.ok:
        result.errors.append(f"archive failed, nothing purged: {backup.error}")
        log.error("yearly archive failed - purge skipped", extra={"error": backup.error})
        return result

    verified, detail = verify_archive(backup.path)  # type: ignore[arg-type]
    if not verified:
        result.errors.append(f"archive unverified, nothing purged: {detail}")
        log.error("archive verification failed - purge skipped", extra={"detail": detail})
        return result

    result.archived = backup.path

    if not purge_old_years:
        log.info("archive written; purge not requested", extra={
            "documents_older_than": older, "year": cutoff_year,
            "archive": str(backup.path)})
        return result

    try:
        with session_scope(session_factory) as session:
            result.batches_purged = UnitOfWork(session).maintenance.purge_year(cutoff_year)
        log.info("year purged after verified archive", extra={
            "year": cutoff_year, "batches": result.batches_purged,
            "archive": str(backup.path)})
    except Exception as exc:  # noqa: BLE001
        result.errors.append(f"year purge: {type(exc).__name__}: {exc}")

    return result


class RetentionScheduler:
    """Runs retention on a timer inside the application process.

    A daemon thread rather than an OS scheduled task: the application is the only
    thing that knows whether a batch is mid-flight, and a dump taken during heavy
    write activity is both slower and larger.
    """

    def __init__(
        self,
        session_factory: Any,
        dsn: str,
        *,
        backup_dir: str | Path = "./backup",
        log_dir: str | Path = "./logs",
        interval_hours: float = 24.0,
        is_busy: Any = None,
        **retention_kwargs: Any,
    ) -> None:
        import threading

        self.session_factory = session_factory
        self.dsn = dsn
        self.backup_dir = backup_dir
        self.log_dir = log_dir
        self.interval_s = max(60.0, interval_hours * 3600)
        #: Callable returning True while a batch is processing; retention waits.
        self.is_busy = is_busy
        self.retention_kwargs = retention_kwargs
        self.last_result: RetentionResult | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import threading

        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="retention", daemon=True)
        self._thread.start()
        log.info("retention scheduler started", extra={"interval_hours": self.interval_s / 3600})

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def run_now(self) -> RetentionResult:
        self.last_result = run_retention(
            self.session_factory, dsn=self.dsn, backup_dir=self.backup_dir,
            log_dir=self.log_dir, **self.retention_kwargs)
        return self.last_result

    def _loop(self) -> None:
        # Wait one interval before the first run: startup is the worst moment to
        # begin a long dump.
        while not self._stop.wait(self.interval_s):
            if self.is_busy is not None:
                try:
                    if self.is_busy():
                        log.debug("retention deferred - batch in progress")
                        continue
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.run_now()
            except Exception as exc:  # noqa: BLE001
                log.error("retention run failed", exc_info=exc)


def restore_instructions(archive: Path, dsn: str) -> str:
    """How to restore. Printed rather than executed - restores overwrite data."""
    url = make_url(dsn)
    unzip = f"gunzip -c {archive.name} | " if archive.suffix == ".gz" else f"< {archive.name} "
    return (
        f"To restore {archive.name}:\n"
        f"  createdb -h {url.host or 'localhost'} -p {url.port or 5432} "
        f"-U {url.username or 'postgres'} {url.database or 'saledeed'}_restored\n"
        f"  {unzip}psql -h {url.host or 'localhost'} -p {url.port or 5432} "
        f"-U {url.username or 'postgres'} -d {url.database or 'saledeed'}_restored\n\n"
        "Restore into a NEW database and verify before switching. Restoring over "
        "the live database replaces current data irreversibly."
    )
