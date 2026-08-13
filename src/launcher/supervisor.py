"""Child-process supervision: start, health-watch, restart, and clean shutdown.

The hard problem here is Windows shutdown. The launcher starts the AI server,
which starts `llama-server.exe`, which holds several GB of VRAM. Calling
`terminate()` on the AI server kills only that process - `llama-server.exe` is
reparented and survives, keeping the GPU memory and the port. The next launch
then fails to bind, and the user has a phantom process they have to find in Task
Manager.

The fix is a Windows **Job Object** with `KILL_ON_JOB_CLOSE`. Every child is
assigned to the job, and the kernel terminates the whole tree when the job handle
closes - including when the launcher itself is killed, which no atexit handler
can cover. `taskkill /T` is kept as the fallback path for the graceful case and
for non-Windows.
"""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

IS_WINDOWS = os.name == "nt"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
CREATE_NEW_PROCESS_GROUP = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


class ProcessJob:
    """Windows Job Object that kills every assigned process when closed.

    A no-op on other platforms, where the process-group path in `stop_all` does
    the same job.
    """

    def __init__(self) -> None:
        self.handle = None
        if not IS_WINDOWS:
            return
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                return

            # JOBOBJECT_EXTENDED_LIMIT_INFORMATION, laid out by hand to avoid a
            # dependency on pywin32 for one struct.
            class IoCounters(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in
                            ("ReadOperationCount", "WriteOperationCount",
                             "OtherOperationCount", "ReadTransferCount",
                             "WriteTransferCount", "OtherTransferCount")]

            class BasicLimits(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", ctypes.c_uint32),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", ctypes.c_uint32),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", ctypes.c_uint32),
                    ("SchedulingClass", ctypes.c_uint32),
                ]

            class ExtendedLimits(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", BasicLimits),
                    ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            info = ExtendedLimits()
            info.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
            kernel32.SetInformationJobObject(
                handle, 9, ctypes.byref(info), ctypes.sizeof(info))
            self.handle = handle
            self._kernel32 = kernel32
        except Exception:  # noqa: BLE001 - supervision must degrade, not crash
            self.handle = None

    def assign(self, pid: int) -> bool:
        if not self.handle:
            return False
        try:
            # PROCESS_SET_QUOTA | PROCESS_TERMINATE
            proc = self._kernel32.OpenProcess(0x0100 | 0x0001, False, pid)
            if not proc:
                return False
            try:
                return bool(self._kernel32.AssignProcessToJobObject(self.handle, proc))
            finally:
                self._kernel32.CloseHandle(proc)
        except Exception:  # noqa: BLE001
            return False

    def close(self) -> None:
        if self.handle:
            try:
                self._kernel32.CloseHandle(self.handle)
            finally:
                self.handle = None


@dataclass
class Service:
    """A supervised child process."""

    name: str
    argv: list[str]
    cwd: Path
    health_url: str | None = None
    #: Restarts attempted before giving up. Zero disables restart.
    max_restarts: int = 3
    log_path: Path | None = None
    #: Extra environment for the child, merged over the parent's. Needed because
    #: a child launched with `-m` resolves the module against its own sys.path,
    #: not ours: the packages live in `src/`, which is not the working directory.
    env: dict[str, str] = field(default_factory=dict)

    process: subprocess.Popen | None = field(default=None, init=False)
    restarts: int = field(default=0, init=False)
    _log_handle: object = field(default=None, init=False)

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, job: ProcessJob | None = None) -> None:
        stdout = subprocess.DEVNULL
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Append, so a restart does not discard the log that explains why the
            # previous attempt died.
            self._log_handle = open(self.log_path, "a", encoding="utf-8",
                                    errors="replace", buffering=1)
            stdout = self._log_handle  # type: ignore[assignment]

        flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP if IS_WINDOWS else 0
        self.process = subprocess.Popen(
            self.argv, cwd=str(self.cwd), stdout=stdout,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            creationflags=flags,
            # PYTHONUNBUFFERED so the log is useful while the process is running,
            # not only after it exits.
            env={**os.environ, "PYTHONUNBUFFERED": "1", **self.env},
        )
        if job:
            job.assign(self.process.pid)

    def stop(self, timeout: float = 10.0) -> None:
        """Ask politely, then insist.

        On Windows the tree is killed with `taskkill /T`: the AI server spawns
        `llama-server.exe`, and terminating only the parent leaves it holding
        VRAM and the port.
        """
        if not self.alive or self.process is None:
            self._close_log()
            return
        pid = self.process.pid
        try:
            self.process.terminate()
            self.process.wait(timeout=timeout)
        except Exception:  # noqa: BLE001
            if IS_WINDOWS:
                try:
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                                   capture_output=True, timeout=15,
                                   creationflags=CREATE_NO_WINDOW)
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    self.process.kill()
                except Exception:  # noqa: BLE001
                    pass
        finally:
            self._close_log()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            try:
                self._log_handle.close()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            self._log_handle = None


def port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def wait_for_http(url: str, timeout_s: float, *, poll_s: float = 0.5,
                  on_wait=None, service: Service | None = None) -> tuple[bool, str]:
    """Wait until the endpoint answers at all.

    Deliberately *not* waiting for `ready: true`. The model takes 30-60 s to load
    and the interface is designed to open while that happens - it shows LOADING
    and gates the actions that need inference. Blocking here would put that time
    back into the startup path for no benefit.

    Returns as soon as the process dies, rather than burning the full timeout on
    a server that has already exited.
    """
    deadline = time.time() + timeout_s
    last = "no response"
    while time.time() < deadline:
        if service is not None and not service.alive:
            code = service.process.returncode if service.process else "?"
            return False, f"process exited with code {code}"
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 500:
                    return True, f"responded {response.status}"
                last = f"HTTP {response.status}"
        except urllib.error.HTTPError as exc:
            # A 4xx still proves something is listening and routing.
            return True, f"responded {exc.code}"
        except Exception as exc:  # noqa: BLE001
            last = type(exc).__name__
        if on_wait:
            on_wait(max(0.0, deadline - time.time()))
        time.sleep(poll_s)
    return False, last


class Supervisor:
    """Owns every child process and the thread that watches them."""

    def __init__(self, log=None) -> None:
        self.services: list[Service] = []
        self.job = ProcessJob()
        self.log = log
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def add(self, service: Service) -> Service:
        self.services.append(service)
        return service

    def start(self, service: Service) -> None:
        service.start(self.job)
        if self.log:
            self.log.info("service started",
                          extra={"service": service.name,
                                 "pid": service.process.pid if service.process else None})

    def watch(self, interval_s: float = 5.0) -> None:
        """Restart any service that dies, up to its own limit."""
        if self._thread:
            return

        def loop() -> None:
            while not self._stop.wait(interval_s):
                for service in self.services:
                    if service.alive or service.max_restarts == 0:
                        continue
                    if service.restarts >= service.max_restarts:
                        continue
                    service.restarts += 1
                    if self.log:
                        self.log.warning(
                            "service died - restarting",
                            extra={"service": service.name,
                                   "attempt": service.restarts,
                                   "of": service.max_restarts})
                    try:
                        service.start(self.job)
                    except Exception as exc:  # noqa: BLE001
                        if self.log:
                            self.log.error("restart failed",
                                           extra={"service": service.name,
                                                  "error": str(exc)})

        self._thread = threading.Thread(target=loop, name="supervisor",
                                        daemon=True)
        self._thread.start()

    def stop_all(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        # Reverse order: dependants before dependencies.
        for service in reversed(self.services):
            if service.alive:
                if self.log:
                    self.log.info("stopping service", extra={"service": service.name})
                service.stop(timeout=timeout)
        self.job.close()
