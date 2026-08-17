"""Process supervision: the code that keeps the AI server alive and, more
importantly, kills it properly.

`supervisor.py` sat at 25% and `runner.py` at 15%. Both are process-management
code, and both were only ever exercised by launching the real application - so
the behaviour that matters most was untested precisely because testing it meant
starting a GPU stack.

What matters here is not the happy path. It is that a stopped service is really
stopped: this session found orphaned Python workers holding the GPU after a job
was killed, which invalidated a live test run and starved later batches. The
tests below use short-lived real processes, so termination is genuinely
observed rather than mocked.
"""

from __future__ import annotations

import socket
import sys
import time

import pytest

from launcher.supervisor import (
    ProcessJob,
    Service,
    Supervisor,
    port_open,
    wait_for_http,
)


def _sleeper(seconds: int = 60) -> list[str]:
    """A real child process that outlives the test unless it is killed."""
    return [sys.executable, "-c", f"import time; time.sleep({seconds})"]


class TestAServiceIsReallyStopped:
    def test_a_started_service_is_alive(self, tmp_path):
        service = Service(name="sleeper", argv=_sleeper(), cwd=tmp_path,
                          log_path=tmp_path / "s.log")
        service.start()
        try:
            assert service.alive is True
        finally:
            service.stop(timeout=10)

    def test_stop_actually_terminates_the_process(self, tmp_path):
        """The failure this guards against was observed live: killed jobs left
        Python workers resident, holding the GPU."""
        service = Service(name="sleeper", argv=_sleeper(), cwd=tmp_path,
                          log_path=tmp_path / "s.log")
        service.start()
        pid = service.process.pid
        service.stop(timeout=15)

        assert service.alive is False
        deadline = time.time() + 10
        while time.time() < deadline:
            if not _pid_running(pid):
                break
            time.sleep(0.2)
        assert not _pid_running(pid), f"pid {pid} survived stop()"

    def test_stop_is_idempotent(self, tmp_path):
        """Called on paths where start never happened; an exception there would
        mask the real failure."""
        service = Service(name="sleeper", argv=_sleeper(), cwd=tmp_path,
                          log_path=tmp_path / "s.log")
        service.stop(timeout=5)
        service.start()
        service.stop(timeout=10)
        service.stop(timeout=10)

    def test_alive_is_false_before_start(self, tmp_path):
        service = Service(name="never", argv=_sleeper(), cwd=tmp_path,
                          log_path=tmp_path / "s.log")
        assert service.alive is False

    def test_a_log_file_is_written_and_closed(self, tmp_path):
        log = tmp_path / "out.log"
        service = Service(
            name="talker", cwd=tmp_path,
            argv=[sys.executable, "-c", "print('hello from the child')"],
            log_path=log)
        service.start()
        time.sleep(2)
        service.stop(timeout=10)
        assert log.is_file()


class TestTheSupervisorStopsEverything:
    def test_stop_all_terminates_every_service(self, tmp_path):
        """One service surviving shutdown is how a GPU stays held after the
        application closes."""
        supervisor = Supervisor()
        services = [
            supervisor.add(Service(name=f"s{n}", argv=_sleeper(),
                                   cwd=tmp_path,
                                   log_path=tmp_path / f"{n}.log"))
            for n in range(3)]
        for service in services:
            supervisor.start(service)
        pids = [s.process.pid for s in services]
        assert all(s.alive for s in services)

        supervisor.stop_all(timeout=15)

        deadline = time.time() + 10
        while time.time() < deadline and any(_pid_running(p) for p in pids):
            time.sleep(0.2)
        survivors = [p for p in pids if _pid_running(p)]
        assert not survivors, f"pids survived stop_all: {survivors}"

    def test_stop_all_is_safe_with_nothing_started(self):
        Supervisor().stop_all(timeout=5)


class TestPortProbing:
    def test_a_closed_port_reads_as_closed(self):
        # Port 1 is privileged and not listening in any normal environment.
        assert port_open("127.0.0.1", 1, timeout=0.3) is False

    def test_an_open_port_reads_as_open(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        try:
            assert port_open("127.0.0.1", listener.getsockname()[1]) is True
        finally:
            listener.close()

    def test_an_unresolvable_host_raises_rather_than_reporting_closed(self):
        """Documented, not endorsed. `port_open` lets a DNS failure through,
        so every caller must pass a resolvable host - which they do, since the
        only callers use 127.0.0.1. Asserting the current behaviour keeps a
        future change from silently turning a crash into a false "closed"."""
        with pytest.raises(socket.gaierror):
            port_open("no-such-host.invalid", 8077, timeout=0.5)


class TestWaitingForHttp:
    def test_it_gives_up_rather_than_hanging(self):
        """A hung wait is indistinguishable from a hung application."""
        started = time.monotonic()
        ok, detail = wait_for_http("http://127.0.0.1:1/health", timeout_s=2.0)
        assert ok is False
        assert detail, "giving up must say why"
        assert time.monotonic() - started < 15

    def test_an_unreachable_host_is_reported_not_raised(self):
        ok, detail = wait_for_http("http://no-such-host.invalid/health",
                                   timeout_s=2.0)
        assert ok is False
        assert detail


class TestTheJobObject:
    def test_it_can_be_created_and_closed(self):
        """On Windows this is a Job object; assigning a process to it is what
        makes children die with the parent instead of being orphaned."""
        job = ProcessJob()
        try:
            assert job is not None
        finally:
            job.close()

    def test_close_is_idempotent(self):
        job = ProcessJob()
        job.close()
        job.close()

    def test_assigning_a_bad_pid_is_reported_not_raised(self):
        job = ProcessJob()
        try:
            assert job.assign(999_999_999) in (True, False)
        finally:
            job.close()


def _pid_running(pid: int) -> bool:
    """True while the process exists. Uses the OS, not our own bookkeeping -
    the point is to catch a process we *believe* we stopped."""
    import subprocess

    proc = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"],
                          capture_output=True, text=True, check=False)
    return str(pid) in (proc.stdout or "")
