"""Startup orchestration - the sequence, the console output, and the log.

Startup order:

    Project validation -> Configuration -> Database -> Migrations
    -> Models -> AI service -> Health check -> Desktop window -> Ready

Two deviations from a naive reading of that order, both deliberate:

**The health check waits for the endpoint, not for the model.** Loading a 4-bit
Gemma into VRAM takes 30-60 s. The interface already knows how to open against a
server that is still loading - it shows LOADING and disables the actions that
need inference. Blocking the window on a fully warm model would add that minute
back to every start for nothing.

**A failed AI service does not abort the launch.** Browsing completed batches,
reviewing flagged documents and exporting CSV all work without inference. The
window opens with those capabilities enabled and processing greyed out, which is
more useful than a dialog box and no application.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

from .config import LauncherConfig, build_config, find_root
from .steps import PREFLIGHT, Outcome, Result
from .supervisor import Service, Supervisor, port_open, wait_for_http

log = logging.getLogger("launcher")

#: Console styling. Disabled when the stream is redirected, so a log file does
#: not fill with escape sequences.
_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
_MARK = {Outcome.OK: ("[ ok ]", "\033[32m"),
         Outcome.WARN: ("[warn]", "\033[33m"),
         Outcome.FAIL: ("[FAIL]", "\033[31m")}


def _emit(label: str, result: Result, elapsed_ms: int) -> None:
    mark, colour = _MARK[result.outcome]
    if _COLOUR:
        mark = f"{colour}{mark}\033[0m"
    timing = f"{elapsed_ms:>5} ms" if elapsed_ms >= 1 else "      "
    print(f"  {mark} {label:<20} {result.detail}   {timing}", flush=True)
    if result.remedy and result.outcome is not Outcome.OK:
        for line in result.remedy.splitlines():
            print(f"         -> {line}", flush=True)


def banner(cfg: LauncherConfig) -> None:
    print()
    print("  Sale Deed AI")
    print(f"  {cfg.root}")
    print("  " + "-" * 62)


def child_env(root: Path) -> dict[str, str]:
    """Environment for a child launched with `-m`.

    `python -m ai_server.server` resolves the module against the *child's*
    sys.path, which begins at its working directory. That directory is the
    project root and the packages live in `src/`, so without `PYTHONPATH` the
    child dies immediately with `No module named 'ai_server'`.

    The failure is invisible from the parent: the supervisor sees an exit code,
    restarts three times, and the only thing the operator ever sees is the UI
    reporting the AI server offline. See R-033.
    """
    src = str(root / "src")
    existing = os.environ.get("PYTHONPATH", "")
    return {"PYTHONPATH": f"{src}{os.pathsep}{existing}" if existing else src}


class _LogRelay(threading.Thread):
    """Echo a supervised child's log file into this terminal, as it is written.

    The AI server is started with CREATE_NO_WINDOW and its output redirected to
    a file, which is right for the packaged application - nobody wants a stray
    console - and useless while debugging, because the process that has the
    interesting logs is the one with no way to show them.

    Following the file rather than piping the handle keeps the supervisor's
    restart logic untouched: it still owns the child and the file, and this only
    reads. If the relay dies, the child does not notice.
    """

    def __init__(self, path: Path, prefix: str = "ai") -> None:
        super().__init__(daemon=True, name=f"log-relay-{prefix}")
        self.path = path
        self.prefix = prefix
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        # Start at the end: the file is appended across restarts, and replaying
        # a previous run's log as though it were happening now is worse than
        # showing nothing.
        handle = None
        try:
            while not self._stop.is_set():
                if handle is None:
                    if not self.path.is_file():
                        self._stop.wait(0.5)
                        continue
                    handle = self.path.open("r", encoding="utf-8", errors="replace")
                    handle.seek(0, 2)
                line = handle.readline()
                if not line:
                    self._stop.wait(0.2)
                    continue
                print(f"  [{self.prefix}] {line.rstrip()}", flush=True)
        except Exception:  # noqa: BLE001 - a broken relay must not stop the app
            log.debug("log relay stopped", exc_info=True)
        finally:
            if handle is not None:
                handle.close()


class Launcher:
    def __init__(self, cfg: LauncherConfig) -> None:
        self.cfg = cfg
        self.log = None
        self.supervisor = Supervisor()
        self.results: dict[str, Result] = {}
        self.ai_service: Service | None = None
        #: Set by `main()` from `--verbose`. When on, the AI server's log is
        #: echoed into this terminal - the child has no console of its own.
        self.verbose = False
        self._relays: list[_LogRelay] = []

    # -- logging ---------------------------------------------------------

    def _configure_logging(self) -> None:
        """Launcher logging is file-only and independent of the application's.

        It has to survive the application failing to import, so it never touches
        the database handler.
        """
        import logging

        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("launcher")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        if not logger.handlers:
            handler = logging.FileHandler(
                self.cfg.log_dir / "launcher.log", encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)-7s %(message)s"))
            logger.addHandler(handler)
        self.log = logger
        self.supervisor.log = logger

    # -- phases ----------------------------------------------------------

    def preflight(self) -> bool:
        """Run every check, then decide. All of them run even after a failure -
        one pass should surface every problem, not send the user round a loop."""
        print("  Checking environment")
        failures: list[tuple[str, Result]] = []
        for label, step in PREFLIGHT:
            started = time.perf_counter()
            try:
                result = step(self.cfg)
            except Exception as exc:  # noqa: BLE001 - a step must not kill startup
                result = Result(Outcome.FAIL,
                                f"check raised {type(exc).__name__}: {exc}")
            elapsed = int((time.perf_counter() - started) * 1000)
            self.results[label] = result
            _emit(label, result, elapsed)
            if self.log:
                self.log.info("preflight %s: %s - %s", label,
                              result.outcome.value, result.detail)
            if result.outcome is Outcome.FAIL:
                failures.append((label, result))

        if failures:
            print()
            print(f"  Cannot start: {len(failures)} blocking problem(s).")
            return False
        return True

    def start_ai(self) -> bool:
        """Start the inference server unless one is already listening."""
        cfg = self.cfg
        if port_open(cfg.ai_host, cfg.ai_port):
            print(f"  [ ok ] AI service          already running on port {cfg.ai_port}")
            return True

        argv = [
            str(cfg.python), "-m", "ai_server.server",
            "--host", cfg.ai_host, "--port", str(cfg.ai_port),
            "--engine", cfg.engine,
            "--model", str(cfg.model_gguf),
            "--model-dir", str(cfg.model_dir),
            "--binary", str(cfg.llama_binary),
        ]
        if getattr(self, "verbose", False):
            # The child logs to a file because it has no console. Follow it.
            relay = _LogRelay(cfg.log_dir / "ai_server.out.log")
            relay.start()
            self._relays.append(relay)

        self.ai_service = self.supervisor.add(Service(
            name="ai-server", argv=argv, cwd=cfg.root, env=child_env(cfg.root),
            health_url=f"{cfg.ai_base_url}/health",
            log_path=cfg.log_dir / "ai_server.out.log",
            max_restarts=3,
        ))
        try:
            self.supervisor.start(self.ai_service)
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] AI service          could not start: {exc}")
            return False

        print("  [....] AI service          waiting for HTTP ...", end="\r", flush=True)
        ok, detail = wait_for_http(
            f"{cfg.ai_base_url}/health", cfg.ai_http_timeout_s,
            service=self.ai_service)
        if ok:
            print(f"  [ ok ] AI service          {detail} on port {cfg.ai_port}      ")
            if self.log:
                self.log.info("ai server up: %s", detail)
            return True

        print(f"  [warn] AI service          did not respond ({detail})            ")
        print(f"         -> see {cfg.log_dir / 'ai_server.out.log'}")
        print("         -> the window will open; processing stays disabled")
        if self.log:
            self.log.warning("ai server unavailable: %s", detail)
        return False

    def launch_ui(self) -> int:
        """Hand control to the desktop application in this process.

        Run in-process rather than as a child: Qt owns the event loop for the
        rest of the session, and a supervising parent would add a process whose
        only job is to wait.
        """
        os.environ.setdefault("SALEDEED_AI_URL", self.cfg.ai_base_url)
        if str(self.cfg.root) not in sys.path:
            sys.path.insert(0, str(self.cfg.root))

        from app.main import main as app_main

        self.supervisor.watch(interval_s=5.0)
        print("  " + "-" * 62)
        print("  Ready")
        print()
        return app_main()

    def run(self) -> int:
        cfg = self.cfg
        started = time.perf_counter()
        self._configure_logging()
        if self.log:
            self.log.info("launcher starting: root=%s python=%s",
                          cfg.root, cfg.python)
        banner(cfg)

        try:
            if not self.preflight():
                return 1
            print()
            print("  Starting services")
            self.start_ai()

            elapsed = time.perf_counter() - started
            print(f"  Startup completed in {elapsed:.1f}s")

            if cfg.headless:
                print("  Headless mode - press Ctrl+C to stop")
                self.supervisor.watch(interval_s=5.0)
                try:
                    while True:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    return 0
            return self.launch_ui()
        except KeyboardInterrupt:
            print("\n  Interrupted")
            return 130
        except Exception as exc:  # noqa: BLE001
            print(f"\n  Unexpected error: {type(exc).__name__}: {exc}")
            if self.log:
                self.log.exception("launcher failed")
            return 1
        finally:
            print("  Shutting down ...")
            self.supervisor.stop_all(timeout=10.0)
            if self.log:
                self.log.info("launcher exited")
            print("  Stopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="launcher", description="Start Sale Deed AI and everything it needs.")
    parser.add_argument("--headless", action="store_true",
                        help="start services without the desktop window")
    parser.add_argument("--check", action="store_true",
                        help="run the checks and exit without starting anything")
    parser.add_argument("--no-ai", action="store_true",
                        help="skip the inference server (browsing and export only)")
    parser.add_argument("--engine", choices=("llamacpp", "mock"),
                        help="override the inference engine")
    parser.add_argument("--root", type=Path, help="override project root detection")
    parser.add_argument("--verbose", action="store_true",
                        help="stream the AI server's log into this terminal")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    root = args.root.resolve() if args.root else find_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    cfg = build_config(root, headless=args.headless)
    if args.engine:
        cfg.engine = args.engine

    launcher = Launcher(cfg)
    launcher.verbose = args.verbose
    if args.check:
        launcher._configure_logging()
        banner(cfg)
        ok = launcher.preflight()
        print()
        print("  All checks passed." if ok else "  Fix the problems above.")
        return 0 if ok else 1

    if args.no_ai:
        launcher.start_ai = lambda: False  # type: ignore[method-assign]
    return launcher.run()
