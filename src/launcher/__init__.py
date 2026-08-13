"""Single entry point that starts, monitors and stops the whole application.

    python launcher.py            start everything and open the window
    python launcher.py --check    run the checks only, change nothing
    python launcher.py --headless services without the desktop window

Layout:

    config.py       root discovery, `.env`, resolved paths
    steps.py        one function per requirement, each independently testable
    supervisor.py   child processes, health watch, restart, clean shutdown
    runner.py       the sequence and the console output
"""

from .config import LauncherConfig, build_config, find_root
from .runner import Launcher, main
from .supervisor import Service, Supervisor

__all__ = ["Launcher", "LauncherConfig", "Service", "Supervisor",
           "build_config", "find_root", "main"]
