# `launcher/` — startup

| Module | Does |
|---|---|
| `steps.py` | the 13 preflight checks, each independently testable |
| `supervisor.py` | child processes: start, health-watch, restart, clean shutdown |
| `runner.py` | the sequence, the console output, the log |
| `config.py` | project-root discovery and `.env` loading |
| `__main__.py` | allows `python -m launcher` |

```
py -3.13 launcher.py              start everything, open the window
py -3.13 launcher.py --check      validate everything, change nothing
py -3.13 launcher.py --no-ai      browsing and export only
py -3.13 launcher.py --headless   services without the window
```

`launcher.py` at the project root is a shim: it puts `src/` on the path and
calls `runner.main`. It stays thin so this package can be imported and tested
without starting anything.

## Two details that matter

**Shutdown uses a Windows Job Object** with `KILL_ON_JOB_CLOSE`, so the whole
process tree dies with the launcher. A child that outlives its parent keeps the
GPU and port 8077, and the next start then fails for a reason that looks
entirely unrelated to the previous run.

**Interpreters are chosen by capability, not by version.** This project
deliberately needs two Pythons — 3.13 for the application, 3.12 for Surya — and
installing the second changes what a bare `python` resolves to. Probing with
`import PySide6` is the check; picking the newest is a defect that has already
occurred here.
