# `src/` — all the code

Six packages. Full detail in [docs/CODE_MAP.md](../docs/CODE_MAP.md).

| Package | Role |
|---|---|
| `core/` | domain logic — no UI, no HTTP, no Qt |
| `ai_server/` | the inference service, its own process |
| `app/` | desktop shell — window, bridge, screens |
| `tools/` | installers, subprocess runners, verification harnesses |
| `launcher/` | starts, supervises and stops everything |
| `migrations/` | Alembic revisions |

Dependencies point one way:

```
launcher/  ──────────────► everything
app/       ──► core/          ──HTTP──► ai_server/
tools/     ──► core/, ai_server/
core/      ──► nothing above it
```

`core/` importing from `app/` or `ai_server/` is a defect: it must stay testable
without a window, a GPU or a running server. The shell talks to the inference
service over HTTP rather than importing it, so the UI process never links CUDA.

This directory is what goes on `sys.path`, not the project root — `import core`
finds `src/core`. `launcher.py`, `tests/conftest.py` and `alembic.ini` each
arrange that; nothing else needs to.

Filesystem locations come from `core/paths.py`. Do not write
`Path(__file__).resolve().parents[N]` — twenty files doing that is what made the
previous layout expensive to change.
