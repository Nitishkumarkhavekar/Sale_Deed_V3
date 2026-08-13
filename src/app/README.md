# `app/` — the desktop shell

A PySide6 window hosting a QWebEngine view of Pystache-rendered HTML. No web
server, no JavaScript framework, one JS file.

| Module | Does |
|---|---|
| `services.py` | every screen and every action — the largest file in the project |
| `status.py` | background probing and capability gating: what the UI may offer right now |
| `main.py` | window, custom URL scheme, the handler that serves assets and PDFs |
| `ui/bridge.py` | QWebChannel slots — the only path from the webview into Python |
| `ui/renderer.py` | Pystache rendering and view models |
| `ui/templates/` | logic-less Mustache, one per screen |
| `ui/assets/app.js` | the channel client |

## Rules, each learned from a defect

**Widgets touch only the GUI thread.** A `QFileDialog` opened on a worker thread
aborts the process natively — no traceback, no exception, the window simply
disappears. See `_GUI_THREAD` in `ui/bridge.py`.

**Slots are `(QString, QString)` and reply on a signal.** QWebChannel strips a
trailing JavaScript function as the reply handler, so a slot with the wrong
arity is never found. The error names the slot, which is not where the fault is.

**Never `document.write`.** It rebuilds the document and destroys the channel,
after which every call fails with `execCallbacks[...] is not a function`.
Replace `#content` instead.

**Templates are logic-less.** Anything requiring a decision is computed in
Python and handed over as a plain value.

## Verifying a change here

Source-level tests cannot see any of the above — each fault lives in the seam
between Qt, the channel and the page, and all of them passed the full suite.

```
py -3.13 src/tools/ui_smoke.py       real webview, real channel, real app.js
py -3.13 src/tools/service_sweep.py  every service entry point, live data
```
