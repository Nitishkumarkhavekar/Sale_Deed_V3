"""End-to-end UI smoke test.  Run:  py -3.13 tools/ui_smoke.py

Drive the real window: real QWebEngineView, real QWebChannel, real app.js.

Every UI defect this session lived in the gap between "the renderer produces
HTML" and "the browser can actually use it". Nothing short of loading the page
into Chromium and letting app.js talk to Python over the channel would have
caught them.

Console capture is done from inside the page rather than by substituting a
QWebEnginePage: replacing the page after MainWindow has built it discards the
channel and produces a convincing but entirely artificial failure.
"""
import json
import os
import pathlib
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS",
                      "--disable-gpu --no-sandbox --disable-software-rasterizer")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from PySide6.QtCore import QElapsedTimer, QEventLoop
from PySide6.QtWebEngineCore import QWebEngineProfile, QWebEngineScript
from PySide6.QtWidgets import QApplication

import app.services as S
from app.main import (
    SCHEME,
    AssetHandler,
    MainWindow,
    install_web_channel_script,
    register_scheme,
)

ASSETS = ROOT / "app" / "ui" / "assets"

TRAP = """
window.__errors = [];
window.addEventListener('error', function (e) {
  window.__errors.push('onerror: ' + e.message);
});
window.addEventListener('unhandledrejection', function (e) {
  window.__errors.push('unhandled: ' + (e.reason && e.reason.message || e.reason));
});
(function () {
  var real = console.error;
  console.error = function () {
    window.__errors.push('console.error: ' +
      Array.prototype.join.call(arguments, ' '));
    real.apply(console, arguments);
  };
})();
"""


def pump(ms):
    t = QElapsedTimer(); t.start()
    while t.elapsed() < ms:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)


def js(view, script, timeout=5000):
    box, done = {}, {"v": False}

    def got(value):
        box["v"] = value
        done["v"] = True

    view.page().runJavaScript(script, 0, got)
    t = QElapsedTimer(); t.start()
    while not done["v"] and t.elapsed() < timeout:
        QApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    return box.get("v")


register_scheme()
app = QApplication.instance() or QApplication(sys.argv)

profile = QWebEngineProfile.defaultProfile()
trap = QWebEngineScript()
trap.setName("trap")
trap.setSourceCode(TRAP)
trap.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
trap.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
profile.scripts().insert(trap)

# Degraded path - no database, no AI server - because that is what the user sees.
S.check_connection = lambda _e: (False, "no database (smoke test)")
service = S.AppService()
service.db_ok = False

# Exactly what main() does.
install_web_channel_script()
# Kept in a module-level name: installUrlSchemeHandler does not take ownership,
# and a collected handler makes the navigation silently never start.
HANDLER = AssetHandler(ASSETS, render_page=lambda name: service.render_page(name, {}))
profile.installUrlSchemeHandler(SCHEME, HANDLER)

window = MainWindow(service)
window.show()
print("waiting for first paint ...")
pump(6000)

print("\n=== page environment ===")
checks = {}
for probe, label in (
    ("document.styleSheets.length", "stylesheets loaded"),
    ("typeof QWebChannel", "qwebchannel.js"),
    ("typeof qt !== 'undefined' && !!qt.webChannelTransport", "transport"),
    ("!!document.querySelector('#content')", "#content"),
    ("getComputedStyle(document.body).backgroundColor", "themed background"),
    ("!!document.querySelector('#capability-banner')", "capability banner"),
    ("document.querySelectorAll('nav a').length", "nav links"),
    # The hero sits outside #content on purpose - inside it, navigation would
    # wipe it on the first page change.
    ("!!document.querySelector('.hero h1')", "hero banner"),
    ("document.querySelectorAll('.tabs a').length", "top tabs"),
    ("!!document.querySelector('#nav-failed-count')", "failed-OCR badge"),
    ("!!document.querySelector('#btn-theme')", "theme toggle"),
):
    checks[label] = js(window.view, probe)
    print(f"  {label:22}: {checks[label]}")

print("\n=== theme toggle ===")
before = js(window.view, "document.documentElement.getAttribute('data-theme')")
js(window.view, "document.querySelector('#btn-theme').click()")
pump(400)
after = js(window.view, "document.documentElement.getAttribute('data-theme')")
dark_bg = js(window.view, "getComputedStyle(document.body).backgroundColor")
js(window.view, "document.querySelector('#btn-theme').click()")
pump(400)
restored = js(window.view, "document.documentElement.getAttribute('data-theme')")
theme_ok = before == "light" and after == "dark" and restored == "light"
print(f"  light -> {after} -> {restored}, dark background {dark_bg}")
print(f"  {'ok ' if theme_ok else '!! '}theme toggles both ways")

print("\n=== navigation through the real channel ===")
ok = True
for page in ("upload", "processing", "failed_ocr", "data", "watermark",
             "settings", "validation", "help", "dashboard"):
    before = js(window.view, "document.querySelector('#content').innerHTML") or ""
    js(window.view, f"location.hash = '#{page}'")
    pump(1300)
    after = js(window.view, "document.querySelector('#content').innerHTML") or ""
    active = js(window.view,
                "(function(a){return a?a.getAttribute('href'):'-'})"
                "(document.querySelector('nav a.active'))")
    good = after != before and active == "#" + page
    ok = ok and good
    print(f"  {'ok ' if good else '!! '}{page:12} bytes={len(after):>6}  active={active}")

print("\n=== status poll for several cycles ===")
pump(9000)
polled = js(window.view, "document.querySelector('#content').innerHTML.length")
print(f"  content intact after polling: {bool(polled)}")

errors = json.loads(js(window.view, "JSON.stringify(window.__errors || [])") or "[]")
print(f"\n=== JavaScript errors: {len(errors)} ===")
counts = {}
for e in errors:
    counts[e[:140]] = counts.get(e[:140], 0) + 1
for msg, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  x{n:<4} {msg}")

env_ok = (checks["stylesheets loaded"] and checks["qwebchannel.js"] == "function"
          and checks["#content"] and checks["nav links"])
print()
print("RESULT:", "PASS" if env_ok and ok and theme_ok and not errors else "FAIL")
window.close()
raise SystemExit(0 if (env_ok and ok and theme_ok and not errors) else 1)
