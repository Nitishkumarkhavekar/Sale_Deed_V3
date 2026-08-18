"""Desktop shell - PySide6 window hosting the Pystache-rendered UI.

Architecture, and why:

    QMainWindow -> QWebEngineView -> HTML from Pystache
                        |
                   QWebChannel -> Bridge -> AppService -> PostgreSQL
                                                       -> AI server (HTTP)

The window process **never loads a model and never links CUDA**. Inference lives
in a separate process reached over HTTP, so a crash in the runtime cannot take
down the interface, and heavy GPU work cannot stall the event loop.

Assets are served from a custom `app://` scheme rather than `file://`. Qt applies
tight origin rules to file URLs, which would block QWebChannel; a registered
scheme keeps the CSS, JS and rendered HTML in one origin.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

#: `src/`, the import root - not the project root. `core.paths` owns that.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import (  # noqa: E402
    QBuffer,
    QByteArray,
    QIODevice,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Slot,
)
from PySide6.QtGui import QDesktopServices, QIcon  # noqa: E402
from PySide6.QtWebChannel import QWebChannel  # noqa: E402
from PySide6.QtWebEngineCore import (  # noqa: E402
    QWebEngineProfile,
    QWebEngineSettings,
    QWebEngineUrlRequestJob,
    QWebEngineUrlScheme,
    QWebEngineUrlSchemeHandler,
)
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from app.services import APP_VERSION, AppService  # noqa: E402
from core import logging_setup  # noqa: E402
from app.ui.bridge import Bridge  # noqa: E402

log = logging.getLogger("saledeed.app")

SCHEME = b"app"
MIME = {".css": b"text/css", ".js": b"application/javascript",
        ".png": b"image/png", ".svg": b"image/svg+xml",
        ".jpg": b"image/jpeg", ".jpeg": b"image/jpeg",
        ".ico": b"image/vnd.microsoft.icon",
        ".woff2": b"font/woff2", ".html": b"text/html"}

#: The departmental emblem, served to the page and used as the window icon.
#: Named here rather than spelled out at each use so the two cannot diverge -
#: a renamed file that still loaded in one place and silently 404'd in the
#: other would be easy to miss.
LOGO_FILE = "income-tax-logo.jpg"


class AssetHandler(QWebEngineUrlSchemeHandler):
    """Serves the page **and** its assets over `app://ui/`.

    The page is served here rather than pushed in with `setHtml`. That is not a
    preference - it is the only arrangement in which the assets load at all.
    `setHtml` does not establish the base URL as a real origin, so Chromium
    refuses every subresource request to the custom scheme and this handler is
    never even consulted: no stylesheet, no `app.js`, and therefore no bridge.
    The failure is silent, because a request that is never made cannot fail.

    Loading `app://ui/page/<name>` gives the document a genuine origin on the
    scheme, and `app://ui/assets/...` is then same-origin and served normally.

    Path handling stays deliberately narrow: only files under the assets
    directory are reachable, so a stray URL in a template cannot read arbitrary
    paths.
    """

    def __init__(self, asset_dir: Path,
                 render_page: Callable[[str], str] | None = None,
                 resolve_pdf: Callable[[str], Path | None] | None = None) -> None:
        super().__init__()
        self.asset_dir = asset_dir.resolve()
        self.render_page = render_page
        self.resolve_pdf = resolve_pdf

    def _serve_pdf(self, job: QWebEngineUrlRequestJob, token: str) -> None:
        """Stream a prepared PDF to the built-in viewer.

        The URL carries a document id, never a path. Accepting a path would let
        any template - or anything that can set `location` - read arbitrary
        files through a handler that exists to show one directory.
        """
        target = self.resolve_pdf(token) if self.resolve_pdf else None
        if target is None or not Path(target).is_file():
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        buffer = QBuffer(job)
        buffer.setData(QByteArray(Path(target).read_bytes()))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        job.reply(b"application/pdf", buffer)

    def requestStarted(self, job: QWebEngineUrlRequestJob) -> None:  # noqa: N802
        path = job.requestUrl().path()

        if self.render_page is not None and (path.startswith("/page/") or path in ("", "/")):
            name = path.removeprefix("/page/").strip("/") or "dashboard"
            try:
                html = self.render_page(name)
            except Exception:  # noqa: BLE001 - a render failure must not hang the view
                job.fail(QWebEngineUrlRequestJob.Error.RequestFailed)
                return
            buffer = QBuffer(job)
            buffer.setData(QByteArray(html.encode("utf-8")))
            buffer.open(QIODevice.OpenModeFlag.ReadOnly)
            job.reply(b"text/html", buffer)
            return

        if path.startswith("/pdf/"):
            # A prepared document, for the viewer. Resolved through the service
            # rather than by path, so a URL cannot name an arbitrary file.
            self._serve_pdf(job, path.removeprefix("/pdf/").strip("/"))
            return

        rel = path.lstrip("/")
        target = (self.asset_dir / rel.removeprefix("assets/")).resolve()
        if not target.is_file() or self.asset_dir not in target.parents:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        buffer = QBuffer(job)
        buffer.setData(QByteArray(target.read_bytes()))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        job.reply(MIME.get(target.suffix.lower(), b"application/octet-stream"), buffer)


class MainWindow(QMainWindow):
    def __init__(self, service: AppService) -> None:
        super().__init__()
        self.service = service
        self.setWindowTitle(f"Sale Deed AI  {APP_VERSION}")
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)

        # The departmental emblem as the window, taskbar and Alt-Tab icon.
        # Resolved from this file's own location, never from a fixed drive, so
        # it is found wherever the application is installed. A missing file is
        # not worth failing a launch over - Qt's default icon is a cosmetic
        # loss, not a broken application - so it is checked rather than trusted.
        icon_path = Path(__file__).resolve().parent / "ui" / "assets" / LOGO_FILE
        if icon_path.is_file():
            self.setWindowIcon(QIcon(str(icon_path)))
        else:
            log.warning("window icon missing: %s", icon_path)

        self.view = QWebEngineView(self)
        self.setCentralWidget(self.view)

        settings = self.view.settings()
        # Nothing here is remote; disabling these removes attack surface and a
        # little memory in a process that is already tight for RAM.
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanOpenWindows, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
        # Chromium's own PDF viewer. It brings text selection, copy, search and
        # zoom for free, and renders the *searchable* PDF the pipeline prepared -
        # so what the operator selects is the same text the model extracted from.
        # Writing a viewer instead would mean reimplementing all of that and
        # getting a different text layer.
        # PluginsEnabled is what actually switches the PDF viewer on - the
        # viewer is implemented as an internal plugin, so PdfViewerEnabled
        # alone does nothing. Nothing here is remote and no third-party plugin
        # can be loaded, so this does not widen the attack surface.
        settings.setAttribute(QWebEngineSettings.WebAttribute.PdfViewerEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, True)

        self.bridge = Bridge(service, self)
        self.channel = QWebChannel(self)
        self.channel.registerObject("bridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        # The service stays Qt-free; the native dialog is injected here.
        service.file_picker = self._pick_files  # type: ignore[attr-defined]
        service.save_picker = self._pick_save_path  # type: ignore[attr-defined]

        self._load("dashboard")

    def _load(self, page: str) -> None:
        """Navigate to the page over the custom scheme.

        `setHtml` would leave the document without a real origin on `app://`,
        and every stylesheet and script request would then be refused before
        the scheme handler ever saw it.
        """
        self.view.load(QUrl(f"app://ui/page/{page}"))

    @Slot()
    def _pick_files(self) -> list[str]:
        """Native file dialog. **GUI thread only.**

        A `QFileDialog` is a widget, and Qt widgets may only be created on the
        thread owning the QApplication. Constructing one on a worker terminates
        the process outright - no exception, no traceback, nothing in the log,
        because the abort happens below the interpreter.

        The guard below turns that silent kill into an ordinary error the bridge
        can report. It should never fire: `Bridge._GUI_THREAD` routes this call
        to the GUI thread. It exists because the failure it prevents is
        unreadable, and a future caller will not know that from the signature.
        """
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            raise RuntimeError(
                "the file dialog was opened from a worker thread; this would "
                "terminate the application. Add the calling method to "
                "Bridge._GUI_THREAD.")

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select sale deed PDFs", str(Path.home()), "PDF files (*.pdf)")
        return list(paths)

    @Slot()
    def _pick_save_path(self, suggested: str = "export.csv") -> str:
        """Native Save As dialog. **GUI thread only** - see `_pick_files`.

        Returns "" when the operator cancels, which the caller must treat as
        "do nothing" rather than as an error: cancelling a save is a decision,
        not a failure.
        """
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            raise RuntimeError(
                "the save dialog was opened from a worker thread; this would "
                "terminate the application. Add the calling method to "
                "Bridge._GUI_THREAD.")

        name = Path(str(suggested or "export.csv")).name or "export.csv"
        chosen, _ = QFileDialog.getSaveFileName(
            self, "Save the batch export as",
            str(Path.home() / name),
            "CSV files (*.csv);;All files (*)")
        return str(chosen or "")

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.service.runner.state.value == "running":
            answer = QMessageBox.question(
                self, "Processing is running",
                "A batch is still processing.\n\n"
                "Progress is saved after every document, so closing now is safe - "
                "processing resumes where it stopped when you reopen.\n\nClose anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.bridge.shutdown()
        self.service.shut_down()
        event.accept()


def install_web_channel_script() -> None:
    """Inject `qwebchannel.js` into every page on the profile.

    It cannot be linked with `<script src="qrc:/qtwebchannel/qwebchannel.js">`.
    `qrc:` is a different origin from `app://ui/`, so Chromium blocks the
    request and `QWebChannel` is simply undefined - app.js then sits in its
    "preview mode" branch with every action inert and no error to show for it.

    Injecting at document creation also guarantees it is defined before app.js
    runs, which a second `<script>` tag would not.
    """
    from PySide6.QtCore import QFile, QIODevice
    from PySide6.QtWebEngineCore import QWebEngineScript

    profile = QWebEngineProfile.defaultProfile()
    if any(s.name() == "qwebchannel" for s in profile.scripts().find("qwebchannel")):
        return

    source = QFile(":/qtwebchannel/qwebchannel.js")
    if not source.open(QIODevice.OpenModeFlag.ReadOnly):
        return
    code = bytes(source.readAll()).decode("utf-8")
    source.close()

    script = QWebEngineScript()
    script.setName("qwebchannel")
    script.setSourceCode(code)
    script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentCreation)
    script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
    script.setRunsOnSubFrames(False)
    profile.scripts().insert(script)


def register_scheme() -> None:
    """Must run before QApplication exists.

    **No `LocalScheme` flag.** It marks the scheme file-like, and Chromium
    refuses to *navigate* to a local scheme: `view.load()` returns without
    error, `loadFinished` never fires, the URL stays empty and the handler is
    never consulted. Nothing is logged, because nothing failed - the navigation
    simply never began.

    `SecureScheme` gives the origin the privileges a modern page needs;
    `LocalAccessAllowed` lets it reach `file:` resources; `CorsEnabled` lets
    `app.js` be fetched as a subresource of the page rather than being treated
    as an opaque cross-origin request.
    """
    scheme = QWebEngineUrlScheme(SCHEME)
    scheme.setFlags(
        QWebEngineUrlScheme.Flag.SecureScheme
        | QWebEngineUrlScheme.Flag.LocalAccessAllowed
        | QWebEngineUrlScheme.Flag.CorsEnabled)
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    QWebEngineUrlScheme.registerScheme(scheme)


def main() -> int:
    register_scheme()

    QApplication.setApplicationName("Sale Deed AI")
    QApplication.setOrganizationName("Sale Deed AI")
    app = QApplication(sys.argv)

    service = AppService(reload_templates="--dev" in sys.argv)

    # Logging is configured after the service so the database session factory can
    # back the DB handler, and before start_up() so recovery is recorded.
    logging_setup.configure(
        session_factory=service.sessions if service.db_ok else None,
        app_name="saledeed")
    log = logging_setup.get_logger("app")
    log.info("application starting", extra={
        "version": APP_VERSION, "database": service.db_ok,
        "ai_server": service.ai.base_url})

    # The scheme handler serves both the page and its assets. It must be
    # created before the window, because MainWindow navigates during __init__.
    install_web_channel_script()
    handler = AssetHandler(Path(__file__).resolve().parent / "ui" / "assets",
                           render_page=lambda name: service.render_page(name, {}),
                           resolve_pdf=service.document_pdf)
    QWebEngineProfile.defaultProfile().installUrlSchemeHandler(SCHEME, handler)

    window = MainWindow(service)
    # `installUrlSchemeHandler` does NOT take ownership. Without a surviving
    # reference the handler is collected, and the failure is silent and total:
    # the navigation never starts, `loadFinished` never fires, the URL stays
    # empty and the window shows nothing at all. Parenting it to the window ties
    # its lifetime to the thing that needs it.
    handler.setParent(window)
    window.show()
    log.info("window shown")

    # Deferred to the first idle turn of the event loop, which is what
    # `start_up`'s own docstring has always specified - "called after the window
    # is painted, so neither step delays first paint". It was being called
    # before `show()`, and it costs ~410 ms: the status probes, the runner
    # settings, the retention scheduler and a crash-recovery scan across every
    # interrupted document. All of that was time the operator spent looking at
    # nothing. None of it is needed to paint the first frame.
    QTimer.singleShot(0, service.start_up)

    if not service.db_ok:
        QMessageBox.warning(
            window, "Database unavailable",
            "The application could not reach PostgreSQL.\n\n"
            f"{service.db_detail.splitlines()[0]}\n\n"
            "The interface will open, but batches cannot be created or processed "
            "until the database is reachable.")

    try:
        return app.exec()
    finally:
        # Flush the queue listener before the process exits, or the final
        # records of a session are lost.
        log.info("application exiting")
        logging_setup.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
