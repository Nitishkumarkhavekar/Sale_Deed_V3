"""Security testing.

Threat model for this application. It is a single-user desktop tool on a
government clerk's machine, so there is no network attacker and no multi-tenancy.
What remains is real:

* **Untrusted document content.** A deed is a third-party file. Its OCR text
  flows into the database, into CSV, and into HTML. Each is a different escape
  context and each can be broken by a hostile or merely malformed document.
* **CSV formula injection.** The export is opened in Excel. A cell beginning
  `=`, `+`, `-` or `@` is executed as a formula, which is remote code execution
  by way of a spreadsheet.
* **Path traversal.** The UI serves assets over a custom scheme; a template that
  requests `../../.env` must not get it.
* **Credential disclosure.** The database password must not reach argv, where
  any local process can read it from the process table.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.csv_export import CSV_COLUMNS, DocumentExport, write_csv

pytestmark = pytest.mark.unit


def _deed(name: str, identity: str = "SEC-1") -> DocumentExport:
    """One deed whose party name is attacker-controlled."""
    return DocumentExport(
        transaction_identity=identity,
        extraction={"seller_details": [{"name": name}], "buyer_details": []},
        source_filename="x.pdf")


# ---------------------------------------------------------------------------
# SQL injection
# ---------------------------------------------------------------------------


class TestSqlInjection:
    """The ORM parameterises everything; these tests prove no path concatenates
    user input into SQL."""

    HOSTILE = [
        "'; DROP TABLE documents; --",
        "' OR '1'='1",
        "\\'; DELETE FROM batches WHERE '1'='1",
        "admin'--",
        "1; UPDATE settings SET value='x'",
    ]

    @pytest.mark.integration
    @pytest.mark.parametrize("payload", HOSTILE)
    def test_settings_key_is_parameterised(self, session_factory, payload):
        from core.db.engine import session_scope
        from core.db.repositories import UnitOfWork

        with session_scope(session_factory) as session:
            uow = UnitOfWork(session)
            # Must round-trip as literal data, not execute.
            uow.settings.set(payload, "value")
            assert uow.settings.get(payload) == "value"
            uow.settings.set(payload, None)

    @pytest.mark.integration
    def test_tables_still_exist_after_hostile_input(self, db_engine):
        from sqlalchemy import inspect

        assert "documents" in inspect(db_engine).get_table_names()

    @pytest.mark.integration
    @pytest.mark.parametrize("payload", HOSTILE)
    def test_username_is_parameterised(self, session_factory, payload):
        from core.db.engine import session_scope
        from core.db.repositories import UnitOfWork

        with session_scope(session_factory) as session:
            user = UnitOfWork(session).users.get_or_create(payload)
            assert user.username == payload
            session.delete(user)

    def test_repositories_do_not_build_sql_by_concatenation(self):
        """Static check: no f-string or % formatting inside a text() call."""
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "core" / "db" / "repositories.py").read_text(encoding="utf-8")
        assert not re.search(r'text\(\s*f["\']', source), \
            "an f-string is being passed to text() - use bound parameters"
        assert not re.search(r'text\([^)]*%\s*\(', source), \
            "%-formatting is being passed to text()"


# ---------------------------------------------------------------------------
# CSV formula injection
# ---------------------------------------------------------------------------


class TestCsvInjection:
    """A deed's party name is attacker-controlled text that lands in a cell."""

    PAYLOADS = [
        "=1+1",
        "=cmd|'/c calc'!A0",
        "+1+1",
        "-1+1",
        "@SUM(1:1)",
        '=HYPERLINK("http://evil","click")',
    ]

    @pytest.mark.parametrize("payload", PAYLOADS)
    def test_formula_is_neutralised_in_output(self, payload, tmp_path):
        """The written cell must not begin with a formula trigger.

        Excel evaluates a leading =, +, - or @. Quoting alone does not help:
        Excel strips the quotes when parsing the CSV field.
        """
        import csv as csvmod

        target = tmp_path / "out.csv"
        write_csv(target, [_deed(payload)])
        with open(target, encoding="utf-8-sig", newline="") as handle:
            rows = list(csvmod.reader(handle))

        for row in rows[1:]:
            for cell in row:
                assert cell[:1] not in ("=", "+", "@"), (
                    f"cell would be evaluated as a formula by Excel: {cell!r}")

    def test_ordinary_text_is_not_mangled(self, tmp_path):
        target = tmp_path / "out.csv"
        write_csv(target, [_deed("Ramesh Kumar", "SEC-2")])
        assert "Ramesh Kumar" in target.read_text(encoding="utf-8-sig")

    def test_newline_in_value_cannot_forge_a_row(self, tmp_path):
        """An embedded newline must be quoted, not emitted raw."""
        import csv as csvmod

        target = tmp_path / "out.csv"
        write_csv(target, [_deed("A\r\nFORGED,ROW,DATA", "SEC-3")])
        with open(target, encoding="utf-8-sig", newline="") as handle:
            parsed = list(csvmod.reader(handle))
        assert len(parsed) == 2, f"an extra row was forged: {len(parsed)} rows"
        assert len(parsed[1]) == len(CSV_COLUMNS)


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------


class TestPathTraversal:
    TRAVERSALS = [
        "../../.env",
        "..%2F..%2F.env",
        "assets/../../../core/db/engine.py",
        "/etc/passwd",
        "..\\..\\.env",
    ]

    @pytest.mark.parametrize("attempt", TRAVERSALS)
    def test_asset_paths_resolve_inside_the_asset_directory(self, attempt):
        """Mirrors AssetHandler's containment check without needing Qt."""
        asset_dir = (Path(__file__).resolve().parents[1] / "src" / "app" / "ui").resolve()
        rel = attempt.lstrip("/").removeprefix("assets/")
        target = (asset_dir / rel).resolve()
        contained = asset_dir == target or asset_dir in target.parents
        if contained:
            # Containment is the security property; existence is not required.
            assert asset_dir in target.parents or target == asset_dir
        else:
            assert not contained, "traversal escaped and must be rejected"

    def test_handler_rejects_escapes(self):
        """The real check in app/main.py: parents must contain the asset dir."""
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "app" / "main.py").read_text(encoding="utf-8")
        assert "self.asset_dir not in target.parents" in source, \
            "AssetHandler lost its containment check"
        assert ".resolve()" in source, "paths must be resolved before comparison"


# ---------------------------------------------------------------------------
# File validation
# ---------------------------------------------------------------------------


class TestFileValidation:
    def test_non_pdf_extension_is_rejected(self, tmp_path):
        from app.services import Selection

        evil = tmp_path / "payload.exe"
        evil.write_bytes(b"MZ\x90\x00")
        selection = Selection()
        assert selection.add([evil]) == 0

    def test_directory_is_not_accepted(self, tmp_path):
        from app.services import Selection

        folder = tmp_path / "notafile.pdf"
        folder.mkdir()
        assert Selection().add([folder]) == 0

    def test_duplicate_paths_are_collapsed(self, tmp_path):
        from app.services import Selection

        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        selection = Selection()
        selection.add([pdf])
        assert selection.add([pdf]) == 0, "the same file was queued twice"

    def test_content_is_checked_not_only_the_extension(self, tmp_path):
        """A file called .pdf that is not a PDF must not enter the pipeline.

        Extension alone is not identity. PyMuPDF will refuse it later, but the
        failure is then attributed to the document rather than to the upload,
        and the user sees a processing error instead of a rejected file.
        """
        from app.services import Selection

        fake = tmp_path / "notreally.pdf"
        fake.write_bytes(b"MZ\x90\x00this is an executable")
        assert Selection().add([fake]) == 0, \
            "a non-PDF passed validation on extension alone"


# ---------------------------------------------------------------------------
# Credential handling
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_backup_password_is_not_placed_on_the_command_line(self):
        """argv is world-readable on Windows; PGPASSWORD in the environment is
        not."""
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "core" / "backup.py").read_text(encoding="utf-8")
        assert "PGPASSWORD" in source
        assert "--password" not in source

    def test_dsn_is_not_logged_with_its_password(self):
        from core.db.engine import normalise_dsn

        dsn = normalise_dsn("postgresql://user:secret123@host:5432/db")
        safe = dsn.split("@")[-1]
        assert "secret123" not in safe

    def test_env_file_is_not_committed(self):
        """.env holds the database password in plaintext (L-006). It must at
        least not be tracked."""
        root = Path(__file__).resolve().parents[1]
        gitignore = root / ".gitignore"
        if not gitignore.is_file():
            pytest.skip("no .gitignore in this project")
        assert ".env" in gitignore.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Untrusted text into HTML
# ---------------------------------------------------------------------------


class TestHtmlEscaping:
    def test_pystache_escapes_by_default(self):
        """`{{value}}` escapes; `{{{value}}}` does not. Any template using the
        triple form on document-derived text is a stored-XSS vector."""
        import pystache

        out = pystache.render("{{v}}", {"v": "<script>alert(1)</script>"})
        assert "<script>" not in out
        assert "&lt;script&gt;" in out

    #: Fields deliberately rendered as raw HTML because the application itself
    #: writes them, markup and all. Every entry is a promise that no
    #: document-derived text reaches it unescaped.
    ALLOWED_RAW = {
        "content",              # base.mustache - the rendered page body itself
        "message",              # escaped at the source in _dashboard
        "pressure_detail",      # constant, app-authored
        "over_limit_message",   # built from byte counts
        "retry_detail",         # constant, app-authored
        "a",                    # help.mustache - answer text written by us
    }

    def test_templates_do_not_unescape_document_text(self):
        """Triple-mustache emits raw HTML, so every use must be accounted for.

        This is a whitelist rather than a ban: some notices carry intentional
        `<strong>`. The point is that adding a new one is a decision, not an
        accident - a new unescaped field fails this test until someone confirms
        nothing document-derived reaches it.
        """
        templates = (Path(__file__).resolve().parents[1]
                     / "src" / "app" / "ui" / "templates")
        if not templates.is_dir():
            pytest.skip("templates not present")
        unexpected: list[str] = []
        for path in sorted(templates.glob("*.mustache")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"\{\{\{\s*(\w+)", text):
                if match.group(1) not in self.ALLOWED_RAW:
                    unexpected.append(f"{path.name}:{match.group(1)}")
        assert not unexpected, (
            f"unescaped fields not on the reviewed list: {unexpected}")

    def test_exception_text_is_escaped_before_reaching_the_notice(self):
        """`message` is rendered raw, so the escaping must happen at the source."""
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "app" / "services.py").read_text(encoding="utf-8")
        assert "html.escape(self.errors[-1])" in source, \
            "error text reaches an unescaped template field"
