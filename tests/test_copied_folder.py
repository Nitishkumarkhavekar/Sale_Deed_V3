"""Copying the whole project folder to another machine.

This is what people actually do - it brings the 43 GB of models with it, which
a clone cannot - and it was the one route that could not work. Three
virtualenvs arrive recording absolute paths they no longer have, and `.env`
arrives naming a database password the new machine has never heard of.

The password was the blocking one. Setup generated a fresh password, created
the role with it, then re-checked the connection using the copied `.env` - and
`SALEDEED_DB_URL` wins outright over the generated parts. The two halves could
never agree, `.env` is deliberately never overwritten, so the install ended
with "PostgreSQL not reachable after install" on a machine where every other
step had succeeded.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class TestACopiedEnvKeepsItsPassword:
    """Whatever `.env` says is what the role must be created with."""

    def _with_env(self, monkeypatch, contents):
        tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(tmp.name)
        (root / ".env").write_text(contents, encoding="utf-8")
        from tools import system_setup as ss
        monkeypatch.setattr(ss.paths, "ROOT", root)
        return ss, tmp

    def test_the_password_is_read_from_a_full_url(self, monkeypatch):
        ss, tmp = self._with_env(monkeypatch, chr(10).join([
            "SALEDEED_DB_URL=postgresql+psycopg://saledeed:s3cret@localhost:5432/saledeed",
            "SALEDEED_AI_URL=http://127.0.0.1:8077", ""]))
        with tmp:
            assert ss._password_from_env_file() == "s3cret"

    def test_percent_encoding_is_undone(self, monkeypatch):
        """build_dsn quotes the password on the way in, because a generated one
        can contain characters that would end the URL early."""
        ss, tmp = self._with_env(monkeypatch, chr(10).join([
            "SALEDEED_DB_URL=postgresql+psycopg://saledeed:a%40b%2Fc@localhost:5432/saledeed",
            ""]))
        with tmp:
            assert ss._password_from_env_file() == "a@b/c"

    def test_an_explicit_password_variable_wins(self, monkeypatch):
        ss, tmp = self._with_env(monkeypatch, "SALEDEED_DB_PASSWORD=plain" + chr(10))
        with tmp:
            assert ss._password_from_env_file() == "plain"

    def test_export_prefixes_and_quotes_are_tolerated(self, monkeypatch):
        ss, tmp = self._with_env(monkeypatch,
                                 'export SALEDEED_DB_PASSWORD="q u o"' + chr(10))
        with tmp:
            assert ss._password_from_env_file() == "q u o"

    def test_no_env_file_means_generate_one(self, monkeypatch):
        """The fresh-machine case must not be disturbed."""
        from tools import system_setup as ss

        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(ss.paths, "ROOT", pathlib.Path(tmp))
            assert ss._password_from_env_file() == ""

    def test_a_malformed_url_does_not_raise(self, monkeypatch):
        ss, tmp = self._with_env(monkeypatch, "SALEDEED_DB_URL=not a url at all" + chr(10))
        with tmp:
            assert ss._password_from_env_file() == ""

    def test_the_generated_password_is_only_the_last_resort(self):
        """Order in main(): flag, then existing .env, then generate."""
        import inspect

        from tools import system_setup as ss

        source = inspect.getsource(ss.main)
        assert "_password_from_env_file()" in source
        flag = source.index("args.db_password")
        existing = source.index("_password_from_env_file()")
        generated = source.index("_generate_password()")
        assert flag < existing < generated


class TestForeignEnvironmentsAreReplaced:
    """All three arrive broken in a copied folder and all three are rebuilt."""

    def test_the_ocr_environment_is_rebuilt(self):
        import inspect

        from tools import system_setup as ss

        assert "_rebuild_surya" in inspect.getsource(ss.ensure_ocr)

    def test_the_vllm_environment_is_replaced_not_installed_into(self):
        """Reaching pip through a broken interpreter fails with a message
        about pip, which sends the reader nowhere useful."""
        import inspect

        from tools import system_setup as ss

        source = inspect.getsource(ss.ensure_vllm)
        rebuild = source.index("rmtree")
        install = source.index('"-m", "pip", "install"')
        assert rebuild < install, "must replace the environment before using it"

    def test_the_application_environment_is_rebuilt_by_the_batch_file(self):
        body = (ROOT / "System Setup.bat").read_text(encoding="utf-8")
        assert "Rebuilding it for this one" in body
