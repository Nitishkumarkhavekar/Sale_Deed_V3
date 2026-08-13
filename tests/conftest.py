"""Shared fixtures.

Tests are split by what they need, so the suite stays useful on a machine with
no database and no GPU:

    unit         pure Python, always runs
    integration  needs PostgreSQL (SALEDEED_DB_URL)
    regression   needs the 50-deed corpus in the repo
    gpu          needs a running AI server
    slow         load and stress

Anything unavailable is skipped with a reason rather than failing, because a red
suite that is red for environmental reasons trains people to ignore it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: The packages live in `src/`. Put that on the path, not the project root -
#: importing `core` must find `src/core`, and the root itself holds no packages.
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CORPUS_OCR = ROOT / "tests" / "corpus" / "OCR saledeeds"
CORPUS_JSON = ROOT / "tests" / "corpus" / "test scripts" / "outputs" / "vllm_ocr"
PDF_DIR = ROOT / "tests" / "corpus" / "saledeeds"
AI_URL = os.environ.get("SALEDEED_AI_URL", "http://127.0.0.1:8077")


def pytest_configure(config: pytest.Config) -> None:
    for marker, description in (
        ("unit", "pure Python, no external dependencies"),
        ("integration", "requires a reachable PostgreSQL database"),
        ("regression", "requires the deed corpus"),
        ("gpu", "requires a running AI server with a loaded model"),
        ("slow", "load and stress tests"),
    ):
        config.addinivalue_line("markers", f"{marker}: {description}")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def corpus_pairs() -> list[tuple[str, Path, Path]]:
    """(stem, ocr_path, reference_json_path) for every deed with both files."""
    if not CORPUS_OCR.is_dir() or not CORPUS_JSON.is_dir():
        pytest.skip("deed corpus not present")
    pairs = []
    for ocr in sorted(CORPUS_OCR.glob("*.txt")):
        reference = CORPUS_JSON / f"{ocr.stem}.json"
        if reference.is_file():
            pairs.append((ocr.stem, ocr, reference))
    if not pairs:
        pytest.skip("no OCR/reference pairs found")
    return pairs


@pytest.fixture(scope="session")
def sample_deed(corpus_pairs) -> tuple[str, str, dict]:
    """One known-good deed: (stem, ocr_text, reference_extraction)."""
    for stem, ocr, reference in corpus_pairs:
        if stem == "117":
            return stem, ocr.read_text(encoding="utf-8", errors="replace"), \
                json.loads(reference.read_text(encoding="utf-8"))
    stem, ocr, reference = corpus_pairs[0]
    return stem, ocr.read_text(encoding="utf-8", errors="replace"), \
        json.loads(reference.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def pdf_files() -> list[Path]:
    if not PDF_DIR.is_dir():
        pytest.skip("no sample PDFs")
    files = sorted(PDF_DIR.glob("*.pdf"))
    if not files:
        pytest.skip("no sample PDFs")
    return files


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_engine():
    """Engine against the configured database, or skip."""
    from core.db.engine import build_engine, check_connection

    engine = build_engine(connect_timeout_s=4)
    ok, detail = check_connection(engine)
    if not ok:
        pytest.skip(f"database unreachable: {detail.splitlines()[0]}")
    return engine


@pytest.fixture()
def session_factory(db_engine):
    from core.db.engine import build_session_factory

    return build_session_factory(db_engine)


@pytest.fixture()
def temp_batch(session_factory):
    """A batch with three documents, removed afterwards whatever the test does."""
    from core.db.engine import session_scope
    from core.db.repositories import UnitOfWork

    created: dict[str, int] = {}
    with session_scope(session_factory) as session:
        uow = UnitOfWork(session)
        user = uow.users.get_or_create("pytest_user")
        batch = uow.batches.create("pytest_batch", user, 3, 3 * 1024 * 1024)
        uow.documents.add_many(batch, [
            {"document_id": f"PYTEST-{i}", "source_filename": f"p{i}.pdf",
             "source_path": f"p{i}.pdf", "size_bytes": 1024}
            for i in range(1, 4)])
        created["batch_id"] = batch.id

    yield created["batch_id"]

    with session_scope(session_factory) as session:
        uow = UnitOfWork(session)
        batch = uow.batches.get(created["batch_id"])
        if batch is not None:
            session.delete(batch)
        user = uow.users.get_or_create("pytest_user")
        session.delete(user)


@pytest.fixture()
def app_service(db_engine):
    """A real `AppService` against the test database.

    Constructed, not started: `start_up()` spawns the status poller and the
    retention scheduler, and a test that leaves those running leaks threads into
    every test that follows. Everything exercised here is synchronous.
    """
    from app.services import AppService

    service = AppService()
    try:
        yield service
    finally:
        # A test that queued work may have started the runner; leaving it alive
        # would keep claiming documents belonging to later tests.
        try:
            service.runner.stop(timeout=5)
        except Exception:  # noqa: BLE001 - teardown must not mask a failure
            pass


# ---------------------------------------------------------------------------
# AI server
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ai_server() -> str:
    """Base URL of a ready AI server, or skip."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{AI_URL}/health", timeout=5) as response:
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        pytest.skip(f"AI server unreachable at {AI_URL}: {exc}")
    if not payload.get("ready"):
        pytest.skip(f"AI server not ready: {payload.get('pressure')}")
    return AI_URL


@pytest.fixture()
def tmp_export(tmp_path) -> Path:
    return tmp_path / "export.csv"
