"""Database bring-up and verification for SQLAlchemy 2.0 + psycopg v3 + Alembic.

One command takes a PostgreSQL server from empty to ready, then proves the stack
actually works end to end.

    python tools/db_setup.py --check      probe the connection only
    python tools/db_setup.py --upgrade    alembic upgrade head
    python tools/db_setup.py --seed       write default settings rows
    python tools/db_setup.py --verify     full round-trip against real tables
    python tools/db_setup.py              all of the above, in order

`--verify` exists because the orchestration layer is otherwise untested. The
repositories, the `FOR UPDATE SKIP LOCKED` claim, per-stage resume, crash
recovery and continuous commit are all written and reviewed but have never
executed - there has been no server to execute them against. This runs them for
real and cleans up after itself.

Connection comes from `SALEDEED_DB_URL`, e.g.

    postgresql+psycopg://saledeed:secret@localhost:5432/saledeed

Anything using `postgresql://` or `postgresql+psycopg2://` is rewritten onto the
v3 driver, since that is the only one installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: `alembic.ini` sits beside the five top-level folders, not inside `src`, and
#: its `script_location` is written relative to itself. Alembic must therefore be
#: launched from the project root - run from `src` it reports "No 'script_location'
#: key found in configuration" and exits non-zero, which is exactly what every
#: --upgrade path here did until this was noticed while adding a migration.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.db.engine import (  # noqa: E402
    build_engine,
    build_session_factory,
    check_connection,
    dsn_from_env,
    session_scope,
)
from core.db.models import BatchState, DocumentState, StageState  # noqa: E402
from core.db.repositories import RepositoryError, UnitOfWork  # noqa: E402

DEFAULT_SETTINGS = {
    "ocr_language": "kn,en",
    "translation_language": "en",
    # -- translation (core/translation) ---------------------------------
    #: Master switch. Off means values pass through in their source language
    #: and the export reports which columns it could not render.
    "translation_enabled": "true",
    #: FLORES-200 codes, which is what NLLB expects. An ISO 639-1 code produces
    #: silent garbage rather than an error.
    "translation_target": "eng_Latn",
    #: "auto" detects per field. A deed is a mixed document - a Kannada name
    #: beside a Latin PAN - so a fixed source language would mistranslate.
    "translation_source": "auto",
    #: Hindi and Marathi share Devanagari and cannot be told apart by script.
    #: Maharashtra records should set this to mar_Deva.
    "translation_devanagari_as": "hin_Deva",
    "translation_model": "nllb-200-distilled-600M",
    #: "auto" uses CUDA only when enough VRAM is genuinely free.
    "translation_device": "auto",
    "translation_batch_size": "16",
    "translation_timeout_s": "600",
    #: One retry. A model that fails twice is broken, and a 500-document batch
    #: must not spend its time retrying.
    "translation_max_retries": "1",
    "ocr_dpi": "300",
    "llm_mode": "medium",
    "stamp_value_multiplier": "1",
    "batch_mode": "manual",
    #: Seconds to wait after a batch finishes before auto mode promotes the next
    #: one. Gives the GPU time to release memory; a zero wait would start the
    #: next batch the instant the last document lands.
    "auto_cooldown_seconds": "60",
    #: Nightly backup and purge. Off by default: retention DELETES data, and a
    #: destructive job that starts itself on first launch is the wrong default
    #: for a records system.
    "retention_interval_hours": "24",
    "debug_logs": "false",
    # Disabled: the formula is undefined and example.csv contradicts the spec.
    "rule_stamp_value": "false",
    "rule_pan": "true",
    "rule_aadhaar": "true",
    "rule_registration_fee": "true",
    "rule_sale_consideration": "true",
    "rule_transaction_date": "true",
    "rule_ocr_cross_verify": "true",
    "rule_confidence": "true",
    "pan_coverage_threshold": "0.6",
    "pan_split_threshold": "25",
}


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def cmd_check() -> bool:
    dsn = dsn_from_env()
    print(f"DSN: {dsn.split('@')[-1] if '@' in dsn else dsn}")
    engine = build_engine(connect_timeout_s=5)
    print(f"  driver: {engine.dialect.driver} "
          f"({engine.dialect.dbapi.__name__} "
          f"{getattr(engine.dialect.dbapi, '__version__', '?')})")
    reachable, detail = check_connection(engine)
    if reachable:
        _ok(detail.split(",")[0])
        return True
    _fail(detail.splitlines()[0])
    print("\n  A PostgreSQL server must be running and reachable. psycopg is a")
    print("  client driver; it cannot store data on its own.")
    return False


def cmd_upgrade() -> bool:
    print("Running alembic upgrade head")
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=PROJECT_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300, check=False)
    tail = (result.stderr or result.stdout or "").strip().splitlines()
    for line in tail[-6:]:
        print(f"    {line}")
    if result.returncode == 0:
        _ok("schema at head")
        return True
    _fail(f"alembic exited {result.returncode}")
    return False


def cmd_seed() -> bool:
    print("Seeding default settings")
    factory = build_session_factory(build_engine())
    try:
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            written = 0
            for key, value in DEFAULT_SETTINGS.items():
                if uow.settings.get(key) is None:
                    uow.settings.set(key, value)
                    written += 1
        _ok(f"{written} new setting(s), {len(DEFAULT_SETTINGS)} total")
        return True
    except Exception as exc:  # noqa: BLE001
        _fail(f"{type(exc).__name__}: {exc}")
        return False


def cmd_verify(dsn: str | None = None) -> bool:
    """Exercise the paths that have never run. Cleans up on the way out.

    `dsn` allows verification against SQLite when no PostgreSQL server exists.
    That proves the repository *logic* - claiming, stage ordering, idempotency,
    cascades, constraints, continuous commit. It does **not** prove the
    concurrency semantics: `FOR UPDATE SKIP LOCKED` is silently omitted by the
    SQLite dialect, so the guarantee that two workers never claim the same row
    remains PostgreSQL-only and untested until a server is available.
    """
    print("Verifying the stack against real tables")
    engine = build_engine(dsn) if dsn else build_engine()
    is_sqlite = engine.dialect.name == "sqlite"
    if is_sqlite:
        from core.db.engine import create_all

        create_all(engine)
        print("  [note] SQLite: logic is verified, SKIP LOCKED concurrency is NOT")
    factory = build_session_factory(engine)
    passed: list[str] = []
    failed: list[str] = []
    batch_id: int | None = None

    def check(name: str, condition: bool, detail: str = "") -> None:
        (passed if condition else failed).append(name)
        (_ok if condition else _fail)(f"{name}{f' - {detail}' if detail else ''}")

    try:
        # -- create ------------------------------------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("db_setup_verify")
            batch = uow.batches.create("verify_batch", user, file_count=3,
                                       total_bytes=3 * 1024 * 1024)
            batch_id = batch.id
            docs = uow.documents.add_many(batch, [
                {"document_id": f"VERIFY-{i}", "source_filename": f"v{i}.pdf",
                 "page_count": 2, "size_bytes": 1024 * 1024} for i in range(1, 4)])
            check("create user/batch/documents", len(docs) == 3, f"{len(docs)} docs")

        # -- idempotent registration -------------------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            again = uow.documents.add_many(batch, [
                {"document_id": "VERIFY-1", "source_filename": "v1.pdf"}])
            check("duplicate document_id skipped", not again,
                  "re-registration created no row")

        # -- queue cap ---------------------------------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            user = uow.users.get_or_create("db_setup_verify")
            made = []
            try:
                for i in range(6):
                    made.append(uow.batches.create(f"cap_{i}", user, 1, 1024).id)
                check("queue cap enforced", False, "6 batches accepted")
            except RepositoryError as exc:
                check("queue cap enforced", "maximum" in str(exc),
                      f"stopped after {len(made)} extra")
            for bid in made:
                b = uow.batches.get(bid)
                if b:
                    session.delete(b)

        # -- SKIP LOCKED claim + stage ordering --------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            first = uow.documents.claim_next("ocr", batch_id)
            check("claim_next(ocr) with SKIP LOCKED", first is not None,
                  first.document_id if first else "nothing claimed")
            blocked = uow.documents.claim_next("extract", batch_id)
            check("stage ordering blocks extract before ocr", blocked is None,
                  "extract correctly unclaimable")
            if first:
                uow.documents.mark_stage(first, "ocr", StageState.DONE,
                                          processing_status="OCR_P")
                uow.ocr.save_pages(first, [(1, "PAGE ONE TEXT"), (2, "PAGE TWO")])
                check("ocr_pages saved", len(first.ocr_pages) == 2,
                      f"{len(first.ocr_pages)} pages")
                nxt = uow.documents.claim_next("extract", batch_id)
                check("extract claimable after ocr done", nxt is not None,
                      nxt.document_id if nxt else "-")

        # -- crash recovery ----------------------------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            reset = uow.documents.reset_running_to_pending(batch_id)
            check("crash recovery resets RUNNING", reset >= 1, f"{reset} reset")

        # -- results + flags ---------------------------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(batch_id, per_page=1)[0][0]
            prop = uow.results.save_property(doc, {
                "schedule_c_property_address": "Test Village",
                "state": "Karnataka", "sale_consideration": "3300000",
                "registration_fee": "33000", "paid_in_cash": "no"})
            uow.results.save_document_meta(prop, {
                "transaction_date": "2025-04-09",
                "registration_office": "Sub Registrar, Test"})
            persons = uow.results.replace_persons(doc, {
                "buyer_details": [{"name": "BUYER ONE",
                                   "pan_card_number": "ADPPN2284H",
                                   "aadhaar_number": "241391305374"}],
                "seller_details": [{"name": "SELLER ONE",
                                    "pan_card_number": "AIMPP2121R"}]})
            check("property + persons persisted", len(persons) == 2,
                  f"{len(persons)} persons")
            check("aadhaar stored as 12-char text",
                  persons[0].aadhaar_number == "241391305374")
            n = uow.results.record_flags(doc, [
                {"flag_code": "OCR_P", "person_id": None, "confidence": 0.97},
                {"flag_code": "PM", "person_id": persons[0].id, "confidence": 1.0}])
            check("validation flags persisted", n == 2, f"{n} rows")

            # idempotency: replacing must not accumulate
            again = uow.results.replace_persons(doc, {
                "buyer_details": [{"name": "BUYER ONE"}], "seller_details": []})
            check("replace_persons is idempotent", len(doc.persons) == 1,
                  f"{len(doc.persons)} after replace")

        # -- progress ----------------------------------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            prog = uow.batches.progress(batch_id)
            check("progress aggregates", prog is not None and prog.total == 3,
                  f"total={prog.total if prog else '?'} "
                  f"ocr={prog.stages['ocr'].done if prog else '?'} done")
            rows, total = uow.batches.list_paginated(page=1, per_page=5)
            check("pagination works", total >= 1, f"{len(rows)} of {total}")

        # -- continuous commit -------------------------------------------
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            doc = uow.documents.list_for_batch(batch_id, per_page=1)[0][0]
            uow.documents.mark_overall(doc, DocumentState.PROCESSED)
            pk = doc.id
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            fresh = uow.documents.get(pk)
            check("committed across sessions",
                  fresh is not None and fresh.overall_state is DocumentState.PROCESSED)

    except Exception as exc:  # noqa: BLE001
        _fail(f"unexpected: {type(exc).__name__}: {exc}")
        failed.append("exception")
    finally:
        if batch_id is not None:
            try:
                with session_scope(factory) as session:
                    uow = UnitOfWork(session)
                    batch = uow.batches.get(batch_id)
                    if batch:
                        session.delete(batch)  # cascades to all children
                    user = uow.users.get_or_create("db_setup_verify")
                    session.delete(user)
                print("  [ok]   cleaned up verification data")
            except Exception as exc:  # noqa: BLE001
                _fail(f"cleanup: {type(exc).__name__}: {exc}")

    print(f"\n  {len(passed)} passed, {len(failed)} failed")
    if failed:
        print(f"  failing: {', '.join(failed)}")
    return not failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--upgrade", action="store_true")
    ap.add_argument("--seed", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sqlite", metavar="PATH", nargs="?", const=":memory:",
                    help="verify against SQLite instead of PostgreSQL. Proves the "
                         "repository logic; SKIP LOCKED concurrency remains untested.")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    if args.sqlite:
        target = ("sqlite+pysqlite:///:memory:" if args.sqlite == ":memory:"
                  else f"sqlite+pysqlite:///{args.sqlite}")
        print(f"Verifying against {target}")
        print("PostgreSQL remains the production target; this checks logic only.")
        print()
        return 0 if cmd_verify(target) else 1

    steps = [n for n in ("check", "upgrade", "seed", "verify") if getattr(args, n)]
    if not steps:
        steps = ["check", "upgrade", "seed", "verify"]

    if "SALEDEED_DB_URL" not in os.environ:
        print("note: SALEDEED_DB_URL is not set; using the built-in default")
        print(f"      {dsn_from_env()}\n")

    handlers = {"check": cmd_check, "upgrade": cmd_upgrade,
                "seed": cmd_seed, "verify": cmd_verify}
    for name in steps:
        print(f"\n--- {name} ---")
        if not handlers[name]():
            print(f"\nStopped at '{name}'.")
            return 1
    print("\nDatabase ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
