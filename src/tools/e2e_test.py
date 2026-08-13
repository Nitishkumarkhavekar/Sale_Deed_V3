"""End-to-end backend test: real PDFs -> OCR -> GPU extraction -> validation -> PostgreSQL -> CSV.

Exercises the one layer that has only ever been audited, never executed: the
`BatchRunner` orchestration. Everything it drives has been tested in isolation -
the stages on real documents, the repositories against real tables - but the wiring
between them has not.

Creates its own batch, processes it, verifies the persisted rows, exports the CSV,
then deletes everything it created.

    python tools/e2e_test.py
    python tools/e2e_test.py --keep      leave the batch in the database
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.csv_export import DocumentExport, write_csv  # noqa: E402
from core.db.engine import build_engine, build_session_factory, check_connection, session_scope  # noqa: E402
from core.db.models import DocumentState, StageState  # noqa: E402
from core.db.repositories import UnitOfWork  # noqa: E402
from core.pipeline.runner import BatchMode, BatchRunner, build_stages  # noqa: E402
from core.validation import validate_extraction  # noqa: E402
from core import paths

PDF_DIR = paths.TESTS / "corpus" / "saledeeds"
AI_URL = "http://127.0.0.1:8077"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="do not delete the batch")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N PDFs (0 = all)")
    ap.add_argument("--ocr", default="auto", choices=("auto", "surya", "textlayer"),
                    help="auto uses whatever production would use")
    ap.add_argument("--translate", default="auto",
                    choices=("auto", "passthrough"),
                    help="auto uses the real translation service")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    engine = build_engine()
    ok, detail = check_connection(engine)
    print(f"database : {'OK - ' + detail.split(',')[0] if ok else 'UNREACHABLE'}")
    if not ok:
        print(detail.splitlines()[-1])
        return 1
    factory = build_session_factory(engine)

    # Production configuration by default. This harness used to pin
    # `ocr_engine="textlayer"` and `translator_engine="passthrough"`, which made
    # it fast and meant it had never once exercised Surya or the translation
    # model - the two slowest and most failure-prone stages in the pipeline.
    # An end-to-end test that skips two stages is not an end-to-end test.
    stages = build_stages(ai_base_url=AI_URL, ocr_engine=args.ocr,
                          translator_engine=args.translate, retry_supported=False)
    print(f"ocr      : {stages.ocr.engine}  {stages.ocr.available()[1]}")
    print(f"translate: {stages.translate.engine}  {stages.translate.available()[1]}")
    # Wait rather than give up on the first probe. Loading the model takes the
    # host RAM down to a few percent free on a small machine, the governor calls
    # that critical and refuses work, and it clears by itself about a minute
    # later. Checking once and exiting reported a working server as broken.
    deadline = time.time() + 180
    ready, engine_detail = stages.extract.health()
    announced = False
    while not ready and time.time() < deadline:
        if not announced:
            print(f"ai server: waiting - {engine_detail}")
            announced = True
        time.sleep(5)
        ready, engine_detail = stages.extract.health()
    print(f"ai server: {'OK - ' + engine_detail if ready else 'NOT READY - ' + engine_detail}")
    if not ready:
        return 1

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[:args.limit]
    print(f"documents: {len(pdfs)} PDFs from {PDF_DIR.name}/")

    # -- register the batch ---------------------------------------------
    with session_scope(factory) as session:
        uow = UnitOfWork(session)
        user = uow.users.get_or_create("e2e_test")
        batch = uow.batches.create(
            f"e2e_{int(time.time())}", user, file_count=len(pdfs),
            total_bytes=sum(p.stat().st_size for p in pdfs))
        batch_id = batch.id
        uow.documents.add_many(batch, [
            {"document_id": p.stem, "source_filename": p.name,
             "source_path": str(p), "size_bytes": p.stat().st_size}
            for p in pdfs])
    print(f"batch    : id={batch_id} registered\n")

    # -- run -------------------------------------------------------------
    runner = BatchRunner(factory, stages, mode=BatchMode.MANUAL,
                         max_workers=args.workers)
    recovered = runner.recover()
    if recovered:
        print(f"recovered {recovered} stranded stage(s) from a previous run")

    started = time.monotonic()
    runner.start()
    print(f"runner   : started, {args.workers} worker(s)")

    last = ""
    while time.monotonic() - started < args.timeout:
        time.sleep(4)
        with session_scope(factory) as session:
            prog = UnitOfWork(session).batches.progress(batch_id)
        if prog is None:
            break
        line = (f"  ocr {prog.stages['ocr'].done}/{prog.total}  "
                f"extract {prog.stages['extract'].done}/{prog.total}  "
                f"validate {prog.stages['validate'].done}/{prog.total}  "
                f"| done {prog.completed} review {prog.needs_review} "
                f"failed {prog.failed}")
        if line != last:
            print(line)
            last = line
        if prog.completed + prog.failed + prog.needs_review >= prog.total:
            break

    runner.stop(timeout=60)
    elapsed = time.monotonic() - started
    print(f"runner   : stopped after {elapsed:.0f}s "
          f"({runner.stats.seconds_per_document:.1f}s/document)\n")

    # -- verify what landed in PostgreSQL -------------------------------
    failures: list[str] = []
    with session_scope(factory) as session:
        uow = UnitOfWork(session)
        prog = uow.batches.progress(batch_id)
        docs, _ = uow.documents.list_for_batch(batch_id, per_page=50)
        print(f"{'document':<26}{'state':<14}{'persons':>8}{'consid':>12}{'regfee':>9}"
              f"{'conf':>7}  flags")
        exports: list[DocumentExport] = []
        for doc in docs:
            prop = doc.property_
            flags = sorted({v.flag_code for v in doc.validations})
            conf = next((float(v.confidence) for v in doc.validations
                         if v.person_id is None and v.confidence is not None), 0.0)
            print(f"{doc.document_id[:25]:<26}{doc.overall_state.value:<14}"
                  f"{len(doc.persons):>8}"
                  f"{str(int(prop.sale_consideration)) if prop and prop.sale_consideration else '-':>12}"
                  f"{str(int(prop.registration_fee)) if prop and prop.registration_fee else '-':>9}"
                  f"{conf:>7.2f}  {' '.join(flags[:4])}")

            if doc.overall_state is DocumentState.PROCESSING:
                failures.append(f"{doc.document_id} still PROCESSING")
            if doc.ocr_state is StageState.DONE and not doc.ocr_pages:
                failures.append(f"{doc.document_id} ocr done but no pages stored")

            exports.append(DocumentExport(
                transaction_identity=doc.transaction_identity or "",
                extraction={
                    "buyer_details": [_person(p) for p in doc.persons
                                      if p.relation.value == "B"],
                    "seller_details": [_person(p) for p in doc.persons
                                       if p.relation.value == "S"],
                    "property_details": {
                        "schedule_c_property_address": prop.schedule_c_address if prop else None,
                        "state": prop.state if prop else None,
                        "sale_consideration": str(int(prop.sale_consideration))
                        if prop and prop.sale_consideration else None,
                        "registration_fee": str(int(prop.registration_fee))
                        if prop and prop.registration_fee else None,
                        "paid_in_cash": ("yes" if prop.paid_in_cash else "no")
                        if prop and prop.paid_in_cash is not None else None,
                    },
                    "document_details": {
                        "transaction_date": prop.transaction_date.isoformat()
                        if prop and prop.transaction_date else None,
                        "registration_office": prop.registration_office if prop else None,
                    },
                },
                source_filename=doc.source_filename))

        print()
        print(f"batch    : {prog.completed} processed, {prog.needs_review} review, "
              f"{prog.failed} failed of {prog.total}")

    # -- CSV -------------------------------------------------------------
    out = paths.EXPORT_DIR / f"e2e_{batch_id}.csv"
    rows = write_csv(out, exports)
    print(f"csv      : {rows} rows -> {out.relative_to(paths.ROOT)}")

    # -- cleanup ---------------------------------------------------------
    if not args.keep:
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            batch = uow.batches.get(batch_id)
            if batch:
                session.delete(batch)
            user = uow.users.get_or_create("e2e_test")
            session.delete(user)
        with session_scope(factory) as session:
            uow = UnitOfWork(session)
            leftover = uow.documents.list_for_batch(batch_id, per_page=1)[1]
            if leftover:
                failures.append(f"cascade left {leftover} documents behind")
        print("cleanup  : batch deleted, cascade verified")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("End-to-end backend test PASSED")
    return 0


def _person(p: object) -> dict[str, object]:
    return {
        "name": getattr(p, "name", None),
        "name_translated": getattr(p, "name_translated", None),
        "gender": getattr(p, "gender", None),
        "father_name": getattr(p, "father_name", None),
        "aadhaar_number": getattr(p, "aadhaar_number", None),
        "pan_card_number": getattr(p, "pan_card_number", None),
        "address": getattr(p, "address", None),
        "state": getattr(p, "state", None),
    }


if __name__ == "__main__":
    raise SystemExit(main())
