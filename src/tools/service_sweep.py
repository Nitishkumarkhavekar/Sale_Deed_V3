"""Exercise every service entry point against the live database.

Run:  py -3.13 tools/service_sweep.py

Original note against the live database.

A DetachedInstanceError only appears when an ORM object outlives its session
*and* the attribute reached for was never loaded. That needs real rows, so no
amount of source reading finds it - the batch-detail case looked correct until a
batch was actually opened.
"""
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import app.services as S
from core.db.engine import session_scope
from core.db.repositories import UnitOfWork

svc = S.AppService()
if not svc.db_ok:
    print("SKIP: no database")
    raise SystemExit(0)

with session_scope(svc.sessions) as s:
    uow = UnitOfWork(s)
    counts = uow.maintenance.table_counts()
    batch_ids = [b.id for b in uow.batches.list_paginated(1, 20)[0]]
    doc_pks = []
    for bid in batch_ids:
        doc_pks += [d.id for d in uow.documents.list_for_batch(bid, 1, 5)[0]]

print("rows:", counts)
print(f"batches={len(batch_ids)} documents={len(doc_pks)}\n")

failures = []


def check(label, fn):
    try:
        fn()
        print(f"  ok   {label}")
    except Exception as exc:  # noqa: BLE001
        failures.append((label, exc))
        print(f"  FAIL {label}: {type(exc).__name__}: {str(exc)[:90]}")
        for line in traceback.format_exc().splitlines():
            if "app\\services.py" in line or "app/services.py" in line:
                print(f"         {line.strip()}")


print("=== pages ===")
for page in ("dashboard", "upload", "processing", "failed_ocr", "data",
             "watermark", "settings", "validation", "help"):
    check(f"render {page}", lambda p=page: svc.render_page(p, {}))
    check(f"render {page} (fragment)",
          lambda p=page: svc.render_page(p, {}, shell_html=False))

print("\n=== paged views ===")
for page in (1, 2):
    check(f"dashboard page {page}", lambda p=page: svc.render_page("dashboard", {"page": p}))
    check(f"data page {page}", lambda p=page: svc.render_page("data", {"page": p}))

print("\n=== fragments with real ids ===")
for bid in batch_ids[:5]:
    check(f"batch_detail {bid}",
          lambda b=bid: svc.render_fragment("batch_detail", {"batch_id": b}))

print("\n=== document views ===")
for pk in doc_pks[:5]:
    check(f"document {pk}", lambda d=pk: svc.render_fragment("document", {"document_pk": d}))

print("\n=== live state ===")
check("status", svc.status)
check("machine panel", svc._machine)
check("capability model", svc._capability_model)

print()
print(f"RESULT: {'PASS' if not failures else 'FAIL - ' + str(len(failures)) + ' broken'}")
for label, exc in failures:
    print(f"  {label}: {type(exc).__name__}")
raise SystemExit(1 if failures else 0)
