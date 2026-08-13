# Database Schema

**Last updated:** 2026-07-31 17:15 IST
**Status:** **LIVE.** Models, engine, repositories and migrations are applied to
PostgreSQL 17.10 and covered by 36 tests (schema shape, DDL rendering,
transactions, concurrent claiming, cascades, idempotency, crash recovery).

One defect was found here by testing and fixed on 2026-07-31 (R-010):
`statement_timeout` was reverted by the connection pool's `ROLLBACK`, so only the
first query on each pooled connection was bounded. It is now passed as a libpq
startup option.

Stack is mandated (ADR-012): **SQLAlchemy 2.0 + psycopg v3 + Alembic** on
PostgreSQL. DSN scheme is `postgresql+psycopg://`; `normalise_dsn()` rewrites
`postgresql://` and `postgresql+psycopg2://` onto the v3 driver, because only
psycopg v3 is installed and the resulting ImportError is otherwise obscure.

| Module | Contents |
|---|---|
| `src/core/db/models.py` | 10 tables, 97 columns, 4 native ENUM types |
| `src/core/db/engine.py` | engine, session scope, offline DDL rendering, health probe |
| `src/core/db/repositories.py` | 8 repositories behind a `UnitOfWork` |
| `src/migrations/` | Alembic env + `0001_initial` |

---

## Design constraints driving the schema

1. **Continuous commit.** Every processed document updates the database
   immediately. Nothing is deferred to batch end — the application must survive
   crash, shutdown and power failure with no data loss.
2. **Per-stage resume.** Stage state is tracked per document, not per batch, so a
   restart never re-runs completed work and never duplicates it.
3. **One row per person in CSV export.** `example.csv` repeats document-level
   fields across each party's row. The database stores this normalised and the
   exporter denormalises.
4. **OCR text is transient.** Deleted after 30 days, never backed up. Only
   filename, path and metadata persist.
5. **Yearly rotation.** Prior year archived to `/backup` then purged, so tables
   need a partitionable time column.

---

## Tables

### `users`
| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | |
| `username` | text unique | supplied at upload |
| `created_at` | timestamptz | |

### `batches`
| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | |
| `name` | text | operator-supplied |
| `user_id` | FK users | |
| `state` | enum | `queued`, `running`, `paused`, `completed`, `failed` |
| `queue_position` | int | max 4 queued |
| `file_count` | int | auto-detected |
| `total_bytes` | bigint | max 25 GB |
| `created_at` | timestamptz | auto |
| `started_at` / `finished_at` | timestamptz null | |

Constraint: at most 4 batches in `queued`. Enforced in the repository, not by a
DB constraint, so the error message can be meaningful.

### `documents`
The resume unit. One row per PDF.

| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | |
| `batch_id` | FK batches | |
| `document_id` | text | registration number, e.g. `1467-2025-26` |
| `source_filename` | text | |
| `page_count` | int | |
| `ocr_state` | enum | `pending`, `running`, `done`, `failed` |
| `extract_state` | enum | same |
| `translate_state` | enum | same |
| `validate_state` | enum | same |
| `ocr_attempts` | int | capped at 1 retry |
| `extract_attempts` | int | capped at 1 retry |
| `overall_state` | enum | `processing`, `processed`, `failed`, `needs_review` |
| `processing_status` | text | `OCR_P` / `OCR_F` flag codes |
| `updated_at` | timestamptz | |

Index on `(batch_id, overall_state)` — drives dashboard counts.
Index on `(batch_id, ocr_state, extract_state, translate_state)` — drives the
resume scan on startup.

### `ocr_pages`
Transient. Purged after 30 days; never backed up.

| Column | Type | Notes |
|---|---|---|
| `id` | bigserial PK | |
| `document_id` | FK documents | |
| `page_number` | int | |
| `text` | text | cleaned OCR, LF line endings only (ADR-005) |
| `char_count` | int | |
| `created_at` | timestamptz | drives the 30-day expiry |

**Images are never stored** — only filename, path and metadata, per spec.

### `extractions`
Raw model output plus accounting, retained for audit.

| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | |
| `document_id` | FK documents | |
| `attempt` | int | 1 or 2 |
| `raw_output` | text | model text before parsing |
| `parsed_ok` | bool | |
| `pan_coverage` | numeric(4,3) | retry trigger, threshold 0.6 |
| `prompt_tokens` / `completion_tokens` | int | |
| `truncated` | bool | hit token ceiling — usually a repetition loop |
| `model_name` | text | e.g. `deeds-v6_7-Q4_K_M.gguf` |
| `quantisation` | text | recorded because it affects accuracy (L-001) |
| `duration_s` | numeric | |
| `created_at` | timestamptz | |

Storing `quantisation` per extraction matters: when the accuracy comparison is
finally run, results must be attributable to a specific precision.

### `properties`
One row per document.

| Column | Type | Notes |
|---|---|---|
| `document_id` | FK documents PK | |
| `schedule_c_address` | text | |
| `state` | text | English |
| `sale_consideration` | numeric(15,0) | plain integer |
| `registration_fee` | numeric(15,0) | |
| `stamp_value` | numeric(15,0) null | passthrough; rule disabled (ADR-010) |
| `paid_in_cash` | bool | never null |
| `transaction_date` | date | |
| `registration_office` | text | |

### `persons`
| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | |
| `document_id` | FK documents | |
| `relation` | enum | `B` buyer / `S` seller — matches CSV |
| `ordinal` | int | stable ordering for export |
| `name` | text | original script |
| `name_translated` | text null | transliterated |
| `gender` | text null | |
| `father_name` | text null | |
| `aadhaar_number` | char(12) null | digits only |
| `pan_card_number` | char(10) null | `^[A-Z]{5}[0-9]{4}[A-Z]$` |
| `address` | text null | original script |
| `address_translated` | text null | translated, not transliterated |
| `state` | text null | English |

Aadhaar and PAN are stored as fixed-width text, never numeric — `example.csv`
already shows the cost of numeric coercion (`6.63E+11`).

### `validation_results`
| Column | Type | Notes |
|---|---|---|
| `id` | serial PK | |
| `document_id` | FK documents | |
| `person_id` | FK persons null | null for document-level findings |
| `flag_code` | text | `PM`, `WAN`, `WSC`, `WSV`, `SSV`, `WTD` |
| `field` | text | |
| `detail` | text | human-readable reason |
| `confidence` | numeric(4,3) | rule-derived (ADR-008) |
| `created_at` | timestamptz | |

Two remarks columns in the CSV are assembled from this table: document-level rows
where `person_id is null`, person-level rows otherwise.

### `settings`
Key-value, single row per key. Mirrors the Settings page and `.env` overrides.

### `logs`
Structured application log. Written only when `DEBUG=true`; the handler is not
installed otherwise, so it costs nothing when disabled.

---

## Migrations

| Revision | Date | Description |
|---|---|---|
| `0001_initial` | 2026-07-30 | Ten tables, 4 ENUM types, 12 indexes, 10 FKs, 4 CHECKs |

### Generated offline

`alembic revision --autogenerate` requires a live connection - it raises an
`AssertionError` without one. Since no server exists, the migration was built from
`Base.metadata` using Alembic's own operation objects (`CreateTableOp.from_table`,
`CreateIndexOp.from_index`) and rendered with `render_python_code`. The output is
what autogenerate would have produced against an empty database.

### Verified without a database

```bash
# emits the full DDL, no connection
alembic upgrade head --sql

# offline downgrade needs an explicit range: Alembic cannot know the
# current revision without querying alembic_version
alembic downgrade 0001_initial:base --sql
```

| | upgrade | downgrade |
|---|---:|---:|
| CREATE/DROP TYPE | 4 | 4 |
| CREATE/DROP TABLE | 11 | 11 |
| CREATE/DROP INDEX | 12 | 12 |
| FOREIGN KEY | 10 | — |
| CHECK | 4 | — |
| UNIQUE | 5 | — |

Table count is 11 rather than 10 because Alembic adds `alembic_version`.

### ENUM types are dropped explicitly

PostgreSQL does **not** remove a native ENUM type when the table using it is
dropped, so `downgrade()` ends with explicit statements:

```sql
DROP TYPE IF EXISTS batch_state;
DROP TYPE IF EXISTS stage_state;
DROP TYPE IF EXISTS document_state;
DROP TYPE IF EXISTS person_relation;
```

Without these, re-running `upgrade` after a `downgrade` fails with
"type already exists" - a downgrade that leaves the database un-upgradeable is
not a downgrade.

## Bring-up and verification: `src/tools/db_setup.py`

One command takes an empty PostgreSQL server to ready and then proves the stack.

```bash
python src/tools/db_setup.py --check      # probe connection, report driver
python src/tools/db_setup.py --upgrade    # alembic upgrade head
python src/tools/db_setup.py --seed       # default settings rows
python src/tools/db_setup.py --verify     # full round-trip against real tables
python src/tools/db_setup.py              # all four, in order
```

### Why `--verify` exists

The orchestration layer is otherwise **untested**. The repositories, the
`FOR UPDATE SKIP LOCKED` claim, stage ordering, crash recovery, idempotency and
continuous commit are all written and reviewed but have never executed - there has
been no server to execute them against. `--verify` runs them for real and deletes
its own data afterwards.

What it asserts:

| Check | Proves |
|---|---|
| create user / batch / documents | basic persistence and relationships |
| duplicate `document_id` skipped | no duplicate processing after a crash mid-upload |
| queue cap enforced | the four-batch ceiling raises a readable error |
| `claim_next(ocr)` with SKIP LOCKED | concurrent workers cannot take the same row |
| extract unclaimable before OCR done | stage ordering holds |
| `ocr_pages` saved | per-page storage for the 30-day expiry |
| crash recovery resets RUNNING | stranded documents become claimable |
| property + persons persisted | result writing |
| Aadhaar stored as 12-char text | no numeric coercion (the `6.63E+11` failure) |
| `replace_persons` idempotent | retries do not accumulate rows |
| progress aggregates | dashboard counts |
| pagination | 5-per-page batch listing |
| committed across sessions | continuous commit is real, not deferred |

### Verified — 15 passed, 0 failed

```bash
python src/tools/db_setup.py --sqlite      # runs today, no server needed
```

The suite executes against real tables and cleans up after itself. Two schema
changes were needed to make the schema creatable on SQLite, neither of which
alters PostgreSQL behaviour:

| Change | Reason |
|---|---|
| `char_length()` -> `length()` in CHECK constraints | identical on PostgreSQL; `char_length` does not exist in SQLite |
| `BigInteger` PKs -> `BigInteger().with_variant(Integer, "sqlite")` | SQLite only auto-increments an `INTEGER PRIMARY KEY`; a BIGINT PK has no default and inserts fail. PostgreSQL still gets `BIGSERIAL` |

### Two real bugs the run exposed

Both in the "idempotent" repository methods, and both would have broken retries in
production.

`save_pages`, `replace_persons` and `record_flags` created child rows with an
explicit `document_id=doc.id` instead of appending to the parent relationship. The
already-loaded collection therefore went **stale**, so the delete-then-insert saw
nothing to delete:

* `save_pages` reported 0 pages after inserting 2.
* the second `replace_persons` violated
  `UNIQUE (document_id, relation, ordinal)`.

Retry safety was the entire purpose of those methods, so this was a genuine defect
- a document retried after a transient failure would have crashed on the unique
constraint. Fixed by appending through the relationship
(`doc.persons.append(...)`, `doc.ocr_pages.clear()`), letting the ORM keep the
collection in step and the delete-orphan cascade issue the DELETEs.

### What SQLite does and does not prove

**Proven:** repository logic, stage ordering, retry caps, the queue cap,
idempotency, cascades, CHECK constraints, pagination, progress aggregation, and
continuous commit across separate sessions.

**Not proven:** `FOR UPDATE SKIP LOCKED`. The SQLite dialect silently omits
`FOR UPDATE`, so the guarantee that two concurrent workers never claim the same
row is **still untested** and remains PostgreSQL-only. That is the one assertion in
the list whose name is accurate but whose semantics are not exercised.

### Against a real PostgreSQL server

```
driver: psycopg (psycopg 3.3.4)
[FAIL] OperationalError: (psycopg.errors.ConnectionTimeout) connection timeout expired
```

Expected and correct with no server running: it fails in ~5 s rather than hanging,
and explains why. `python src/tools/db_setup.py` runs check -> upgrade -> seed ->
verify against PostgreSQL and will confirm the concurrency behaviour too.

### Applying to a real server

```bash
set SALEDEED_DB_URL=postgresql+psycopg://user:pass@host:5432/saledeed
alembic upgrade head
```

The URL is read from the environment, never from `alembic.ini` - the template
ships the placeholder `driver://user:pass@localhost/dbname`, and the alternative
to reading the environment is committing a password.

---

## Open questions

- `stamp_value` semantics are undefined (ADR-010), so the column is a nullable
  passthrough rather than a computed value.
- Whether `extractions.raw_output` should be retained indefinitely or expire with
  the OCR cache. Audit value argues for keeping it; storage argues otherwise.
