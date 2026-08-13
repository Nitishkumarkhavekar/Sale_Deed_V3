# `migrations/` — Alembic

`env.py` plus one file per revision under `versions/`. `alembic.ini` at the
project root points here:

```ini
script_location = %(here)s/src/migrations
prepend_sys_path = src
```

```
py -3.13 src/tools/db_setup.py --check      compare the schema to head
py -3.13 src/tools/db_setup.py --upgrade    apply pending revisions
```

The launcher applies pending revisions at startup, so a normal run needs neither
command.

## Two rules

**Never run `alembic init`.** Alembic is already configured and has applied
revisions; re-initialising would orphan the migration history.

**Models first, revision second.** Tables live in `core/db/models.py`;
autogenerate reads them through `env.py`. Editing a revision to add a column
that the model does not have produces a schema no code can use.
