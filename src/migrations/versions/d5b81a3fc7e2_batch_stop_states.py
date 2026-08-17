"""Add the STOPPING and STOPPED batch states.

Stopping a batch is not instantaneous: the document in flight is allowed to
finish so its partial work is not thrown away, and on a long deed that is
minutes. `STOPPING` is the interval between the operator asking and the work
actually ceasing; without it the UI must either claim the batch stopped while a
worker is still inside it, or show no change at all and look broken.

`STOPPED` is deliberately separate from `FAILED` and from `COMPLETED`: a stopped
batch is healthy and resumable, and lumping it in with either would make the
dashboard's counts wrong and hide the Run action behind the wrong condition.

PostgreSQL cannot add an enum value inside a transaction that then uses it, so
the ADD VALUEs run in an autocommit block. They are idempotent (`IF NOT EXISTS`)
because a half-applied migration on this project's single-machine deployments is
a real possibility, and re-running must not be a hard error.

Revision ID: d5b81a3fc7e2
Revises: c8f42a1d6b30
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = 'd5b81a3fc7e2'
down_revision: Union[str, Sequence[str], None] = 'c8f42a1d6b30'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: The type as it stood before this migration, needed to rebuild it on
#: downgrade - PostgreSQL has no ALTER TYPE ... DROP VALUE.
PREVIOUS_VALUES = ("queued", "running", "paused", "completed", "failed")

NEW_VALUES = ("stopping", "stopped")


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        # SQLite renders this enum as VARCHAR with a CHECK constraint, and the
        # verification schema is built from the models rather than from
        # migrations, so it already has the new members.
        return
    with op.get_context().autocommit_block():
        for value in NEW_VALUES:
            op.execute(
                f"ALTER TYPE batch_state ADD VALUE IF NOT EXISTS '{value}'")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return

    # Any batch actually sitting in one of the removed states becomes PAUSED,
    # which is what STOPPED means to the older code. Data is preserved rather
    # than the downgrade failing on rows it cannot represent.
    op.execute("UPDATE batches SET state = 'paused' "
               "WHERE state IN ('stopping', 'stopped')")

    values = ", ".join(f"'{v}'" for v in PREVIOUS_VALUES)
    op.execute("ALTER TYPE batch_state RENAME TO batch_state_old")
    op.execute(f"CREATE TYPE batch_state AS ENUM ({values})")
    op.execute("ALTER TABLE batches ALTER COLUMN state TYPE batch_state "
               "USING state::text::batch_state")
    op.execute("DROP TYPE batch_state_old")
