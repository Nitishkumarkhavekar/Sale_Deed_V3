"""failure events

Append-only diagnosis history. A retry clears `documents.failure_reason`, so a
document that failed three different ways showed only the last one and the
sequence that would explain it was gone. This table keeps every verdict.

Purely additive: no existing column changes, and an empty table is the correct
state for every document that has already been processed.

Revision ID: c8f42a1d6b30
Revises: a7c31f9b2e04
Create Date: 2026-08-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c8f42a1d6b30'
down_revision: Union[str, Sequence[str], None] = 'a7c31f9b2e04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'failure_events',
        sa.Column('id', sa.BigInteger().with_variant(sa.Integer, 'sqlite'),
                  nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        # Denormalised so a batch-wide report needs no join through documents.
        sa.Column('batch_id', sa.Integer(), nullable=True),
        sa.Column('stage', sa.String(length=24), nullable=False),
        sa.Column('code', sa.String(length=48), nullable=False),
        sa.Column('message', sa.Text(), nullable=True),
        sa.Column('technical', sa.Text(), nullable=True),
        sa.Column('attempt', sa.Integer(), nullable=False,
                  server_default=sa.text('1')),
        sa.Column('retryable', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        # CASCADE with the document: the history of a deleted document is not
        # worth orphaning. SET NULL on the batch so a purged batch leaves the
        # per-document history intact.
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'],
                                name='fk_failure_events_document_id_documents',
                                ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['batch_id'], ['batches.id'],
                                name='fk_failure_events_batch_id_batches',
                                ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id', name='pk_failure_events'),
    )
    op.create_index('ix_failure_events_document', 'failure_events',
                    ['document_id', 'created_at'])
    op.create_index('ix_failure_events_batch_code', 'failure_events',
                    ['batch_id', 'code'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_failure_events_batch_code', table_name='failure_events')
    op.drop_index('ix_failure_events_document', table_name='failure_events')
    op.drop_table('failure_events')
