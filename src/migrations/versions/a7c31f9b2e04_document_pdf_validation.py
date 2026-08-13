"""document pdf validation

Records what a failed document's PDF actually is: corrupt, incomplete,
encrypted, or perfectly fine with the failure lying elsewhere. Columns on
`documents` rather than a new table - one row of facts about one document,
read on the screens that already load it.

Every column is nullable. NULL means "never validated", which is the correct
state for every existing row and for every document that processed cleanly:
validation runs only after a failure.

Revision ID: a7c31f9b2e04
Revises: 45243245c28d
Create Date: 2026-08-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a7c31f9b2e04'
down_revision: Union[str, Sequence[str], None] = '45243245c28d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('documents',
                  sa.Column('validation_status', sa.String(length=32), nullable=True))
    op.add_column('documents',
                  sa.Column('validation_error_code', sa.String(length=48), nullable=True))
    op.add_column('documents',
                  sa.Column('validation_error_message', sa.Text(), nullable=True))
    op.add_column('documents',
                  sa.Column('corrupted_pages', sa.String(length=500), nullable=True))
    op.add_column('documents',
                  sa.Column('validated_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('documents',
                  sa.Column('validator_version', sa.String(length=16), nullable=True))
    op.add_column('documents',
                  sa.Column('is_retryable', sa.Boolean(), nullable=True))
    # The corrupted-PDF list filters on this and nothing else.
    op.create_index('ix_documents_validation_status', 'documents',
                    ['validation_status'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_documents_validation_status', table_name='documents')
    for column in ('is_retryable', 'validator_version', 'validated_at',
                   'corrupted_pages', 'validation_error_message',
                   'validation_error_code', 'validation_status'):
        op.drop_column('documents', column)
