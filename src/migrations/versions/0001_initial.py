"""initial schema

Creates the ten tables of the Sale Deed system on PostgreSQL.

Generated offline: Alembic's --autogenerate requires a live connection, so the
operations were built from Base.metadata and rendered with Alembic's own
renderer. The result is identical to what autogenerate would emit against an
empty database.

Revision ID: 0001_initial
Revises: None
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

# Native ENUM types are created implicitly by create_table and must be dropped
# explicitly on downgrade - PostgreSQL does not remove them with the table.
ENUM_TYPES = ['batch_state', 'stage_state', 'document_state', 'person_relation']


def upgrade() -> None:
    op.create_table('settings',
    sa.Column('key', sa.String(length=120), nullable=False),
    sa.Column('value', sa.Text(), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('key', name=op.f('pk_settings'))
    )
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(length=150), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_users')),
    sa.UniqueConstraint('username', name=op.f('uq_users_username'))
    )
    op.create_table('batches',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=200), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('state', sa.Enum('queued', 'running', 'paused', 'completed', 'failed', name='batch_state'), nullable=False),
    sa.Column('queue_position', sa.Integer(), nullable=False),
    sa.Column('file_count', sa.Integer(), nullable=False),
    sa.Column('total_bytes', sa.BigInteger(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint('file_count >= 0', name=op.f('ck_batches_file_count_non_negative')),
    sa.CheckConstraint('total_bytes >= 0', name=op.f('ck_batches_total_bytes_non_negative')),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], name=op.f('fk_batches_user_id_users')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_batches'))
    )
    op.create_index(op.f('ix_batches_created_at'), 'batches', ['created_at'], unique=False)
    op.create_index('ix_batches_state_position', 'batches', ['state', 'queue_position'], unique=False)
    op.create_index(op.f('ix_batches_user_id'), 'batches', ['user_id'], unique=False)
    op.create_table('documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('batch_id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.String(length=100), nullable=False),
    sa.Column('source_filename', sa.String(length=500), nullable=False),
    sa.Column('source_path', sa.String(length=1000), nullable=True),
    sa.Column('page_count', sa.Integer(), nullable=False),
    sa.Column('size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('ocr_state', sa.Enum('pending', 'running', 'done', 'failed', 'skipped', name='stage_state'), nullable=False),
    sa.Column('extract_state', sa.Enum('pending', 'running', 'done', 'failed', 'skipped', name='stage_state'), nullable=False),
    sa.Column('translate_state', sa.Enum('pending', 'running', 'done', 'failed', 'skipped', name='stage_state'), nullable=False),
    sa.Column('validate_state', sa.Enum('pending', 'running', 'done', 'failed', 'skipped', name='stage_state'), nullable=False),
    sa.Column('ocr_attempts', sa.Integer(), nullable=False),
    sa.Column('extract_attempts', sa.Integer(), nullable=False),
    sa.Column('overall_state', sa.Enum('processing', 'processed', 'failed', 'needs_review', name='document_state'), nullable=False),
    sa.Column('processing_status', sa.String(length=20), nullable=True),
    sa.Column('failure_reason', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['batch_id'], ['batches.id'], name=op.f('fk_documents_batch_id_batches'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_documents')),
    sa.UniqueConstraint('batch_id', 'document_id', name='batch_document')
    )
    op.create_index('ix_documents_batch_overall', 'documents', ['batch_id', 'overall_state'], unique=False)
    op.create_index('ix_documents_resume', 'documents', ['batch_id', 'ocr_state', 'extract_state', 'translate_state'], unique=False)
    op.create_table('extractions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('attempt', sa.Integer(), nullable=False),
    sa.Column('raw_output', sa.Text(), nullable=True),
    sa.Column('parsed_ok', sa.Boolean(), nullable=False),
    sa.Column('pan_coverage', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('prompt_tokens', sa.Integer(), nullable=False),
    sa.Column('completion_tokens', sa.Integer(), nullable=False),
    sa.Column('truncated', sa.Boolean(), nullable=False),
    sa.Column('model_name', sa.String(length=200), nullable=True),
    sa.Column('quantisation', sa.String(length=40), nullable=True),
    sa.Column('prompt_name', sa.String(length=100), nullable=True),
    sa.Column('duration_s', sa.Numeric(precision=10, scale=3), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_extractions_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_extractions')),
    sa.UniqueConstraint('document_id', 'attempt', name='document_attempt')
    )
    op.create_index(op.f('ix_extractions_document_id'), 'extractions', ['document_id'], unique=False)
    op.create_table('logs',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), nullable=False),
    sa.Column('level', sa.String(length=12), nullable=False),
    sa.Column('logger', sa.String(length=120), nullable=True),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('batch_id', sa.Integer(), nullable=True),
    sa.Column('document_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['batch_id'], ['batches.id'], name=op.f('fk_logs_batch_id_batches'), ondelete='SET NULL'),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_logs_document_id_documents'), ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_logs'))
    )
    op.create_index(op.f('ix_logs_created_at'), 'logs', ['created_at'], unique=False)
    op.create_index('ix_logs_level_created', 'logs', ['level', 'created_at'], unique=False)
    op.create_table('ocr_pages',
    sa.Column('id', sa.BigInteger().with_variant(sa.Integer(), 'sqlite'), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('page_number', sa.Integer(), nullable=False),
    sa.Column('text', sa.Text(), nullable=False),
    sa.Column('char_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_ocr_pages_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ocr_pages')),
    sa.UniqueConstraint('document_id', 'page_number', name='document_page')
    )
    op.create_index(op.f('ix_ocr_pages_created_at'), 'ocr_pages', ['created_at'], unique=False)
    op.create_table('persons',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('relation', sa.Enum('B', 'S', name='person_relation'), nullable=False),
    sa.Column('ordinal', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=500), nullable=True),
    sa.Column('name_translated', sa.String(length=500), nullable=True),
    sa.Column('gender', sa.String(length=20), nullable=True),
    sa.Column('father_name', sa.String(length=500), nullable=True),
    sa.Column('father_name_translated', sa.String(length=500), nullable=True),
    sa.Column('aadhaar_number', sa.String(length=12), nullable=True),
    sa.Column('pan_card_number', sa.String(length=10), nullable=True),
    sa.Column('address', sa.Text(), nullable=True),
    sa.Column('address_translated', sa.Text(), nullable=True),
    sa.Column('state', sa.String(length=100), nullable=True),
    sa.Column('postal_code', sa.String(length=10), nullable=True),
    sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('remarks', sa.String(length=500), nullable=True),
    sa.CheckConstraint('aadhaar_number IS NULL OR length(aadhaar_number) = 12', name=op.f('ck_persons_aadhaar_twelve_digits')),
    sa.CheckConstraint('pan_card_number IS NULL OR length(pan_card_number) = 10', name=op.f('ck_persons_pan_ten_chars')),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_persons_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_persons')),
    sa.UniqueConstraint('document_id', 'relation', 'ordinal', name='document_relation_ordinal')
    )
    op.create_index('ix_persons_pan', 'persons', ['pan_card_number'], unique=False)
    op.create_table('properties',
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('schedule_c_address', sa.Text(), nullable=True),
    sa.Column('address_translated', sa.Text(), nullable=True),
    sa.Column('state', sa.String(length=100), nullable=True),
    sa.Column('postal_code', sa.String(length=10), nullable=True),
    sa.Column('sale_consideration', sa.Numeric(precision=15, scale=0), nullable=True),
    sa.Column('registration_fee', sa.Numeric(precision=15, scale=0), nullable=True),
    sa.Column('stamp_value', sa.Numeric(precision=15, scale=0), nullable=True),
    sa.Column('paid_in_cash', sa.Boolean(), nullable=True),
    sa.Column('transaction_date', sa.Date(), nullable=True),
    sa.Column('registration_office', sa.String(length=300), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_properties_document_id_documents'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('document_id', name=op.f('pk_properties'))
    )
    op.create_table('validation_results',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('person_id', sa.Integer(), nullable=True),
    sa.Column('flag_code', sa.String(length=12), nullable=False),
    sa.Column('field', sa.String(length=80), nullable=True),
    sa.Column('detail', sa.Text(), nullable=True),
    sa.Column('confidence', sa.Numeric(precision=4, scale=3), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], name=op.f('fk_validation_results_document_id_documents'), ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['person_id'], ['persons.id'], name=op.f('fk_validation_results_person_id_persons'), ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_validation_results'))
    )
    op.create_index('ix_validation_document_flag', 'validation_results', ['document_id', 'flag_code'], unique=False)
    op.create_index(op.f('ix_validation_results_document_id'), 'validation_results', ['document_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_validation_document_flag', table_name='validation_results')
    op.drop_index(op.f('ix_validation_results_document_id'), table_name='validation_results')
    op.drop_table('validation_results')
    op.drop_table('properties')
    op.drop_index('ix_persons_pan', table_name='persons')
    op.drop_table('persons')
    op.drop_index(op.f('ix_ocr_pages_created_at'), table_name='ocr_pages')
    op.drop_table('ocr_pages')
    op.drop_index(op.f('ix_logs_created_at'), table_name='logs')
    op.drop_index('ix_logs_level_created', table_name='logs')
    op.drop_table('logs')
    op.drop_index(op.f('ix_extractions_document_id'), table_name='extractions')
    op.drop_table('extractions')
    op.drop_index('ix_documents_batch_overall', table_name='documents')
    op.drop_index('ix_documents_resume', table_name='documents')
    op.drop_table('documents')
    op.drop_index(op.f('ix_batches_created_at'), table_name='batches')
    op.drop_index('ix_batches_state_position', table_name='batches')
    op.drop_index(op.f('ix_batches_user_id'), table_name='batches')
    op.drop_table('batches')
    op.drop_table('users')
    op.drop_table('settings')
    for name in ENUM_TYPES:
        op.execute(f"DROP TYPE IF EXISTS {name}")
