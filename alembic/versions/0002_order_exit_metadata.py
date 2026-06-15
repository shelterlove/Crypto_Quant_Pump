"""add structured order exit metadata

Revision ID: 0002_order_exit_metadata
Revises: 0001_initial_schema
Create Date: 2026-06-09
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision = "0002_order_exit_metadata"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("orders", sa.Column("mechanism", sa.String(64), nullable=True))
    op.add_column("orders", sa.Column("trigger", sa.String(128), nullable=True))
    op.add_column("orders", sa.Column("details", sa.JSON(), nullable=True))


def downgrade() -> None:
    columns: Sequence[str] = ("details", "trigger", "mechanism")
    for column in columns:
        op.drop_column("orders", column)
