from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from crypto_quant.config.settings import get_settings


def get_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or get_settings().database_url, future=True)


def get_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    return sessionmaker(get_engine(database_url), expire_on_commit=False, future=True)


def session_scope(database_url: str | None = None) -> Iterator[Session]:
    factory = get_session_factory(database_url)
    with factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
