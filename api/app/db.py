from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

pool = ConnectionPool(conninfo=settings.database_url, max_size=10, kwargs={"autocommit": True})


def fetch_all(query: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> list[dict]:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            return list(cur.fetchall())


def fetch_one(query: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> dict | None:
    with pool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            result = cur.fetchone()
            return dict(result) if result else None


def execute(query: str, params: Mapping[str, Any] | Sequence[Any] | None = None) -> None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)


def execute_many(query: str, rows: Iterable[Sequence[Any]]) -> None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
