"""Phase 2 stub: PostgreSQL upgrade path from SQLite.

Architecture hook for future implementation. When ready:
- Migrate schema from SQLite to PostgreSQL
- Use asyncpg for async database access
- Add connection pooling
- Migrate historical data
"""


async def migrate_sqlite_to_postgres(sqlite_path: str, postgres_dsn: str):
    """STUB — migrate all data from SQLite to PostgreSQL."""
    raise NotImplementedError("Phase 2: PostgreSQL migration not yet implemented")
