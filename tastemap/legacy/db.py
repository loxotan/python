import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tastemap.legacy.models import PreferenceClassification, PreferenceEntry

DB_PATH = Path(__file__).parent / "data" / "tastemap.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def _existing_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(entries)").fetchall()
    return {row[1] for row in rows}


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id                          INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text                    TEXT NOT NULL,
                rating                      INTEGER NOT NULL,
                category                    TEXT NOT NULL,
                subcategories_json          TEXT NOT NULL,
                tags_json                   TEXT NOT NULL,
                preference_signal           TEXT NOT NULL,
                confidence                  REAL NOT NULL,
                created_at                  TEXT NOT NULL
            )
        """)

        existing = _existing_columns(conn)
        migrations = [
            ("context_categories_json", "TEXT"),
            ("object_tags_json",        "TEXT"),
            ("sensory_tags_json",       "TEXT"),
            ("context_tags_json",       "TEXT"),
            ("social_tags_json",        "TEXT"),
            ("place_tags_json",         "TEXT"),
            ("updated_at",              "TEXT"),
            ("event_date",              "TEXT"),
            ("event_date_precision",    "TEXT"),
            ("time_perspective",        "TEXT"),
        ]
        for col_name, col_type in migrations:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE entries ADD COLUMN {col_name} {col_type}")


def add_entry(
    raw_text: str,
    rating: int,
    classification: PreferenceClassification,
    event_date: str | None = None,
    event_date_precision: str = "none",
    time_perspective: str = "unspecified",
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO entries
                (raw_text, rating, category, subcategories_json, tags_json,
                 preference_signal, confidence, created_at,
                 context_categories_json,
                 object_tags_json, sensory_tags_json, context_tags_json,
                 social_tags_json, place_tags_json,
                 event_date, event_date_precision, time_perspective)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_text,
                rating,
                classification.primary_category,
                json.dumps(classification.subcategories, ensure_ascii=False),
                json.dumps(classification.tags, ensure_ascii=False),
                classification.preference_signal,
                classification.confidence,
                now,
                json.dumps(classification.context_categories, ensure_ascii=False),
                json.dumps(classification.object_tags, ensure_ascii=False),
                json.dumps(classification.sensory_tags, ensure_ascii=False),
                json.dumps(classification.context_tags, ensure_ascii=False),
                json.dumps(classification.social_tags, ensure_ascii=False),
                json.dumps(classification.place_tags, ensure_ascii=False),
                event_date,
                event_date_precision,
                time_perspective,
            ),
        )
        return cursor.lastrowid


def _load_json_list(value) -> list[str]:
    if value is None:
        return []
    try:
        result = json.loads(value)
        return result if isinstance(result, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def list_entries() -> list[PreferenceEntry]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM entries ORDER BY id DESC"
        ).fetchall()

    return [
        PreferenceEntry(
            id=row["id"],
            raw_text=row["raw_text"],
            rating=row["rating"],
            primary_category=row["category"],
            context_categories=_load_json_list(row["context_categories_json"]),
            subcategories=_load_json_list(row["subcategories_json"]),
            tags=_load_json_list(row["tags_json"]),
            object_tags=_load_json_list(_safe_col(row, "object_tags_json")),
            sensory_tags=_load_json_list(_safe_col(row, "sensory_tags_json")),
            context_tags=_load_json_list(_safe_col(row, "context_tags_json")),
            social_tags=_load_json_list(_safe_col(row, "social_tags_json")),
            place_tags=_load_json_list(_safe_col(row, "place_tags_json")),
            preference_signal=row["preference_signal"],
            confidence=row["confidence"],
            created_at=row["created_at"],
            event_date=_safe_col(row, "event_date"),
            event_date_precision=_safe_col(row, "event_date_precision") or "none",
            time_perspective=_safe_col(row, "time_perspective") or "unspecified",
        )
        for row in rows
    ]


def _safe_col(row: sqlite3.Row, col: str):
    try:
        return row[col]
    except IndexError:
        return None


def update_entry(
    entry_id: int,
    raw_text: str,
    rating: int,
    classification: PreferenceClassification,
    event_date: str | None = None,
    event_date_precision: str = "none",
    time_perspective: str = "unspecified",
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE entries
            SET raw_text                = ?,
                rating                  = ?,
                category                = ?,
                context_categories_json = ?,
                subcategories_json      = ?,
                tags_json               = ?,
                object_tags_json        = ?,
                sensory_tags_json       = ?,
                context_tags_json       = ?,
                social_tags_json        = ?,
                place_tags_json         = ?,
                preference_signal       = ?,
                confidence              = ?,
                updated_at              = ?,
                event_date              = ?,
                event_date_precision    = ?,
                time_perspective        = ?
            WHERE id = ?
            """,
            (
                raw_text,
                rating,
                classification.primary_category,
                json.dumps(classification.context_categories, ensure_ascii=False),
                json.dumps(classification.subcategories, ensure_ascii=False),
                json.dumps(classification.tags, ensure_ascii=False),
                json.dumps(classification.object_tags, ensure_ascii=False),
                json.dumps(classification.sensory_tags, ensure_ascii=False),
                json.dumps(classification.context_tags, ensure_ascii=False),
                json.dumps(classification.social_tags, ensure_ascii=False),
                json.dumps(classification.place_tags, ensure_ascii=False),
                classification.preference_signal,
                classification.confidence,
                now,
                event_date,
                event_date_precision,
                time_perspective,
                entry_id,
            ),
        )


def delete_entry(entry_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))


# ── Export helpers ────────────────────────────────────────────────────────────

_EXPORT_COLUMNS: list[str] = [
    "id", "raw_text", "rating", "category",
    "context_categories_json", "subcategories_json", "tags_json",
    "object_tags_json", "sensory_tags_json", "context_tags_json",
    "social_tags_json", "place_tags_json",
    "preference_signal", "confidence",
    "created_at", "updated_at",
    "event_date", "event_date_precision", "time_perspective",
]

_JSON_COLUMNS: set[str] = {
    "context_categories_json", "subcategories_json", "tags_json",
    "object_tags_json", "sensory_tags_json", "context_tags_json",
    "social_tags_json", "place_tags_json",
}


def _fetch_all_rows() -> list[sqlite3.Row]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT * FROM entries ORDER BY id ASC"
        ).fetchall()


def export_entries_for_json() -> list[dict]:
    """JSON export용: JSON 컬럼은 list로 파싱해 반환한다."""
    rows = _fetch_all_rows()
    result = []
    for row in rows:
        d: dict = {}
        for col in _EXPORT_COLUMNS:
            try:
                val = row[col]
            except IndexError:
                val = None
            if col in _JSON_COLUMNS:
                try:
                    parsed = json.loads(val) if val is not None else []
                    val = parsed if isinstance(parsed, list) else []
                except (json.JSONDecodeError, TypeError):
                    val = []
            d[col] = val
        result.append(d)
    return result


def export_entries_for_csv() -> list[dict]:
    """CSV export용: JSON 컬럼은 문자열 그대로 반환한다."""
    rows = _fetch_all_rows()
    result = []
    for row in rows:
        d: dict = {}
        for col in _EXPORT_COLUMNS:
            try:
                val = row[col]
            except IndexError:
                val = None
            d[col] = val if val is not None else ""
        result.append(d)
    return result


def get_db_backup_bytes() -> bytes:
    """현재 SQLite DB 파일을 bytes로 읽어 반환한다."""
    if not DB_PATH.exists():
        return b""
    try:
        return DB_PATH.read_bytes()
    except OSError:
        return b""
