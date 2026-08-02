import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from models_v2 import AnalysisResult, CategoryV2, InsightResult, TasteEntry

DB_PATH = Path(__file__).parent / "data" / "tastemap_v2.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_input               TEXT NOT NULL,
                rating                  INTEGER NOT NULL,
                input_type              TEXT NOT NULL,
                category_json           TEXT NOT NULL,
                contexts_json           TEXT NOT NULL,
                recommended_tags_json   TEXT NOT NULL,
                selected_tags_json      TEXT NOT NULL,
                one_line_insight        TEXT NOT NULL,
                needs_more_context      INTEGER NOT NULL DEFAULT 0,
                context_question        TEXT,
                created_at              TEXT NOT NULL,
                updated_at              TEXT NOT NULL
            )
        """)
        # 새 컬럼 마이그레이션
        existing = {row[1] for row in conn.execute("PRAGMA table_info(entries)").fetchall()}
        migrations = [
            ("source_tags_json",        "TEXT"),
            ("feature_tags_json",       "TEXT"),
            ("preference_tags_json",    "TEXT"),
            ("candidate_tags_json",     "TEXT"),
            ("selected_tag_types_json", "TEXT"),
            ("knowledge_confidence",    "TEXT"),
            ("is_named_entity",         "INTEGER"),
            ("key_context_json",         "TEXT"),
            ("preference_rule",          "TEXT"),
            ("preference_contrast",      "TEXT"),
            ("insight_depth",            "TEXT"),
            ("insight_mode",             "TEXT"),
            ("abstraction_level",        "TEXT"),
            ("pattern_insight",          "TEXT"),
            ("pattern_confidence",       "TEXT"),
            ("related_entry_ids_json",   "TEXT"),
        ]
        for col, typ in migrations:
            if col not in existing:
                conn.execute(f"ALTER TABLE entries ADD COLUMN {col} {typ}")


def add_entry(
    raw_input: str,
    rating: int,
    analysis: AnalysisResult,
    selected_tags: list[str],
    insight: InsightResult,
    selected_tag_types: dict[str, str] | None = None,
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO entries
                (raw_input, rating, input_type, category_json, contexts_json,
                 recommended_tags_json, selected_tags_json, one_line_insight,
                 needs_more_context, context_question, created_at, updated_at,
                 source_tags_json, feature_tags_json, preference_tags_json,
                 candidate_tags_json, selected_tag_types_json, knowledge_confidence,
                 key_context_json, preference_rule, preference_contrast,
                 insight_depth, insight_mode, abstraction_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_input, rating, analysis.input_type,
                json.dumps(analysis.category.model_dump(), ensure_ascii=False),
                json.dumps(analysis.contexts, ensure_ascii=False),
                json.dumps(analysis.recommended_tags, ensure_ascii=False),
                json.dumps(selected_tags, ensure_ascii=False),
                insight.one_line_insight,
                1 if analysis.needs_more_context else 0,
                analysis.context_question, now, now,
                json.dumps(analysis.source_tags, ensure_ascii=False),
                json.dumps(analysis.feature_tags, ensure_ascii=False),
                json.dumps(analysis.preference_tags, ensure_ascii=False),
                json.dumps(analysis.candidate_tags, ensure_ascii=False),
                json.dumps(selected_tag_types or {}, ensure_ascii=False),
                analysis.knowledge_confidence,
                json.dumps(insight.key_context, ensure_ascii=False),
                insight.preference_rule,
                insight.preference_contrast,
                insight.insight_depth,
                insight.insight_mode,
                insight.abstraction_level,
            ),
        )
        return cursor.lastrowid


def update_full_entry(
    entry_id: int,
    raw_input: str,
    rating: int,
    analysis: AnalysisResult,
    selected_tags: list[str],
    insight: InsightResult,
    selected_tag_types: dict[str, str] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE entries
            SET raw_input               = ?,
                rating                  = ?,
                input_type              = ?,
                category_json           = ?,
                contexts_json           = ?,
                recommended_tags_json   = ?,
                selected_tags_json      = ?,
                one_line_insight        = ?,
                needs_more_context      = ?,
                context_question        = ?,
                updated_at              = ?,
                source_tags_json        = ?,
                feature_tags_json       = ?,
                preference_tags_json    = ?,
                candidate_tags_json     = ?,
                selected_tag_types_json = ?,
                knowledge_confidence    = ?,
                key_context_json        = ?,
                preference_rule         = ?,
                preference_contrast     = ?,
                insight_depth           = ?,
                insight_mode            = ?,
                abstraction_level       = ?
            WHERE id = ?
            """,
            (
                raw_input, rating, analysis.input_type,
                json.dumps(analysis.category.model_dump(), ensure_ascii=False),
                json.dumps(analysis.contexts, ensure_ascii=False),
                json.dumps(analysis.recommended_tags, ensure_ascii=False),
                json.dumps(selected_tags, ensure_ascii=False),
                insight.one_line_insight,
                1 if analysis.needs_more_context else 0,
                analysis.context_question, now,
                json.dumps(analysis.source_tags, ensure_ascii=False),
                json.dumps(analysis.feature_tags, ensure_ascii=False),
                json.dumps(analysis.preference_tags, ensure_ascii=False),
                json.dumps(analysis.candidate_tags, ensure_ascii=False),
                json.dumps(selected_tag_types or {}, ensure_ascii=False),
                analysis.knowledge_confidence,
                json.dumps(insight.key_context, ensure_ascii=False),
                insight.preference_rule,
                insight.preference_contrast,
                insight.insight_depth,
                insight.insight_mode,
                insight.abstraction_level,
                entry_id,
            ),
        )


def update_tags_and_insight(
    entry_id: int,
    selected_tags: list[str],
    insight: InsightResult,
    selected_tag_types: dict[str, str] | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE entries
            SET selected_tags_json      = ?,
                one_line_insight        = ?,
                selected_tag_types_json = ?,
                key_context_json        = ?,
                preference_rule         = ?,
                preference_contrast     = ?,
                insight_depth           = ?,
                insight_mode            = ?,
                updated_at              = ?
            WHERE id = ?
            """,
            (
                json.dumps(selected_tags, ensure_ascii=False),
                insight.one_line_insight,
                json.dumps(selected_tag_types or {}, ensure_ascii=False),
                json.dumps(insight.key_context, ensure_ascii=False),
                insight.preference_rule,
                insight.preference_contrast,
                insight.insight_depth,
                insight.insight_mode,
                now, entry_id,
            ),
        )


def update_insight_only(entry_id: int, insight: InsightResult) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE entries
            SET one_line_insight    = ?,
                key_context_json    = ?,
                preference_rule     = ?,
                preference_contrast = ?,
                insight_depth       = ?,
                insight_mode        = ?,
                abstraction_level   = ?,
                updated_at          = ?
            WHERE id = ?
            """,
            (insight.one_line_insight,
             json.dumps(insight.key_context, ensure_ascii=False),
             insight.preference_rule,
             insight.preference_contrast,
             insight.insight_depth,
             insight.insight_mode,
             insight.abstraction_level,
             now, entry_id),
        )


def update_pattern_insight(
    entry_id: int,
    pattern_insight: str,
    pattern_confidence: str,
    related_entry_ids: list[int],
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE entries
            SET pattern_insight        = ?,
                pattern_confidence     = ?,
                related_entry_ids_json = ?,
                updated_at             = ?
            WHERE id = ?
            """,
            (pattern_insight, pattern_confidence,
             json.dumps(related_entry_ids, ensure_ascii=False), now, entry_id),
        )


def delete_entry(entry_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))


def list_entries() -> list[TasteEntry]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM entries ORDER BY id DESC").fetchall()
    return [_row_to_entry(row) for row in rows]


def get_entry(entry_id: int) -> Optional[TasteEntry]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
    return _row_to_entry(row) if row else None


def _jl(val) -> list:
    if not val:
        return []
    try:
        r = json.loads(val)
        return r if isinstance(r, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _jd(val) -> dict:
    if not val:
        return {}
    try:
        r = json.loads(val)
        return r if isinstance(r, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _safe(row: sqlite3.Row, col: str, default=None):
    try:
        return row[col]
    except IndexError:
        return default


def _row_to_entry(row: sqlite3.Row) -> TasteEntry:
    cat_data = _jd(_safe(row, "category_json"))
    confidence = _safe(row, "knowledge_confidence") or "medium"
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    return TasteEntry(
        id=row["id"],
        raw_input=row["raw_input"],
        rating=row["rating"],
        input_type=row["input_type"],
        category=CategoryV2(**cat_data) if isinstance(cat_data, dict) else CategoryV2(),
        contexts=_jl(row["contexts_json"]),
        source_tags=_jl(_safe(row, "source_tags_json")),
        feature_tags=_jl(_safe(row, "feature_tags_json")),
        preference_tags=_jl(_safe(row, "preference_tags_json")),
        candidate_tags=_jl(_safe(row, "candidate_tags_json")),
        recommended_tags=_jl(row["recommended_tags_json"]),
        selected_tags=_jl(row["selected_tags_json"]),
        selected_tag_types=_jd(_safe(row, "selected_tag_types_json")),
        one_line_insight=row["one_line_insight"] or "",
        key_context=_jl(_safe(row, "key_context_json")),
        preference_rule=_safe(row, "preference_rule") or "",
        preference_contrast=_safe(row, "preference_contrast") or "",
        abstraction_level=_safe(row, "abstraction_level") or "moderate",
        insight_depth=_safe(row, "insight_depth") or "medium",
        insight_mode=_safe(row, "insight_mode") or "contrast",
        pattern_insight=_safe(row, "pattern_insight") or "",
        pattern_confidence=_safe(row, "pattern_confidence") or "none",
        related_entry_ids=_jl(_safe(row, "related_entry_ids_json")),
        needs_more_context=bool(row["needs_more_context"]),
        context_question=_safe(row, "context_question"),
        knowledge_confidence=confidence,
        created_at=row["created_at"] or "",
        updated_at=row["updated_at"] or "",
    )
