# -*- coding: utf-8 -*-
"""
TasteMap smoke test — 주요 함수가 오류 없이 실행되는지 검증한다.
Run: python scripts/smoke_test.py
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

SEP = "=" * 60
passed = failed = 0

def check(name: str, fn):
    global passed, failed
    try:
        result = fn()
        print(f"[PASS] {name}")
        passed += 1
        return result
    except Exception as exc:
        print(f"[FAIL] {name}: {exc}")
        failed += 1
        return None

# ── 1. classifier ─────────────────────────────────────────────────────────────
print(SEP)
print("1. classifier")

from tastemap.legacy.ai_classifier import classify_preference_rule_based, _normalize_classification

def _test_sake():
    c = classify_preference_rule_based("사케 맛있다", 5)
    c = _normalize_classification(c, "사케 맛있다")
    assert c.primary_category == "음료", f"got {c.primary_category}"
    return c

def _test_factorio():
    c = classify_preference_rule_based("팩토리오", 5)
    c = _normalize_classification(c, "팩토리오")
    # rule-based에서는 미분류 허용, AI에서는 게임 예상
    assert c.primary_category in ("미분류", "게임"), f"got {c.primary_category}"
    return c

def _test_gbr():
    c = classify_preference_rule_based("오늘 GBR flap reflection이 생각보다 잘 됐다", 5)
    c = _normalize_classification(c, "오늘 GBR flap reflection이 생각보다 잘 됐다")
    assert c.primary_category == "직업", f"got {c.primary_category}"
    assert "GBR" in c.object_tags, f"object_tags={c.object_tags}"
    return c

def _test_perfume():
    c = classify_preference_rule_based("애프터눈 딜라이트 향수", 5)
    c = _normalize_classification(c, "애프터눈 딜라이트 향수")
    assert c.primary_category == "물건", f"got {c.primary_category}"
    return c

def _test_empty_text():
    c = classify_preference_rule_based("", 3)
    assert c.primary_category in ("미분류", ""), f"got {c.primary_category}"
    return c

check("사케 → 음료", _test_sake)
check("팩토리오 → 미분류 또는 게임", _test_factorio)
check("GBR flap → 직업, object_tags에 GBR", _test_gbr)
check("애프터눈 딜라이트 향수 → 물건", _test_perfume)
check("빈 텍스트 안전 처리", _test_empty_text)

# ── 2. date helpers ───────────────────────────────────────────────────────────
print(SEP)
print("2. date helpers")

from datetime import date as _date

def infer_time_perspective(event_date):
    if not event_date:
        return "unspecified"
    try:
        ed = _date.fromisoformat(event_date)
        today = _date.today()
        if ed < today: return "past"
        if ed == today: return "present"
        return "future"
    except ValueError:
        return "unspecified"

def format_event_date(event_date, precision):
    if not event_date: return ""
    try:
        if precision == "day":   return event_date[:10]
        if precision == "month": return event_date[:7]
        if precision == "year":  return event_date[:4] + "년"
        return ""
    except Exception:
        return ""

def assert_(expr):
    assert expr

check("infer past",         lambda: assert_(infer_time_perspective("2020-01-01") == "past"))
check("infer future",       lambda: assert_(infer_time_perspective("2099-01-01") == "future"))
check("infer unspecified",  lambda: assert_(infer_time_perspective(None) == "unspecified"))
check("format month",       lambda: assert_(format_event_date("2025-05-01", "month") == "2025-05"))
check("format year",        lambda: assert_(format_event_date("2025-01-01", "year") == "2025년"))
check("format none",        lambda: assert_(format_event_date(None, "none") == ""))

# ── 3. analysis functions with empty entries ──────────────────────────────────
print(SEP)
print("3. analysis with entries=[]")

from tastemap.legacy.analysis import (
    build_preference_profile, generate_profile_sentences,
    build_mindmap_data, get_related_tags, get_searchable_tags,
    get_time_perspective_summary, get_average_rating_by_tag,
    get_top_tags_by_rating,
)

check("build_preference_profile([])",  lambda: build_preference_profile([]))
check("generate_profile_sentences({})", lambda: generate_profile_sentences(
    {"total_entries": 0, "average_rating": 0.0,
     "positive_categories": [], "negative_categories": [],
     "top_positive_objects": [], "top_positive_sensory": [],
     "top_positive_contexts": [], "top_positive_social": [],
     "top_positive_places": [], "top_negative_objects": [],
     "top_negative_sensory": []}
))
check("build_mindmap_data([])",         lambda: build_mindmap_data([]))
check("get_related_tags([], 'test')",   lambda: get_related_tags([], "test"))
check("get_searchable_tags([])",        lambda: get_searchable_tags([]))
check("get_time_perspective_summary([])", lambda: get_time_perspective_summary([]))
check("get_average_rating_by_tag([])",  lambda: get_average_rating_by_tag([], "object_tags"))
check("get_top_tags_by_rating([])",     lambda: get_top_tags_by_rating([]))

# ── 4. analysis with dummy entries ────────────────────────────────────────────
print(SEP)
print("4. analysis with dummy entries")

from tastemap.legacy.models import PreferenceEntry

def _entry(id, cat, rat, obj=None, sen=None, tags=None):
    return PreferenceEntry(
        id=id, raw_text=f"test{id}", rating=rat,
        primary_category=cat, context_categories=[],
        subcategories=[], tags=tags or (obj or []) + (sen or []),
        object_tags=obj or [], sensory_tags=sen or [],
        context_tags=[], social_tags=[], place_tags=[],
        preference_signal="ok", confidence=0.9, created_at="2026-01-01",
    )

dummy = [
    _entry(1, "음식", 5, obj=["내장", "돼지국밥"], sen=["고소함"]),
    _entry(2, "음료", 4, obj=["커피", "드립커피"], sen=["고소함"]),
    _entry(3, "음식", 2, obj=["불닭볶음면"], sen=["매운맛"]),
    _entry(4, "게임", 5, obj=["팩토리오"]),
    _entry(5, "미분류", 3),
]

def _chk_related(r):
    assert r["matched_entries_count"] >= 1
    assert isinstance(r["related_tags"], list)
    return r

def _chk_mindmap(d):
    assert "nodes" in d and "edges" in d
    assert len(d["nodes"]) >= 1
    return d

check("get_related_tags dummy",         lambda: _chk_related(get_related_tags(dummy, "내장")))
check("get_time_perspective_summary",   lambda: get_time_perspective_summary(dummy))
check("build_mindmap_data dummy",       lambda: _chk_mindmap(build_mindmap_data(dummy)))
check("build_preference_profile dummy", lambda: build_preference_profile(dummy))

# ── 5. DB and models ──────────────────────────────────────────────────────────
print(SEP)
print("5. DB migration and list_entries")

from tastemap.legacy.db import init_db, list_entries

def _test_db():
    init_db()
    entries = list_entries()
    for e in entries:
        _ = e.event_date
        _ = e.event_date_precision
        _ = e.time_perspective
    return len(entries)

n = check("init_db + list_entries", _test_db)
if n is not None:
    print(f"       {n}개 entry 로드 OK")

# ── 결과 ──────────────────────────────────────────────────────────────────────
print(SEP)
print(f"PASSED: {passed}  FAILED: {failed}")
if failed == 0:
    print("ALL PASS")
else:
    print("일부 실패 — 위 로그 확인")
    sys.exit(1)
