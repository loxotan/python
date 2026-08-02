# -*- coding: utf-8 -*-
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tastemap.legacy.models import PreferenceEntry
from tastemap.legacy.analysis import get_related_tags, get_searchable_tags

def me(id, cat, rat, obj=None, sen=None, ctx=None, soc=None, plc=None, tags=None, sub=None):
    return PreferenceEntry(
        id=id, raw_text=f"entry{id}", rating=rat,
        primary_category=cat, context_categories=[],
        subcategories=sub or [],
        tags=tags or (obj or []) + (sen or []) + (ctx or []) + (soc or []) + (plc or []),
        object_tags=obj or [], sensory_tags=sen or [],
        context_tags=ctx or [], social_tags=soc or [],
        place_tags=plc or [], preference_signal="test",
        confidence=0.9, created_at="2026-01-01",
    )

entries = [
    me(1, "음식", 5, obj=["내장", "돼지국밥"], sen=["고소함", "맛있음"]),
    me(2, "음식", 4, obj=["내장요리", "곱창전골"], sen=["진한맛", "국물"], plc=["노포"]),
    me(3, "음식", 5, obj=["내장", "라면"], sen=["맛있음"]),
    me(4, "음식", 5, obj=["내장", "돼지국밥"], sen=["녹진함", "고소함"]),
    me(5, "음료", 5, obj=["커피", "드립커피"], sen=["고소함", "향긋함"], plc=["카페"]),
    me(6, "음료", 4, obj=["커피", "아메리카노"], sen=["쓴맛"], soc=["혼자"]),
    # entry with duplicate tags across fields (test dedup)
    me(7, "음식", 5,
       obj=["커피"], sen=["고소함"],
       tags=["커피", "고소함"]),  # 커피가 obj와 tags에 동시에 있음
]

SEP = "=" * 60

# 테스트 1: query="내장"
r = get_related_tags(entries, "내장")
print(SEP)
print("테스트 1: query='내장'")
print(f"  matched_entries_count: {r['matched_entries_count']}")
assert r["matched_entries_count"] == 4, f"expected 4, got {r['matched_entries_count']}"
tags_in_result = [x["tag"] for x in r["related_tags"]]
print(f"  related_tags: {tags_in_result[:8]}")
assert "내장" not in tags_in_result, "query 자체가 결과에 있으면 안 됨"
assert "내장요리" not in tags_in_result, "매칭된 태그가 결과에 있으면 안 됨"
assert "돼지국밥" in tags_in_result, f"돼지국밥이 없음: {tags_in_result}"
assert "고소함" in tags_in_result, f"고소함이 없음: {tags_in_result}"
# 빈도 순 정렬 확인
counts = [x["count"] for x in r["related_tags"]]
assert counts == sorted(counts, reverse=True), f"count 정렬 실패: {counts}"
print("테스트 1 PASS")

# 테스트 2: query="커피"
r2 = get_related_tags(entries, "커피")
print(SEP)
print("테스트 2: query='커피'")
print(f"  matched_entries_count: {r2['matched_entries_count']}")
assert r2["matched_entries_count"] == 3, f"expected 3 (entry 5,6,7), got {r2['matched_entries_count']}"
tags2 = [x["tag"] for x in r2["related_tags"]]
print(f"  related_tags: {tags2[:8]}")
assert "커피" not in tags2, "커피가 결과에 있으면 안 됨"
print("테스트 2 PASS")

# 테스트 3: 같은 entry에서 tags/object_tags에 같은 태그 중복 → count 1만 증가
# entry7: obj=["커피"], tags=["커피","고소함"] — query="고소함"
r3 = get_related_tags([entries[6]], "고소함")
print(SEP)
print("테스트 3: query='고소함', entry7 중복 태그 dedup")
커피_item = next((x for x in r3["related_tags"] if x["tag"] == "커피"), None)
assert 커피_item is not None, "커피가 related에 없음"
assert 커피_item["count"] == 1, f"중복으로 count가 증가하면 안 됨: {커피_item['count']}"
print(f"  커피 count={커피_item['count']}  PASS")

# 테스트 4: 없는 태그 검색
r4 = get_related_tags(entries, "XYZABC없는태그")
print(SEP)
print("테스트 4: query='XYZABC없는태그'")
assert r4["matched_entries_count"] == 0
print("테스트 4 PASS")

# 테스트 5: get_searchable_tags
stags = get_searchable_tags(entries, limit=10)
print(SEP)
print("테스트 5: get_searchable_tags")
print(f"  top tags: {stags[:5]}")
assert len(stags) > 0
# 가장 많이 등장한 태그가 먼저
assert stags[0][1] >= stags[-1][1], "count 정렬 실패"
print("테스트 5 PASS")

print(SEP)
print("ALL PASS")
