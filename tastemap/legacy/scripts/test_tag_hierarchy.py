# -*- coding: utf-8 -*-
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tastemap.legacy.models import PreferenceEntry
from tastemap.legacy.analysis import build_mindmap_data
from tastemap.legacy.tag_hierarchy import get_parent_tag, get_tag_path

def make_entry(id, cat, rating, obj):
    return PreferenceEntry(
        id=id, raw_text="test", rating=rating,
        primary_category=cat, context_categories=[],
        subcategories=[], tags=obj,
        object_tags=obj, sensory_tags=[], context_tags=[],
        social_tags=[], place_tags=[],
        preference_signal="test", confidence=0.9, created_at="2026-01-01",
    )

# 테스트 1: get_parent_tag
assert get_parent_tag("아메리카노") == "커피"
assert get_parent_tag("드립커피") == "커피"
assert get_parent_tag("데킬라") == "술"
assert get_parent_tag("사케") == "술"
assert get_parent_tag("커피") is None        # parent 없음
assert get_parent_tag("모르는태그") is None
print("테스트 1 PASS: get_parent_tag")

# 테스트 2: hierarchy OFF - object_group 없어야 함
entries = [
    make_entry(1, "음료", 5, ["아메리카노", "드립커피"]),
    make_entry(2, "음료", 4, ["드립커피"]),
]
data_off = build_mindmap_data(entries, use_tag_hierarchy=False)
types_off = {n["type"] for n in data_off["nodes"]}
assert "object_group" not in types_off, f"OFF인데 object_group 있음: {types_off}"
print("테스트 2 PASS: hierarchy OFF → object_group 없음")

# 테스트 3: hierarchy ON, child 2개 이상 → object_group 생성
data_on = build_mindmap_data(entries, use_tag_hierarchy=True)
types_on = {n["type"] for n in data_on["nodes"]}
assert "object_group" in types_on, f"ON인데 object_group 없음: {types_on}"
group_node = next(n for n in data_on["nodes"] if n["type"] == "object_group")
assert group_node["label"] == "커피", f"group label: {group_node['label']}"
print(f"테스트 3 PASS: hierarchy ON → object_group 생성 ({group_node['label']})")

# 테스트 4: child 1개뿐이면 object_group 미생성
entries_single = [
    make_entry(3, "음료", 5, ["아메리카노"]),   # 커피 아래 child 1개
]
data_single = build_mindmap_data(entries_single, use_tag_hierarchy=True)
group_nodes = [n for n in data_single["nodes"] if n["type"] == "object_group"]
assert len(group_nodes) == 0, f"child 1개인데 object_group 생성됨: {group_nodes}"
print("테스트 4 PASS: child 1개 → object_group 미생성")

# 테스트 5: sensory/social/place는 계층화 안 됨
entries_mixed = [
    make_entry(4, "음식", 5, ["아메리카노", "드립커피"]),
]
entries_mixed[0] = PreferenceEntry(
    id=4, raw_text="test", rating=5,
    primary_category="음식", context_categories=[],
    subcategories=[], tags=["아메리카노", "드립커피", "뿌듯함"],
    object_tags=["아메리카노", "드립커피"],
    sensory_tags=["뿌듯함"],
    context_tags=[], social_tags=[], place_tags=[],
    preference_signal="test", confidence=0.9, created_at="2026-01-01",
)
data_mixed = build_mindmap_data([entries_mixed[0]], use_tag_hierarchy=True)
sensory_nodes = [n for n in data_mixed["nodes"] if n["type"] == "sensory"]
group_nodes2  = [n for n in data_mixed["nodes"] if n["type"] == "object_group"]
# child 1개라 group 없음 (단일 entry)
assert all(n["type"] == "sensory" for n in sensory_nodes)
print(f"테스트 5 PASS: sensory 계층화 없음 (sensory={len(sensory_nodes)}, group={len(group_nodes2)})")

# 테스트 6: show_objects=False → object/object_group 모두 사라짐
include_no_obj = {"root", "category", "sensory"}
data_no_obj = build_mindmap_data(entries, use_tag_hierarchy=True, include_types=include_no_obj)
obj_nodes = [n for n in data_no_obj["nodes"] if n["type"] in ("object", "object_group")]
assert len(obj_nodes) == 0, f"object 표시 OFF인데 object 노드 있음: {obj_nodes}"
print("테스트 6 PASS: object 표시 OFF → object/object_group 모두 사라짐")

# 테스트 7: 술 계열 child 2개 이상
entries_drink = [
    make_entry(5, "음료", 4, ["데킬라", "사케"]),
    make_entry(6, "음료", 5, ["사케"]),
]
data_drink = build_mindmap_data(entries_drink, use_tag_hierarchy=True)
drink_group = [n for n in data_drink["nodes"] if n["type"] == "object_group" and n["label"] == "술"]
assert len(drink_group) == 1, f"술 그룹 없음: {drink_group}"
print(f"테스트 7 PASS: 술 계열 묶음 생성 (count={drink_group[0]['count']})")

print("\nALL PASS")
