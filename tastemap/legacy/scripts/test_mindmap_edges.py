# -*- coding: utf-8 -*-
"""
Mindmap edge structure test (uses real DB data)
Run: python scripts/test_mindmap_edges.py
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tastemap.legacy.db import init_db, list_entries
from tastemap.legacy.analysis import build_mindmap_data

init_db()
entries = list_entries()
print(f"DB entries: {len(entries)}")

if not entries:
    print("No entries — exiting.")
    sys.exit(0)

def edges_involving(label: str, data: dict) -> list[dict]:
    node_ids = {n["id"] for n in data["nodes"] if n["label"] == label}
    return [e for e in data["edges"] if e["source"] in node_ids or e["target"] in node_ids]

def nodes_by_label(label: str, data: dict) -> list[dict]:
    return [n for n in data["nodes"] if n["label"] == label]

SEP = "=" * 65

for use_hier, label in [(False, "hierarchy=OFF"), (True, "hierarchy=ON")]:
    data = build_mindmap_data(entries, use_tag_hierarchy=use_hier)
    print(SEP)
    print(f"[{label}] nodes={len(data['nodes'])} edges={len(data['edges'])}")

    # object_group 존재 여부
    group_nodes = [n for n in data["nodes"] if n["type"] == "object_group"]
    print(f"  object_group nodes: {[n['label'] for n in group_nodes]}")

    # 커피 노드 목록
    coffee_nodes = [n for n in data["nodes"] if "커피" in n["label"]]
    print(f"  '커피' 관련 nodes: {[(n['label'], n['type']) for n in coffee_nodes]}")

    # 드립커피 연결 엣지
    dc_edges = edges_involving("드립커피", data)
    print(f"  '드립커피' edges: {[(e['source'], e['target']) for e in dc_edges[:5]]}")

    # 고소함 연결 엣지
    go_edges = edges_involving("고소함", data)
    print(f"  '고소함' edges: {[(e['source'], e['target']) for e in go_edges[:5]]}")

    # 꽃 연결 엣지
    flower_edges = edges_involving("꽃", data)
    print(f"  '꽃' edges: {[(e['source'], e['target']) for e in flower_edges[:5]]}")

    # 휴일 연결 엣지
    holiday_edges = edges_involving("휴일", data)
    print(f"  '휴일' edges: {[(e['source'], e['target']) for e in holiday_edges[:5]]}")

    # 소주/사케 연결
    for sake_label in ["소주", "사케", "데킬라"]:
        sake_edges = edges_involving(sake_label, data)
        if sake_edges:
            print(f"  '{sake_label}' edges: {[(e['source'], e['target']) for e in sake_edges[:3]]}")

    # 검증: hierarchy=OFF에서 object_group 없어야 함
    if not use_hier:
        has_group = any(n["type"] == "object_group" for n in data["nodes"])
        status = "FAIL" if has_group else "PASS"
        print(f"  [CHECK] hierarchy=OFF에 object_group 없음: {status}")

    # 검증: 소주/사케가 음식 category에만 붙지 않는지
    drink_labels = {"소주", "사케", "맥주", "데킬라", "사케", "술"}
    drink_under_food = [
        e for e in data["edges"]
        if e["source"] == "category:음식"
        and any(dl in e["target"] for dl in drink_labels)
    ]
    if drink_under_food:
        print(f"  [WARN] 음료가 음식 category 밑에 직접 붙은 엣지: {drink_under_food[:3]}")
    else:
        print("  [CHECK] 음료 직접 category:음식 엣지: 없음 (domain routing 정상)")

print(SEP)
