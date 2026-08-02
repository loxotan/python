
# -*- coding: utf-8 -*-
"""
rule-based classifier context / category test
Run: python scripts/test_classifier_context.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tastemap.legacy.ai_classifier import classify_preference_rule_based

TESTS = [
    ("맥도날드에서 드립커피 마셨는데 따뜻해서 좋았다", 5),
    ("여자친구랑 맥도날드에서 드립커피 마셨는데 따뜻해서 좋았다", 5),
    ("첫 매복 사랑니 발치... 힘들었음 하지만 다음엔 더 잘할 수 있을 거 같아", 4),
    ("치주 수술 중에는 GTR이 제일 뿌듯함", 5),
    ("운동하니까 시간이 잘 간다", 4),
    ("불닭볶음면은 너무 맵다", 2),
    ("교수님 피드백 받고 논문 discussion 방향이 잡혔다", 5),
    ("애인이랑 노포에서 곱창전골 먹었다", 5),
]

EXPECTED = {
    0: {"primary": "음료",  "notes": "드립커피 → 음료, place_tags에 맥도날드 없음(rule-based 한계)"},
    1: {"primary": "음료",  "notes": "여자친구 → 애인 정규화"},
    2: {"primary": "직업",  "notes": "매복 사랑니 발치 → 직업, object_tags에 발치 포함"},
    3: {"primary": "직업",  "notes": "GTR → 직업, sensory에 뿌듯함"},
    4: {"primary": "운동",  "notes": "운동 → 운동"},
    5: {"primary": "음식",  "notes": "불닭볶음면 → 음식 (keyword 미등록이면 미분류 가능)"},
    6: {"primary": "직업",  "notes": "논문+피드백 → 직업/학습"},
    7: {"primary": "음식",  "notes": "곱창전골 → 음식, social=애인, place=노포"},
}

SEP = "=" * 65

for idx, (text, rating) in enumerate(TESTS):
    c = classify_preference_rule_based(text, rating)
    exp = EXPECTED.get(idx, {})
    exp_primary = exp.get("primary", "?")
    match = "PASS" if c.primary_category == exp_primary else "FAIL"

    print(SEP)
    print(f"[{idx+1}] {match} input : {text[:55]}")
    print(f"     rating: {rating}  |  expected primary: {exp_primary}")
    print(f"     primary_category  : {c.primary_category}")
    print(f"     context_categories: {c.context_categories}")
    print(f"     object_tags       : {c.object_tags}")
    print(f"     sensory_tags      : {c.sensory_tags}")
    print(f"     context_tags      : {c.context_tags}")
    print(f"     social_tags       : {c.social_tags}")
    print(f"     place_tags        : {c.place_tags}")
    print(f"     note: {exp.get('notes', '')}")

print(SEP)
passed = sum(
    1 for idx, (text, rating) in enumerate(TESTS)
    if classify_preference_rule_based(text, rating).primary_category
       == EXPECTED.get(idx, {}).get("primary", "")
)
print(f"통과: {passed}/{len(TESTS)}")
