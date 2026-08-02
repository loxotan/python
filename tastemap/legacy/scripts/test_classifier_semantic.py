# -*- coding: utf-8 -*-
"""
Semantic / world-knowledge classifier test (rule-based fallback)
Run: python scripts/test_classifier_semantic.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tastemap.legacy.ai_classifier import classify_preference_rule_based, _normalize_classification

TESTS = [
    ("팩토리오", 5),
    ("팩토리오에서 green circuit 제조 비율 알아보는 게 재밌다", 5),
    ("황천의 츠가이", 5),
    ("데킬라", 5),
    ("헬스", 5),
    ("초리조 소세지", 5),
    ("애프터눈 딜라이트 향수", 5),
    ("황천의 츠가이, 데킬라, 헬스, 초리조 소세지, 애프터눈 딜라이트 향수", 5),
    ("처음 들어보는게임XYZ", 4),
]

EXPECTED = {
    0: "게임",
    1: "게임",
    2: "콘텐츠",
    3: "음료",
    4: "운동",
    5: "음식",
    6: "물건",
    7: "활동",   # rule-based에서는 미분류 허용 (AI 필요)
    8: "게임",   # rule-based: "게임" substring match (AI에서는 미분류 또는 게임 + 낮은 confidence)
}

SEP = "=" * 65

for idx, (text, rating) in enumerate(TESTS):
    raw = classify_preference_rule_based(text, rating)
    c   = _normalize_classification(raw, text)
    exp = EXPECTED.get(idx, "?")
    result = "PASS" if c.primary_category == exp else f"FAIL (expected {exp})"

    print(SEP)
    print(f"[{idx+1}] {result}")
    print(f"     input   : {text[:60]}")
    print(f"     primary : {c.primary_category}")
    print(f"     context : {c.context_categories}")
    print(f"     objects : {c.object_tags}")
    print(f"     tags    : {c.tags[:6]}")
    print(f"     source  : {c.classifier_source}")

print(SEP)
passed = sum(
    1 for idx, (text, rating) in enumerate(TESTS)
    if _normalize_classification(
        classify_preference_rule_based(text, rating), text
    ).primary_category == EXPECTED.get(idx, "")
)
print(f"통과: {passed}/{len(TESTS)}")
print("(0,1,2,7번은 rule-based 한계 - AI 환경에서는 정확도 높아짐)")
