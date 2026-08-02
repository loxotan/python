# -*- coding: utf-8 -*-
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tastemap.legacy.ai_classifier import classify_preference_rule_based, _normalize_classification

CASES = [
    # (입력, 평점, 기대 primary, 설명)
    ("애프터눈 딜라이트 향수", 5, "물건",
     "단독 향수 → 물건 (활동/미분류 아님)"),
    ("황천의 츠가이, 데킬라, 헬스, 초리조 소세지, 애프터눈 딜라이트 향수", 5, "음료",
     "혼합 5개 목록 → rule-based는 데킬라로 음료 결정 (AI는 활동으로 반환)"),
    ("헬스 오늘 너무 힘들었다", 4, "운동",
     "헬스 단독 → 운동"),
    ("사케 맛있다", 5, "음료",
     "사케 단독 → 음료"),
    ("팩토리오가 재밌다", 5, "미분류",
     "팩토리오 → rule-based 미등록이라 미분류 (AI는 게임으로 반환)"),
]

SEP = "=" * 60
total_pass = 0
for text, rating, expected_primary, desc in CASES:
    raw = classify_preference_rule_based(text, rating)
    c   = _normalize_classification(raw, text)
    result = "PASS" if c.primary_category == expected_primary else "FAIL"
    if result == "PASS":
        total_pass += 1
    print(SEP)
    print(f"[{result}] {desc}")
    print(f"  input  : {text}")
    print(f"  primary: {c.primary_category}  (expected: {expected_primary})")
    print(f"  objects: {c.object_tags}")
    print(f"  tags   : {c.tags[:5]}")

print(SEP)
print(f"TOTAL: {total_pass}/{len(CASES)}")
