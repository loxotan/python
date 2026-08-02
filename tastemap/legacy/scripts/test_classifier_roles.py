# -*- coding: utf-8 -*-
"""
Tag role / category assignment test (rule-based)
Run: python scripts/test_classifier_roles.py
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tastemap.legacy.ai_classifier import classify_preference_rule_based, _normalize_classification

TESTS = [
    ("드립커피가 고소해서 좋았다", 5),
    ("애인이 휴일에 꽃 줘서 기뻤다", 5),
    ("곱창전골에 소주 마셨는데 좋았다", 5),
    ("사케 맛있다", 5),
    ("맥도날드에서 드립커피 마셨는데 따뜻했다", 5),
]

PASS_CRITERIA = {
    0: {
        "desc": "드립커피 primary=음료, object에 드립커피/커피, sensory에 고소함",
        "primary": "음료",
        "obj_must":     ["드립커피"],
        "sen_must":     ["고소함"],
        "soc_forbidden": [],
    },
    1: {
        "desc": "꽃은 object, 휴일은 context, 애인은 social, 꽃이 social에 없어야",
        "primary": "관계",
        "obj_must":     ["꽃"],
        "ctx_must":     ["휴일"],
        "soc_must":     ["애인"],
        "soc_forbidden": ["꽃", "휴일"],
        "obj_forbidden": ["휴일"],
    },
    2: {
        "desc": "곱창전골 음식, 소주 object에 포함",
        "primary": "음식",
        "obj_must":     ["곱창전골", "소주"],
    },
    3: {
        "desc": "사케 primary=음료, object에 사케/술",
        "primary": "음료",
        "obj_must":     ["사케"],
    },
    4: {
        "desc": "드립커피 primary=음료 (맥도날드는 rule-based PLACE에 없어 place_tags 빈 것 허용)",
        "primary": "음료",
        "obj_must":     ["드립커피"],
    },
}

SEP = "=" * 65

total_pass = 0
for idx, (text, rating) in enumerate(TESTS):
    raw = classify_preference_rule_based(text, rating)
    c   = _normalize_classification(raw, text)
    crit = PASS_CRITERIA.get(idx, {})

    issues = []
    exp_primary = crit.get("primary", "")
    if exp_primary and c.primary_category != exp_primary:
        issues.append(f"primary: got '{c.primary_category}', expected '{exp_primary}'")

    for must in crit.get("obj_must", []):
        if not any(must in t for t in c.object_tags):
            issues.append(f"object_tags missing '{must}': {c.object_tags}")

    for must in crit.get("sen_must", []):
        if not any(must in t for t in c.sensory_tags):
            issues.append(f"sensory_tags missing '{must}': {c.sensory_tags}")

    for must in crit.get("ctx_must", []):
        if not any(must in t for t in c.context_tags):
            issues.append(f"context_tags missing '{must}': {c.context_tags}")

    for must in crit.get("soc_must", []):
        if not any(must in t for t in c.social_tags):
            issues.append(f"social_tags missing '{must}': {c.social_tags}")

    for forb in crit.get("soc_forbidden", []):
        if any(forb in t for t in c.social_tags):
            issues.append(f"social_tags should NOT contain '{forb}': {c.social_tags}")

    for forb in crit.get("obj_forbidden", []):
        if any(forb in t for t in c.object_tags):
            issues.append(f"object_tags should NOT contain '{forb}': {c.object_tags}")

    result = "PASS" if not issues else "FAIL"
    if not issues:
        total_pass += 1

    print(SEP)
    print(f"[{idx+1}] {result}  {crit.get('desc','')}")
    print(f"     input   : {text}")
    print(f"     primary : {c.primary_category}")
    print(f"     objects : {c.object_tags}")
    print(f"     sensory : {c.sensory_tags}")
    print(f"     context : {c.context_tags}")
    print(f"     social  : {c.social_tags}")
    print(f"     place   : {c.place_tags}")
    if issues:
        for iss in issues:
            print(f"  !! {iss}")

print(SEP)
print(f"TOTAL: {total_pass}/{len(TESTS)}")
