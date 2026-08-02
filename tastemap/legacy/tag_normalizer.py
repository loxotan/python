CANONICAL_TAG_MAP: dict[str, str] = {
    # 연애 관계
    "여자친구": "애인",
    "여친":     "애인",
    "남자친구": "애인",
    "남친":     "애인",
    "연인":     "애인",
    "파트너":   "애인",
    # 직장 동료
    "직장동료":   "직장 동료",
    "직장 동료":  "직장 동료",
    "동료":       "직장 동료",
    # 부모님
    "엄마":   "어머니",
    "어머니": "어머니",
    "아빠":   "아버지",
    "아버지": "아버지",
}


def normalize_tag(tag: str) -> str:
    cleaned = tag.strip()
    return CANONICAL_TAG_MAP.get(cleaned, cleaned)


def normalize_tags(tags: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        normalized = normalize_tag(str(tag))
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result
