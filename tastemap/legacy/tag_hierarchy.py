"""
태그 계층 구조 정의.
정규화(tag_normalizer.py)와는 별개 — 하위 개념을 상위 개념 아래로 묶는 역할.
순환 참조 없이 1단계 parent만 지원한다.
"""

TAG_PARENT_MAP: dict[str, str] = {
    # ── 커피 계열 ─────────────────────────────────────────────────────────────
    "드립커피":        "커피",
    "아이스 아메리카노": "커피",
    "아메리카노":      "커피",
    "카페라떼":        "커피",
    "라떼":            "커피",
    # ── 술/주류 계열 ──────────────────────────────────────────────────────────
    "사케":    "술",
    "데킬라":  "술",
    "소주":    "술",
    "맥주":    "술",
    "하이볼":  "술",
    "와인":    "술",
    "막걸리":  "술",
    # ── 에너지음료 계열 ────────────────────────────────────────────────────────
    "더킹 제로 에너지": "에너지음료",
    "제로 에너지":      "에너지음료",
    "에너지드링크":     "에너지음료",
    "더킹":            "에너지음료",
}


def get_parent_tag(tag: str) -> str | None:
    """태그의 상위 태그를 반환. 없으면 None."""
    cleaned = str(tag).strip()
    parent = TAG_PARENT_MAP.get(cleaned)
    if not parent or parent == cleaned:
        return None
    return parent


def get_tag_path(tag: str) -> list[str]:
    """태그의 경로를 반환. parent가 있으면 [parent, tag], 없으면 [tag]."""
    cleaned = str(tag).strip()
    parent = get_parent_tag(cleaned)
    return [parent, cleaned] if parent else [cleaned]
