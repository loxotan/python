import json
import os

from dotenv import load_dotenv

import re

from models_v2 import AnalysisResult, CategoryV2, InsightResult, TasteEntry
from prompt_templates import (
    ANALYZE_INPUT_SYSTEM_PROMPT,
    ANALYZE_INPUT_USER_TEMPLATE,
    GENERATE_INSIGHT_SYSTEM_PROMPT,
    GENERATE_INSIGHT_USER_TEMPLATE,
    PATTERN_INSIGHT_SYSTEM_PROMPT,
    PATTERN_INSIGHT_USER_TEMPLATE,
)

load_dotenv()

_REQUIRED_ANALYSIS_FIELDS = {
    "input_type", "category", "needs_more_context"
}


def analyze_input(raw_input: str, rating: int) -> AnalysisResult:
    """입력과 별점을 분석해서 AnalysisResult를 반환한다. 실패 시 fallback."""
    if not os.environ.get("OPENAI_API_KEY"):
        return _fallback_analysis("API 키가 없습니다. 태그를 직접 추가해주세요.")

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": ANALYZE_INPUT_SYSTEM_PROMPT},
                {"role": "user", "content": ANALYZE_INPUT_USER_TEMPLATE.format(
                    raw_input=raw_input, rating=rating
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=700,
        )
        data = json.loads(response.choices[0].message.content)
        return _parse_analysis(data)
    except Exception as exc:
        return _fallback_analysis(f"분석 중 오류가 발생했습니다: {exc}")


def generate_insight(
    raw_input: str,
    rating: int,
    input_type: str,
    category: CategoryV2,
    contexts: list[str],
    selected_tags: list[str],
    selected_tag_types: dict[str, str] | None = None,
    source_tags: list[str] | None = None,
    candidate_tags: list[str] | None = None,
) -> InsightResult:
    """선택된 태그를 바탕으로 InsightResult를 생성한다. 품질 검사 후 1회 재시도. 실패 시 fallback."""
    if not os.environ.get("OPENAI_API_KEY"):
        return _fallback_insight(rating)

    types = selected_tag_types or {}
    src  = source_tags or []
    cand = candidate_tags or []

    feature_selected   = [t for t in selected_tags if types.get(t) == "feature"]
    preference_selected = [t for t in selected_tags if types.get(t) == "preference"]
    candidate_selected  = [t for t in selected_tags if types.get(t) == "candidate"]
    if not feature_selected and not preference_selected and not candidate_selected:
        feature_selected = selected_tags

    cat_parts = [category.large, category.medium, category.small]
    category_str      = " > ".join(p for p in cat_parts if p) or "미분류"
    source_str        = ", ".join(src) if src else "없음"
    feature_str       = ", ".join(feature_selected) if feature_selected else "없음"
    preference_str    = ", ".join(preference_selected) if preference_selected else "없음"
    candidate_str     = ", ".join(cand) if cand else "없음"
    cand_selected_str = ", ".join(candidate_selected) if candidate_selected else "없음"
    all_tags_str      = ", ".join(selected_tags) if selected_tags else "없음"

    user_msg = GENERATE_INSIGHT_USER_TEMPLATE.format(
        raw_input=raw_input, rating=rating, category=category_str,
        source_tags=source_str, feature_tags=feature_str,
        preference_tags=preference_str, candidate_tags=candidate_str,
        candidate_selected=cand_selected_str, selected_tags=all_tags_str,
    )

    try:
        from openai import OpenAI
        client = OpenAI()

        def _call(messages) -> InsightResult:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=500,
            )
            d = json.loads(resp.choices[0].message.content)
            depth = d.get("insight_depth", "medium")
            if depth not in ("low", "medium", "high"):
                depth = "medium"
            mode = d.get("insight_mode", "contrast")
            if mode not in ("observation", "contrast", "routine", "relationship", "sensory", "avoidance"):
                mode = "contrast"
            alevel = d.get("abstraction_level", "moderate")
            if alevel not in ("literal", "moderate", "high"):
                alevel = "moderate"
            one_line = str(d.get("one_line_insight", ""))
            return InsightResult(
                key_context=[str(v) for v in d.get("key_context", [])],
                preference_rule=str(d["preference_rule"]) if d.get("preference_rule") else "",
                preference_contrast=str(d["preference_contrast"]) if d.get("preference_contrast") else "",
                insight_depth=depth,
                insight_mode=mode,
                abstraction_level=alevel,
                one_line_insight=one_line or _fallback_insight(rating).one_line_insight,
            )

        msgs = [
            {"role": "system", "content": GENERATE_INSIGHT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        result = _call(msgs)

        # 품질 검사
        ok, reason = _quality_check(raw_input, result.one_line_insight, selected_tags, rating)
        if not ok:
            print(f"[INSIGHT QUALITY FAIL] {reason} → 재시도")
            correction = (
                f"\n\n이전 해석: {result.one_line_insight}\n"
                f"실패 이유: {reason}\n"
                "위 문제를 수정해서 JSON을 다시 작성하라. "
                "특히 one_line_insight만 고치고, 나머지 필드는 이전과 동일하게 유지하라."
            )
            msgs.append({"role": "assistant", "content": json.dumps(
                {"one_line_insight": result.one_line_insight,
                 "key_context": result.key_context,
                 "preference_rule": result.preference_rule,
                 "preference_contrast": result.preference_contrast,
                 "insight_depth": result.insight_depth,
                 "insight_mode": result.insight_mode,
                 "abstraction_level": result.abstraction_level},
                ensure_ascii=False
            )})
            msgs.append({"role": "user", "content": correction})
            try:
                result = _call(msgs)
            except Exception:
                pass  # 재시도 실패 시 원래 결과 유지

        return result
    except Exception:
        return _fallback_insight(rating)


def _parse_analysis(data: dict) -> AnalysisResult:
    missing = _REQUIRED_ANALYSIS_FIELDS - data.keys()
    if missing:
        raise ValueError(f"응답 필드 누락: {missing}")

    raw_cat = data.get("category", {})
    category = CategoryV2(
        large=str(raw_cat.get("large", "")) if isinstance(raw_cat, dict) else "",
        medium=str(raw_cat.get("medium", "")) if isinstance(raw_cat, dict) else "",
        small=str(raw_cat.get("small", "")) if isinstance(raw_cat, dict) else "",
    )

    source_tags     = [str(v) for v in data.get("source_tags", [])]
    feature_tags    = [str(v) for v in data.get("feature_tags", [])]
    preference_tags = [str(v) for v in data.get("preference_tags", [])]
    candidate_tags  = [str(v) for v in data.get("candidate_tags", [])]
    recommended     = [str(v) for v in data.get("recommended_tags", [])]

    # recommended_tags가 비어 있으면 feature + preference로 구성
    if not recommended:
        recommended = list(dict.fromkeys(feature_tags + preference_tags))

    confidence = data.get("knowledge_confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    return AnalysisResult(
        input_type=str(data.get("input_type", "word")),
        is_named_entity=bool(data.get("is_named_entity", False)),
        entity_lookup_needed=bool(data.get("entity_lookup_needed", False)),
        category=category,
        contexts=[str(v) for v in data.get("contexts", [])],
        source_tags=source_tags,
        feature_tags=feature_tags,
        preference_tags=preference_tags,
        candidate_tags=candidate_tags,
        recommended_tags=recommended,
        needs_more_context=bool(data.get("needs_more_context", False)),
        context_question=data.get("context_question") or None,
        knowledge_confidence=confidence,
    )


def _fallback_analysis(reason: str) -> AnalysisResult:
    return AnalysisResult(
        input_type="word",
        category=CategoryV2(),
        contexts=[],
        source_tags=[],
        feature_tags=[],
        preference_tags=[],
        recommended_tags=[],
        needs_more_context=True,
        context_question=reason,
        knowledge_confidence="low",
    )


def _fallback_insight(rating: int) -> InsightResult:
    if rating >= 4:
        msg = "이 경험에서 긍정적인 요소를 발견한 편일 수 있습니다."
    elif rating <= 2:
        msg = "이 경험에서 불편하거나 맞지 않는 요소를 느꼈을 수 있습니다."
    else:
        msg = "이 경험은 특별히 강한 인상을 남기지 않았을 수 있습니다."
    return InsightResult(one_line_insight=msg)


# ── 품질 검사 ─────────────────────────────────────────────────────────────────

def _quality_check(
    raw_input: str, insight: str, selected_tags: list[str], rating: int
) -> tuple[bool, str]:
    """Returns (ok, failure_reason). Heuristic checks only."""
    issues = []

    # 1. insight longer than raw_input (for inputs > 20 chars)
    if len(raw_input) > 20 and len(insight) > len(raw_input):
        issues.append(f"해석({len(insight)}자)이 원문({len(raw_input)}자)보다 깁니다")

    # 2. Raw input noun repetition (for inputs > 20 chars)
    if len(raw_input) > 20:
        tokens = [t for t in re.split(r"[\s,\.!?]+", raw_input) if len(t) >= 2]
        repeated = [t for t in tokens if t in insight]
        if len(repeated) >= 3:
            issues.append(f"원문 단어 반복: {', '.join(repeated[:3])}")

    # 3. Too many selected tags in insight
    tags_in = [t for t in selected_tags if t in insight]
    if len(tags_in) >= 3:
        issues.append(f"태그 {len(tags_in)}개 반복: {', '.join(tags_in[:3])}")

    # 4. Forbidden patterns
    for fp in ["를 통해 만족감", "을 통해 만족감", "이 긍정적인 경험으로", "가 조화를 이루어 만족감"]:
        if fp in insight:
            issues.append(f"금지 문형 포함: '{fp}'")
            break

    # 5. Rating 3 + strong preference expression
    if rating == 3:
        for word in ["선호하는 편", "좋아하는 편", "즐기는 편일 수 있습니다", "긍정적으로 평가"]:
            if word in insight:
                issues.append(f"3점인데 강한 선호 표현: '{word}'")
                break

    return len(issues) == 0, " / ".join(issues)


# ── 패턴 분석 ─────────────────────────────────────────────────────────────────

def _rating_direction(rating: int) -> str:
    if rating >= 4: return "positive"
    if rating <= 2: return "negative"
    return "neutral"


def find_related_entries(entry: TasteEntry, all_entries: list[TasteEntry]) -> list[TasteEntry]:
    """같은 카테고리 + 별점 방향 + 태그 유사성이 있는 기존 기록들을 반환한다."""
    current_dir = _rating_direction(entry.rating)
    current_tags = set(entry.selected_tags)

    related: list[TasteEntry] = []
    for e in all_entries:
        if e.id == entry.id:
            continue
        # Same category (large or medium)
        same_cat = bool(
            (entry.category.large and e.category.large == entry.category.large)
            or (entry.category.medium and e.category.medium == entry.category.medium)
        )
        if not same_cat:
            continue
        # Same rating direction
        if _rating_direction(e.rating) != current_dir:
            continue
        # Tag overlap >= 2
        e_tags = set(e.selected_tags)
        if len(current_tags & e_tags) >= 2:
            related.append(e)

    return related


def pattern_confidence(related: list[TasteEntry]) -> str:
    n = len(related)
    if n == 0: return "none"
    if n == 1: return "low"
    if n == 2: return "medium"
    return "high"


def generate_pattern_insight(
    entry: TasteEntry,
    related: list[TasteEntry],
) -> dict:
    """누적 패턴 해석을 생성한다. {"pattern_insight": str, "pattern_confidence": str}"""
    if not os.environ.get("OPENAI_API_KEY") or not related:
        return {"pattern_insight": "", "pattern_confidence": "none"}

    current_tags_str = ", ".join(entry.selected_tags[:6]) if entry.selected_tags else "없음"

    related_lines = []
    for e in related[:4]:
        tags = ", ".join(e.selected_tags[:4]) if e.selected_tags else "없음"
        related_lines.append(f"- {e.raw_input} (★{e.rating}): {tags}")
    related_summary = "\n".join(related_lines)

    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": PATTERN_INSIGHT_SYSTEM_PROMPT},
                {"role": "user", "content": PATTERN_INSIGHT_USER_TEMPLATE.format(
                    current_input=entry.raw_input,
                    current_rating=entry.rating,
                    current_tags=current_tags_str,
                    related_summary=related_summary,
                )},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=250,
        )
        data = json.loads(response.choices[0].message.content)
        conf = data.get("pattern_confidence", "low")
        if conf not in ("low", "medium", "high"):
            conf = "low"
        return {
            "pattern_insight": str(data.get("pattern_insight", "")),
            "pattern_confidence": conf,
        }
    except Exception:
        return {"pattern_insight": "", "pattern_confidence": "none"}
