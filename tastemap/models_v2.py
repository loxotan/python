from typing import Optional
from pydantic import BaseModel


class CategoryV2(BaseModel):
    large: str = ""
    medium: str = ""
    small: str = ""


class InsightResult(BaseModel):
    key_context: list[str] = []
    preference_rule: str = ""
    preference_contrast: str = ""
    insight_depth: str = "medium"       # "low" | "medium" | "high"
    insight_mode: str = "contrast"      # "observation"|"contrast"|"routine"|"relationship"|"sensory"|"avoidance"
    abstraction_level: str = "moderate" # "literal" | "moderate" | "high"
    one_line_insight: str = ""


class AnalysisResult(BaseModel):
    input_type: str                   # "sentence" | "word"
    is_named_entity: bool = False
    entity_lookup_needed: bool = False
    category: CategoryV2
    contexts: list[str]
    source_tags: list[str] = []       # 정보성 태그 (대상 분류, 선택 불필요)
    feature_tags: list[str] = []      # 구체 속성 태그 (선택 대상)
    preference_tags: list[str] = []   # 느낌/해석 태그 (선택 대상)
    candidate_tags: list[str] = []    # 예비 태그 (사용자가 추가 선택 가능)
    recommended_tags: list[str] = []  # feature + preference 혼합 (UI 선택용)
    needs_more_context: bool
    context_question: Optional[str] = None
    knowledge_confidence: str = "medium"  # "high" | "medium" | "low"


class TasteEntry(BaseModel):
    id: Optional[int] = None
    raw_input: str
    rating: int
    input_type: str
    category: CategoryV2
    contexts: list[str]
    source_tags: list[str] = []
    feature_tags: list[str] = []
    preference_tags: list[str] = []
    candidate_tags: list[str] = []
    recommended_tags: list[str] = []
    selected_tags: list[str]
    selected_tag_types: dict[str, str] = {}   # label → "feature"|"preference"|"source"|"custom"
    one_line_insight: str = ""
    key_context: list[str] = []               # 원문에서 추출한 핵심 맥락
    preference_rule: str = ""                 # 내부 추론 기준 문장
    preference_contrast: str = ""             # "A보다 B" 대비 구조
    abstraction_level: str = "moderate"       # "literal" | "moderate" | "high"
    insight_depth: str = "medium"             # "low" | "medium" | "high"
    insight_mode: str = "contrast"            # "observation"|"contrast"|"routine"|"relationship"|"sensory"|"avoidance"
    pattern_insight: str = ""                 # 누적 패턴 해석
    pattern_confidence: str = "none"          # "none" | "low" | "medium" | "high"
    related_entry_ids: list[int] = []
    needs_more_context: bool = False
    context_question: Optional[str] = None
    knowledge_confidence: str = "medium"
    created_at: str = ""
    updated_at: str = ""
