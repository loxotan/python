import json
import os

from dotenv import load_dotenv

from tastemap.legacy.models import PreferenceClassification

load_dotenv()

# ── Object keywords ───────────────────────────────────────────────────────────
# Longer compound words must be checked before their sub-components.
# _match_objects() sorts by length descending and masks matched spans.

OBJECT_KEYWORD_MAP: dict[str, list[str]] = {
    # ── 음식 compounds (4+ chars) ──────────────────────────────────────────────
    "곱창전골":   ["곱창전골", "곱창", "전골", "내장"],
    "김치찌개":   ["김치찌개", "찌개"],
    "된장찌개":   ["된장찌개", "찌개"],
    "아이스크림": ["아이스크림", "디저트"],
    "아메리카노": ["아메리카노", "커피"],
    "카페라떼":   ["카페라떼", "커피"],
    # ── 음식 single ────────────────────────────────────────────────────────────
    "곱창":   ["곱창", "내장"],
    "전골":   ["전골"],
    "국밥":   ["국밥"],
    "라멘":   ["라멘"],
    "초밥":   ["초밥"],
    "삼겹살": ["삼겹살"],
    "고기":   ["고기"],
    "커피":   ["커피"],
    "소주":   ["소주"],
    "맥주":   ["맥주"],
    "술":     ["술"],
    "치킨":   ["치킨"],
    "피자":   ["피자"],
    "파스타": ["파스타"],
    "라면":   ["라면"],
    "케이크":    ["케이크", "디저트"],
    "불닭볶음면": ["불닭볶음면", "라면"],
    "떡볶이":    ["떡볶이"],
    "순대":      ["순대"],
    "김밥":      ["김밥"],
    # 기타 물건
    "꽃":  ["꽃"],
    # ── 게임 ───────────────────────────────────────────────────────────────────
    "젤다":   ["젤다"],
    "포켓몬": ["포켓몬"],
    "야숨":   ["야숨", "젤다"],
    "왕눈":   ["왕눈", "젤다"],
    "게임":   ["게임"],
    "스위치": ["닌텐도 스위치"],
    # ── 치과/임상 compounds — 가장 긴 복합어부터 ──────────────────────────────
    "매복 사랑니 발치": ["매복 사랑니 발치", "매복 사랑니", "사랑니 발치", "사랑니", "발치"],
    "flap reflection": ["flap reflection", "flap", "판막"],
    "매복 사랑니": ["매복 사랑니", "사랑니", "발치"],
    "사랑니 발치": ["사랑니 발치", "사랑니", "발치"],
    "치주수술":   ["치주수술", "수술"],
    "치근활택":   ["치근활택", "SRP"],
    "임플란트":   ["임플란트"],
    "스케일링":   ["스케일링"],
    # ── 치과/임상 single ────────────────────────────────────────────────────────
    "사랑니": ["사랑니", "발치"],
    "발치":   ["발치"],
    "GBR":    ["GBR", "골이식", "치주수술"],
    "GTR":    ["GTR", "치주재생", "치주수술"],
    "flap":   ["flap", "판막"],
    "SRP":    ["SRP", "치근활택"],
    "판막":   ["판막"],
    "봉합":   ["봉합"],
    # ── 학술/연구 ───────────────────────────────────────────────────────────────
    "논문":   ["논문"],
    "발표":   ["발표"],
    "세미나": ["세미나"],
    # ── 음료 ────────────────────────────────────────────────────────────────────
    "드립커피":  ["드립커피", "커피"],
    "탄산음료":  ["탄산음료"],
    "에너지음료": ["에너지음료"],
    "사케":     ["사케", "술"],
    "위스키":   ["위스키", "술"],
    "와인":     ["와인", "술"],
    "하이볼":   ["하이볼", "술"],
    "막걸리":   ["막걸리", "술"],
    "데킬라":   ["데킬라", "술"],
    # ── 운동 ────────────────────────────────────────────────────────────────────
    "러닝":  ["러닝", "달리기"],
    "헬스":  ["헬스", "웨이트"],
    "수영":  ["수영"],
    "등산":  ["등산"],
    "운동":  ["운동"],
    # ── 콘텐츠 ─────────────────────────────────────────────────────────────────
    "영화":    ["영화"],
    "드라마":  ["드라마"],
    "소설":    ["소설", "책"],
    "웹툰":    ["웹툰"],
    "음악":    ["음악"],
    "책":      ["책"],
    "오디오북": ["오디오북", "책"],
}

SUBCATEGORY_MAP: dict[str, list[str]] = {
    # 음식
    "곱창전골": ["한식", "전골", "내장요리"],
    "곱창":     ["한식", "내장요리"],
    "전골":     ["한식", "전골"],
    "국밥":     ["한식"],
    "라멘":     ["일식", "라멘"],
    "초밥":     ["일식", "초밥"],
    "삼겹살":   ["한식", "구이"],
    "고기":     ["육류"],
    "커피":     ["음료"],
    "아메리카노": ["음료", "커피"],
    "카페라떼": ["음료", "커피"],
    "소주":     ["주류", "한국술"],
    "맥주":     ["주류"],
    "술":       ["주류"],
    "치킨":     ["치킨"],
    "피자":     ["양식"],
    "파스타":   ["양식"],
    "라면":     ["한식", "면류"],
    "김치찌개": ["한식", "찌개"],
    "된장찌개": ["한식", "찌개"],
    "케이크":   ["디저트", "베이커리"],
    "아이스크림": ["디저트"],
    # 게임
    "젤다":   ["어드벤처", "닌텐도"],
    "포켓몬": ["RPG", "닌텐도"],
    "야숨":   ["어드벤처", "닌텐도"],
    "왕눈":   ["어드벤처", "닌텐도"],
    "게임":   ["게임 일반"],
    "스위치": ["닌텐도"],
    # 치과/임상
    "매복 사랑니 발치": ["치과", "구강외과", "발치"],
    "flap reflection":  ["치과", "치주과", "수술", "판막"],
    "매복 사랑니": ["치과", "구강외과", "발치"],
    "사랑니 발치": ["치과", "구강외과", "발치"],
    "사랑니":     ["치과", "구강외과"],
    "발치":       ["치과", "구강외과"],
    "임플란트":   ["치과", "임플란트"],
    "GBR":        ["치과", "치주과", "골재생", "수술"],
    "GTR":        ["치과", "치주과", "치주재생", "수술"],
    "flap":       ["치과", "치주과", "수술", "판막"],
    "SRP":        ["치과", "치주과", "비외과치료"],
    "치근활택":   ["치과", "치주과", "비외과치료"],
    "판막":       ["치과", "수술"],
    "봉합":       ["치과", "수술"],
    "치주수술":   ["치과", "치주과", "수술"],
    "스케일링":   ["치과", "예방치료"],
    # 학술/연구
    "논문":   ["연구", "논문"],
    "발표":   ["학술", "발표"],
    "세미나": ["학술", "세미나"],
    # 음료
    "드립커피":  ["음료", "커피"],
    "탄산음료":  ["음료", "탄산"],
    "에너지음료": ["음료", "에너지음료"],
    "사케":     ["음료", "주류"],
    "위스키":   ["음료", "주류"],
    "와인":     ["음료", "주류"],
    "하이볼":   ["음료", "주류"],
    "막걸리":   ["음료", "주류"],
    "데킬라":   ["음료", "주류"],
    # 물건
    "꽃":       ["물건", "선물"],
    # 운동
    "러닝":  ["운동", "유산소"],
    "헬스":  ["운동", "웨이트"],
    "수영":  ["운동", "유산소"],
    "등산":  ["운동", "야외활동"],
    "운동":  ["운동"],
    # 콘텐츠
    "영화":    ["콘텐츠", "영상"],
    "드라마":  ["콘텐츠", "영상"],
    "소설":    ["콘텐츠", "독서"],
    "웹툰":    ["콘텐츠", "웹툰"],
    "음악":    ["콘텐츠", "음악"],
    "책":      ["콘텐츠", "독서"],
    "오디오북": ["콘텐츠", "독서"],
}

FOOD_SENSORY_INFERENCE: dict[str, list[str]] = {
    "곱창전골": ["진한맛", "뜨끈함", "국물"],
    "곱창":     ["진한맛"],
    "전골":     ["뜨끈함", "국물"],
    "국밥":     ["진한맛", "뜨끈함", "국물"],
    "라멘":     ["진한맛", "국물"],
    "라면":     ["국물"],
    "김치찌개": ["얼큰함", "국물"],
    "된장찌개": ["구수함", "국물"],
    "삼겹살":   ["고소함"],
    "커피":     ["쓴맛", "향긋함"],
    "아메리카노": ["쓴맛"],
    "소주":     ["쓴맛"],
    "케이크":   ["달콤함"],
    "아이스크림": ["달콤함", "시원함"],
}

SENSORY_DIRECT_MAP: dict[str, str] = {
    # 음식/감각
    "맛있":  "맛있음",
    "맛없":  "맛없음",
    "뜨끈":  "뜨끈함",
    "뜨거":  "뜨거움",
    "시원":  "시원함",
    "달콤":  "달콤함",
    "달달":  "달달함",
    "매콤":  "매콤함",
    "얼큰":  "얼큰함",
    "구수":  "구수함",
    "국물":  "국물",
    "바삭":  "바삭함",
    "고소":  "고소함",
    "느끼":  "느끼함",
    "담백":  "담백함",
    "촉촉":  "촉촉함",
    "진한":  "진한맛",
    "향긋":  "향긋함",
    "쫄깃":  "쫄깃함",
    "부드러": "부드러움",
    # 직업/임상 경험 신호 — 복합어 먼저 (substring 충돌 방지)
    "더 잘할": "성장감",
    "잘할 수": "성장감",
    "더 잘":   "성장감",
    "잘 안":   "어려움",
    "잘 됐":   "잘됨",
    "잘됐":    "잘됨",
    "잘했":    "잘됨",
    "감 잡":   "숙련감",
    "늘었":    "성장감",
    "배웠":    "학습감",
    "힘들":    "어려움",
    "어렵":    "어려움",
    "엉망":    "어려움",
    "잡혔":    "명확해짐",
    "뿌듯":    "뿌듯함",
    "긴장":    "긴장감",
    "무서":    "부담감",
}

CONTEXT_KEYWORD_MAP: dict[str, str] = {
    # 일상
    "아침":   "아침",
    "점심":   "점심",
    "저녁":   "저녁",
    "야식":   "야식",
    "술자리": "술자리",
    "한잔":   "술자리",
    "혼술":   "혼술",
    "혼밥":   "혼밥",
    "회식":   "회식",
    "데이트": "데이트",
    "여행":   "여행",
    "주말":   "주말",
    "평일":   "평일",
    # 직업/임상
    "임상":   "임상",
    "수술":   "수술",
    "술식":   "술식",
    "진료":   "진료",
    "전공의": "전공의",
    "의국":   "의국",
    "공부":   "학습",
    "피드백": "피드백",
    "발표":   "발표",
    "연구":   "연구",
    "세미나": "세미나",
    "처음":   "첫 경험",
    "첫":     "첫 경험",
    # 시간/상황
    "휴일":  "휴일",
}

SOCIAL_KEYWORD_MAP: dict[str, str] = {
    # 연애 관계 — 모두 "애인"으로 통일
    "여자친구": "애인",
    "여친":     "애인",
    "남자친구": "애인",
    "남친":     "애인",
    "연인":     "애인",
    "파트너":   "애인",
    "애인":     "애인",
    # 가족
    "친구":   "친구",
    "부모님": "부모님",
    "엄마":   "어머니",
    "어머니": "어머니",
    "아빠":   "아버지",
    "아버지": "아버지",
    "가족":   "가족",
    "동생":   "동생",
    # 기타
    "교수님": "교수님",
    "혼자":   "혼자",
    # 직업
    "환자":      "환자",
    "스탭":      "스탭",
    "동기":      "동기",
    "선배":      "선배",
    "후배":      "후배",
    "직장동료":  "직장 동료",
    "직장 동료": "직장 동료",
}

PLACE_KEYWORD_MAP: dict[str, str] = {
    "노포":    "노포",
    "바다":    "바다",
    "카페":    "카페",
    "학교":    "학교",
    "식당":    "식당",
    "레스토랑": "레스토랑",
    "술집":    "술집",
    "편의점":  "편의점",
    "공원":    "공원",
    "영화관":  "영화관",
    # 직업/의료
    "치과":    "치과",
    "병원":    "병원",
    "수술실":  "수술실",
    "의국":    "의국",
    "세미나실": "세미나실",
}

# ── Object key sets by domain ─────────────────────────────────────────────────
FOOD_OBJECT_KEYS = {
    "곱창전골", "곱창", "전골", "국밥", "라멘", "초밥", "삼겹살", "고기",
    "치킨", "피자", "파스타", "라면", "케이크", "아이스크림",
    "김치찌개", "된장찌개", "불닭볶음면", "떡볶이", "순대", "김밥",
}
DRINK_OBJECT_KEYS = {
    "커피", "아메리카노", "카페라떼", "드립커피",
    "소주", "맥주", "술", "탄산음료", "에너지음료",
    "사케", "위스키", "와인", "하이볼", "막걸리", "데킬라",
}
GAME_OBJECT_KEYS    = {"젤다", "포켓몬", "야숨", "왕눈", "게임", "스위치"}
EXERCISE_OBJECT_KEYS = {"러닝", "헬스", "운동", "수영", "등산"}
CONTENT_OBJECT_KEYS  = {"영화", "드라마", "소설", "웹툰", "음악", "책", "오디오북"}

CLINICAL_OBJECT_KEYS = {
    "매복 사랑니 발치", "매복 사랑니", "사랑니 발치", "사랑니", "발치",
    "임플란트", "GBR", "GTR", "flap reflection", "flap", "판막",
    "봉합", "치주수술", "스케일링", "SRP", "치근활택",
}
ACADEMIC_OBJECT_KEYS = {"논문", "발표", "세미나"}
WORK_OBJECT_KEYS = CLINICAL_OBJECT_KEYS | ACADEMIC_OBJECT_KEYS

# 치과 술식 보정에 사용하는 집합 (AI 오분류 수정용)
DENTAL_OBJECT_TERMS: set[str] = {
    "발치", "사랑니", "사랑니 발치", "매복 사랑니", "매복 사랑니 발치",
    "GBR", "GTR", "flap", "flap reflection", "판막", "봉합",
    "임플란트", "치주수술", "스케일링", "SRP", "치근활택",
    "골이식", "치주재생",
}

# 시맨틱 힌트: (원문/태그에서 찾을 신호어 집합, 보정 category)
# primary_category가 미분류일 때만 적용한다.
_SEMANTIC_HINTS: list[tuple[set[str], str]] = [
    ({"향수", "퍼퓸", "perfume", "parfum"},            "물건"),
    ({"헬스", "러닝", "달리기", "등산", "수영", "트레이닝", "웨이트"}, "운동"),
    ({"데킬라", "사케", "위스키", "와인", "드립커피",
      "탄산음료", "에너지음료", "칵테일"},              "음료"),
    ({"게임", "플레이", "클리어", "퀘스트", "레벨업",
      "스팀", "콘솔", "rpg", "mmo"},                   "게임"),
    ({"영화", "드라마", "소설", "웹툰", "만화", "오디오북"}, "콘텐츠"),
    ({"소세지", "소시지", "파스타", "스테이크", "샌드위치",
      "버거", "타코", "부리토"},                        "음식"),
]


# 태그 역할 보정에 사용하는 집합
# OBJECT_ONLY: 사람이 아닌 물건 — social/context에서 object로 이동
OBJECT_ONLY_TERMS: set[str] = {"꽃", "향수", "선물", "케이크"}
# CONTEXT_ONLY: 상황/시간 — object/social에서 context로 이동
CONTEXT_ONLY_TERMS: set[str] = {"휴일", "주말", "평일", "회식", "운동중", "야식", "첫 경험"}
# 미분류에 준하는 weak category — 명확한 도메인 단서가 있으면 보정
WEAK_PRIMARY_CATEGORIES: set[str] = {"미분류", "활동", "상황"}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _match_objects(text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    working = text
    matched_kws: list[str] = []
    object_tags: list[str] = []
    subcategories: list[str] = []
    sensory_inferred: list[str] = []

    for kw in sorted(OBJECT_KEYWORD_MAP, key=len, reverse=True):
        if kw in working:
            matched_kws.append(kw)
            object_tags.extend(OBJECT_KEYWORD_MAP[kw])
            subcategories.extend(SUBCATEGORY_MAP.get(kw, []))
            sensory_inferred.extend(FOOD_SENSORY_INFERENCE.get(kw, []))
            working = working.replace(kw, "\x00" * len(kw))

    def dedup(lst: list[str]) -> list[str]:
        return list(dict.fromkeys(lst))

    return dedup(matched_kws), dedup(object_tags), dedup(subcategories), dedup(sensory_inferred)


def _preference_signal(rating: int) -> str:
    if rating >= 4:
        return "긍정적인 선호로 보입니다."
    if rating <= 2:
        return "부정적인 선호로 보입니다."
    return "중립적이거나 불명확한 선호입니다."


def _determine_categories(
    matched_kws: list[str],
    social_tags: list[str],
    place_tags: list[str],
    context_tags: list[str] | None = None,
) -> tuple[str, list[str]]:
    context_tags = context_tags or []
    has_clinical = any(kw in CLINICAL_OBJECT_KEYS for kw in matched_kws)
    has_academic = any(kw in ACADEMIC_OBJECT_KEYS for kw in matched_kws)
    has_work     = has_clinical or has_academic

    work_ctx_signals = {"임상", "수술", "술식", "진료", "전공의", "의국", "학습", "연구", "피드백", "발표", "세미나"}
    if not has_work and any(t in work_ctx_signals for t in context_tags):
        has_work     = True
        has_clinical = any(t in {"임상", "수술", "술식", "진료"} for t in context_tags)
        has_academic = any(t in {"학습", "연구", "피드백", "발표", "세미나"} for t in context_tags)

    has_food    = any(kw in FOOD_OBJECT_KEYS     for kw in matched_kws)
    has_drink   = any(kw in DRINK_OBJECT_KEYS    for kw in matched_kws)
    has_game    = any(kw in GAME_OBJECT_KEYS     for kw in matched_kws)
    has_content = any(kw in CONTENT_OBJECT_KEYS  for kw in matched_kws)
    has_exercise = any(kw in EXERCISE_OBJECT_KEYS for kw in matched_kws)

    ctx: list[str] = []

    # 우선순위: 직업 > 음식 > 음료 > 게임 > 콘텐츠 > 운동 > 장소 > 관계 > 미분류
    if has_work:
        primary = "직업"
        if has_clinical:
            ctx.append("임상")
            ctx.append("치과")
        if has_academic or any(t in {"학습", "연구", "피드백"} for t in context_tags):
            ctx.append("학습")
        if social_tags: ctx.append("관계")
        if place_tags:  ctx.append("장소")
    elif has_food:
        primary = "음식"
        if has_drink: ctx.append("음료")
        if place_tags:  ctx.append("장소")
        if social_tags: ctx.append("관계")
    elif has_drink:
        primary = "음료"
        if place_tags:  ctx.append("장소")
        if social_tags: ctx.append("관계")
    elif has_game:
        primary = "게임"
        if place_tags:  ctx.append("장소")
        if social_tags: ctx.append("관계")
    elif has_content:
        primary = "콘텐츠"
        if place_tags:  ctx.append("장소")
        if social_tags: ctx.append("관계")
    elif has_exercise:
        primary = "운동"
        if place_tags:  ctx.append("장소")
        if social_tags: ctx.append("관계")
    elif place_tags:
        primary = "장소"
        if social_tags: ctx.append("관계")
    elif social_tags:
        primary = "관계"
    else:
        primary = "미분류"

    return primary, list(dict.fromkeys(ctx))


# ── Rule-based classifier ─────────────────────────────────────────────────────

def classify_preference_rule_based(
    raw_text: str, rating: int, source: str = "RULE_BASED_NO_API_KEY"
) -> PreferenceClassification:
    matched_kws, object_tags, subcategories, sensory_inferred = _match_objects(raw_text)

    sensory_direct = [tag for kw, tag in SENSORY_DIRECT_MAP.items() if kw in raw_text]
    sensory_tags = list(dict.fromkeys(sensory_inferred + sensory_direct))

    context_tags = list(dict.fromkeys(
        tag for kw, tag in CONTEXT_KEYWORD_MAP.items() if kw in raw_text
    ))
    social_tags = list(dict.fromkeys(
        tag for kw, tag in SOCIAL_KEYWORD_MAP.items() if kw in raw_text
    ))
    place_tags = list(dict.fromkeys(
        tag for kw, tag in PLACE_KEYWORD_MAP.items() if kw in raw_text
    ))

    primary_category, context_categories = _determine_categories(
        matched_kws, social_tags, place_tags, context_tags
    )

    active_types = sum(
        1 for t in [object_tags, sensory_tags, context_tags, social_tags, place_tags] if t
    )
    confidence = round(min(0.40 + active_types * 0.10, 0.90), 2) if active_types else 0.30

    combined_tags = list(dict.fromkeys(
        object_tags + sensory_tags + context_tags + social_tags + place_tags
    ))

    return PreferenceClassification(
        primary_category=primary_category,
        context_categories=context_categories,
        subcategories=subcategories,
        tags=combined_tags,
        object_tags=object_tags,
        sensory_tags=sensory_tags,
        context_tags=context_tags,
        social_tags=social_tags,
        place_tags=place_tags,
        preference_signal=_preference_signal(rating),
        confidence=confidence,
        classifier_source=source,
    )


# ── AI classifier ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a preference classification engine for a Korean preference diary app called TasteMap.

The user inputs:
- raw_text: a short word, phrase, or sentence about something they liked or disliked
- rating: 1 to 5

Your task:
Extract structured preference data.

Return JSON only.

Schema:
{
  "primary_category": "string",
  "context_categories": ["string"],
  "subcategories": ["string"],
  "tags": ["string"],
  "object_tags": ["string"],
  "sensory_tags": ["string"],
  "context_tags": ["string"],
  "social_tags": ["string"],
  "place_tags": ["string"],
  "preference_signal": "string",
  "confidence": 0.0
}

Field definitions and roles (IMPORTANT — do not mix these):
- object_tags: CONCRETE SUBJECTS — food, drink, game, product, physical object, clinical procedure.
  e.g. 드립커피, 사케, 팩토리오, 꽃, 매복 사랑니, GBR, 향수
  WARNING: 꽃(flower), 향수(perfume) are OBJECTS not social. 사케·소주·데킬라 are DRINKS not food.
- sensory_tags: SENSORY / EVALUATIVE quality — taste, texture, feeling, mood, experiential signal.
  e.g. 고소함, 따뜻함, 진한맛, 어려움, 성장감, 기쁨, 뿌듯함
- context_tags: SITUATION / TIME / ACTIVITY context.
  e.g. 야식, 술자리, 휴일, 회식, 운동중, 첫 경험, 임상, 수술
  WARNING: 휴일(holiday), 주말(weekend) are CONTEXT not social or object.
- social_tags: PEOPLE / RELATIONSHIPS only — who was present.
  e.g. 혼자, 애인, 어머니, 교수님, 환자
  WARNING: 꽃, 휴일, 케이크 are NOT social. Only persons/relationships.
- place_tags: LOCATION.
  e.g. 카페, 맥도날드, 치과, 병원, 노포
- tags: combined flat list (union of all dimensions, deduplicated)

Allowed primary_category values:
- 음식
- 음료
- 게임
- 콘텐츠
- 운동
- 직업
- 학습
- 관계
- 장소
- 활동
- 물건
- 미분류

Category selection principles (apply in this priority order):
1. WORK/CLINICAL: Any mention of 발치, 사랑니, GBR, GTR, flap, 봉합, 임플란트, 치주수술, SRP, 수술, 술식, 진료, 의국, 전공의 → primary: "직업"
2. ACADEMIC: 논문, 발표, 세미나, 피드백, 공부, 연구 (without clinical terms) → primary: "직업" or "학습"
3. FOOD: Specific food names → primary: "음식"
4. DRINK: Coffee, alcohol, beverages WITHOUT food → primary: "음료". If combined with food → primary: "음식"
5. GAME: Game titles/gaming → primary: "게임"
6. CONTENT: 영화, 드라마, 책, 음악, 웹툰 → primary: "콘텐츠"
7. EXERCISE: 러닝, 헬스, 수영, 등산, 운동 → primary: "운동"
8. PLACE only: place experience with no concrete object → primary: "장소"
9. RELATION only: relationship experience → primary: "관계"
10. Unclear → "미분류"

Additional principles:
- "장소"는 구체적 대상이 없고 장소 경험 자체가 핵심일 때만 primary로 사용한다.
- "상황"은 primary_category로 사용하지 말고 context_categories로 분류하라.
- 가게/브랜드명은 place_tags에만 넣고 primary_category를 "장소"로 잡지 마라.
  예: "맥도날드에서 드립커피" → primary: "음료", place_tags: ["맥도날드"]
- 술식명(발치, GBR, 봉합 등)은 object_tags에 넣어라. context_tags에만 넣지 마라.
- 감정/평가("뿌듯함", "어려움", "따뜻함")는 sensory_tags에 넣고 primary로 만들지 마라.

CRITICAL RULES FOR CLINICAL/PROFESSIONAL ENTRIES (apply FIRST before other rules):
1. If the input mentions ANY of these terms — 발치, 사랑니, 매복 사랑니, 임플란트, GBR, GTR, flap, 봉합, 치주수술, 판막, SRP, 치근활택, 스케일링, 수술, 술식, 진료, 의국, 전공의, 논문, 발표, 세미나, 피드백 — classify as primary_category: "직업".
2. 발치, 사랑니, GBR, GTR, flap, flap reflection, 봉합, 임플란트 같은 술식명은 반드시 object_tags에 넣어라. context_tags에만 넣지 마라.
3. "힘들었음", "어려웠다", "잘 안 됐다", "엉망이었다" → sensory_tags에 "어려움" 추가.
4. "다음엔 더 잘할 수 있을 거 같아", "늘었다", "배웠다", "감 잡았다", "잘할 수 있을 것 같다" → sensory_tags에 "성장감" 추가.
5. 어렵지만 향후 개선 가능성을 느낀 기록은 단순 부정이 아니라 성장 경험으로 해석하라.
6. 치과/임상 술식이 있으면 context_categories에 "임상"과 "치과" 추가.
7. 논문/발표/세미나/공부/피드백이 있으면 context_categories에 "학습" 또는 "연구" 추가.
8. 절대 사용자의 능력·성격·정신상태를 진단하지 마라. "현재 기록상", "~로 보입니다" 같은 표현만 사용하라.

Semantic classification rules (IMPORTANT):
- Do NOT rely only on explicit keyword lists. Use general world knowledge.
- If raw_text is a well-known proper noun, infer its domain even if not in any keyword list.
  Examples:
  - "팩토리오" or "Factorio" → 게임 (factory automation game)
  - "데킬라" → 음료 (alcoholic drink)
  - "황천의 츠가이" → 콘텐츠 (manga/anime)
  - "초리조 소세지" → 음식 (food)
  - "애프터눈 딜라이트 향수" → 물건 (perfume/product)
  - "헬스" → 운동 (exercise at gym)
  - "GTR" in dental context → 직업
- If a place name appears with a concrete object, the object is primary, place goes to place_tags.
  Example: "맥도날드에서 드립커피" → primary: 음료, place_tags: 맥도날드
- If the input lists multiple distinct objects/activities, use "활동" as primary and list all in object_tags.
- Use "미분류" only when the target is genuinely unclear even with world knowledge.
- If you are uncertain about subcategory but clear about primary, use primary confidently and keep subcategories minimal.

General rules:
- Choose exactly one primary_category.
- CLINICAL content takes priority over food/game classification.
- Always populate object_tags, sensory_tags, context_tags, social_tags, place_tags. Use [] if nothing applies.
- tags = union of all dimension tags (deduplicated) + any other useful tags.
- Generate concise Korean tags.
- If the input is ambiguous, use "미분류" and low confidence.
- rating 4-5 → positive preference_signal. rating 1-2 → negative. rating 3 → neutral.
- For clinical entries with mixed signals (hard but hopeful) → describe both in preference_signal, no judgment.

Examples:

Input:
raw_text = "첫 매복 사랑니 발치... 힘들었음 하지만 다음엔 더 잘할 수 있을 거 같아"
rating = 4

Output:
{
  "primary_category": "직업",
  "context_categories": ["임상", "치과", "학습"],
  "subcategories": ["치과", "구강외과", "발치"],
  "object_tags": ["매복 사랑니", "사랑니 발치", "발치"],
  "sensory_tags": ["어려움", "성장감"],
  "context_tags": ["임상", "술식", "첫 경험"],
  "social_tags": [],
  "place_tags": ["치과"],
  "tags": ["매복 사랑니", "사랑니 발치", "발치", "어려움", "성장감", "임상", "술식", "첫 경험", "치과"],
  "preference_signal": "매복 사랑니 발치 술식이 어렵게 느껴졌으나 향후 개선 가능성을 긍정적으로 인식한 임상 경험으로 보입니다.",
  "confidence": 0.9
}

Input:
raw_text = "오늘 봉합이 너무 엉망이었다"
rating = 2

Output:
{
  "primary_category": "직업",
  "context_categories": ["임상", "치과"],
  "subcategories": ["치과", "수술", "봉합"],
  "object_tags": ["봉합"],
  "sensory_tags": ["어려움"],
  "context_tags": ["임상", "수술", "술식"],
  "social_tags": [],
  "place_tags": ["치과"],
  "tags": ["봉합", "어려움", "임상", "수술", "술식", "치과"],
  "preference_signal": "봉합 술식 수행에 대한 부정적 경험으로 보입니다.",
  "confidence": 0.88
}

Input:
raw_text = "오늘 GBR flap reflection이 생각보다 잘 됐다"
rating = 5

Output:
{
  "primary_category": "직업",
  "context_categories": ["임상", "치과"],
  "subcategories": ["치과", "치주과", "GBR", "수술"],
  "object_tags": ["GBR", "flap reflection", "flap", "치주수술"],
  "sensory_tags": ["잘됨", "숙련감"],
  "context_tags": ["임상", "수술", "술식"],
  "social_tags": [],
  "place_tags": ["치과"],
  "tags": ["GBR", "flap reflection", "flap", "치주수술", "잘됨", "숙련감", "임상", "수술", "술식", "치과"],
  "preference_signal": "GBR 수술 과정에서 술식 수행에 대한 긍정적 경험으로 보입니다.",
  "confidence": 0.9
}

Input:
raw_text = "교수님 피드백 받고 논문 discussion 방향이 좀 잡혔다"
rating = 5

Output:
{
  "primary_category": "직업",
  "context_categories": ["학습", "연구", "관계"],
  "subcategories": ["논문", "연구", "피드백"],
  "object_tags": ["논문", "discussion", "연구 방향"],
  "sensory_tags": ["명확해짐"],
  "context_tags": ["학습", "연구", "피드백"],
  "social_tags": ["교수님"],
  "place_tags": [],
  "tags": ["논문", "discussion", "연구 방향", "명확해짐", "학습", "연구", "피드백", "교수님"],
  "preference_signal": "연구 피드백을 통해 논문 방향이 정리된 긍정적 경험으로 보입니다.",
  "confidence": 0.87
}

Input:
raw_text = "오늘 분위기 좋은 노포 곱창전골집에서 애인이랑 소주 한잔 했음"
rating = 5

Output:
{
  "primary_category": "음식",
  "context_categories": ["장소", "관계", "상황"],
  "subcategories": ["한식", "전골", "내장요리", "술안주"],
  "object_tags": ["곱창전골", "소주"],
  "sensory_tags": ["진한맛", "국물"],
  "context_tags": ["술자리"],
  "social_tags": ["애인"],
  "place_tags": ["노포"],
  "tags": ["곱창전골", "소주", "진한맛", "국물", "술자리", "애인", "노포"],
  "preference_signal": "진한 한식 전골과 노포 술자리 분위기에 대한 긍정적 선호로 보입니다.",
  "confidence": 0.9
}

Input:
raw_text = "혼자 카페에서 아메리카노 마시면서 공부하니까 좋았다"
rating = 4

Output:
{
  "primary_category": "음료",
  "context_categories": ["장소", "상황"],
  "subcategories": ["커피", "아메리카노"],
  "object_tags": ["아메리카노", "커피"],
  "sensory_tags": ["쓴맛", "향긋함"],
  "context_tags": ["공부중"],
  "social_tags": ["혼자"],
  "place_tags": ["카페"],
  "tags": ["아메리카노", "커피", "쓴맛", "향긋함", "공부중", "혼자", "카페"],
  "preference_signal": "카페에서 혼자 공부하며 마시는 아메리카노에 대한 긍정적 선호로 보입니다.",
  "confidence": 0.88
}

Input:
raw_text = "드립커피가 고소해서 좋았다"
rating = 5

Output:
{
  "primary_category": "음료",
  "context_categories": [],
  "subcategories": ["커피", "드립커피"],
  "object_tags": ["드립커피", "커피"],
  "sensory_tags": ["고소함"],
  "context_tags": [],
  "social_tags": [],
  "place_tags": [],
  "tags": ["드립커피", "커피", "고소함"],
  "preference_signal": "드립커피의 고소한 향에 대한 긍정적 선호로 보입니다.",
  "confidence": 0.9
}

Input:
raw_text = "사케 맛있다"
rating = 5

Output:
{
  "primary_category": "음료",
  "context_categories": [],
  "subcategories": ["주류", "일본술"],
  "object_tags": ["사케", "술"],
  "sensory_tags": ["맛있음"],
  "context_tags": [],
  "social_tags": [],
  "place_tags": [],
  "tags": ["사케", "술", "맛있음"],
  "preference_signal": "사케에 대한 긍정적 선호로 보입니다.",
  "confidence": 0.88
}

Input:
raw_text = "애인이 휴일에 꽃 줘서 기뻤다"
rating = 5

Output:
{
  "primary_category": "관계",
  "context_categories": ["상황"],
  "subcategories": ["선물", "꽃"],
  "object_tags": ["꽃"],
  "sensory_tags": ["기쁨"],
  "context_tags": ["휴일"],
  "social_tags": ["애인"],
  "place_tags": [],
  "tags": ["꽃", "기쁨", "휴일", "애인"],
  "preference_signal": "애인에게 꽃을 받은 긍정적 경험으로 보입니다.",
  "confidence": 0.9
}

Input:
raw_text = "팩토리오"
rating = 5

Output:
{
  "primary_category": "게임",
  "context_categories": [],
  "subcategories": ["자동화", "시뮬레이션"],
  "object_tags": ["팩토리오"],
  "sensory_tags": [],
  "context_tags": [],
  "social_tags": [],
  "place_tags": [],
  "tags": ["팩토리오", "자동화", "시뮬레이션", "게임"],
  "preference_signal": "팩토리오라는 게임에 대한 긍정적 선호로 보입니다.",
  "confidence": 0.9
}

Input:
raw_text = "팩토리오에서 green circuit 제조 비율 알아보는 게 재밌다"
rating = 5

Output:
{
  "primary_category": "게임",
  "context_categories": ["학습"],
  "subcategories": ["자동화", "시뮬레이션", "공략"],
  "object_tags": ["팩토리오", "green circuit"],
  "sensory_tags": ["재미"],
  "context_tags": ["공략", "학습"],
  "social_tags": [],
  "place_tags": [],
  "tags": ["팩토리오", "green circuit", "제조 비율", "자동화", "공략", "재미"],
  "preference_signal": "팩토리오의 제작 비율 학습 과정에 대한 긍정적 경험으로 보입니다.",
  "confidence": 0.92
}

Input:
raw_text = "애프터눈 딜라이트 향수"
rating = 5

Output:
{
  "primary_category": "물건",
  "context_categories": [],
  "subcategories": ["향수"],
  "object_tags": ["애프터눈 딜라이트 향수"],
  "sensory_tags": [],
  "context_tags": [],
  "social_tags": [],
  "place_tags": [],
  "tags": ["애프터눈 딜라이트 향수", "향수"],
  "preference_signal": "해당 향수에 대한 긍정적 선호로 보입니다.",
  "confidence": 0.85
}

Input:
raw_text = "황천의 츠가이, 데킬라, 헬스, 초리조 소세지, 애프터눈 딜라이트 향수"
rating = 5

Output:
{
  "primary_category": "활동",
  "context_categories": ["혼합 취향 기록"],
  "subcategories": ["콘텐츠", "음료", "운동", "음식", "물건"],
  "object_tags": ["황천의 츠가이", "데킬라", "헬스", "초리조 소세지", "애프터눈 딜라이트 향수"],
  "sensory_tags": [],
  "context_tags": ["혼합 취향 기록"],
  "social_tags": [],
  "place_tags": [],
  "tags": ["황천의 츠가이", "데킬라", "헬스", "초리조 소세지", "애프터눈 딜라이트 향수", "만화", "술", "운동", "향수"],
  "preference_signal": "여러 대상에 대한 긍정적 선호를 한 번에 기록한 것으로 보입니다.",
  "confidence": 0.82
}
"""

_REQUIRED_FIELDS = {
    "primary_category", "context_categories", "subcategories", "tags",
    "object_tags", "sensory_tags", "context_tags", "social_tags", "place_tags",
    "preference_signal", "confidence",
}


def classify_preference_ai(raw_text: str, rating: int) -> PreferenceClassification:
    from openai import OpenAI  # lazy import — only needed when key is present

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f'raw_text = "{raw_text}"\nrating = {rating}'},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=600,
    )

    data = json.loads(response.choices[0].message.content)

    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        raise ValueError(f"AI response missing fields: {missing}")

    confidence = max(0.0, min(1.0, float(data["confidence"])))

    object_tags  = [str(v) for v in data.get("object_tags",  [])]
    sensory_tags = [str(v) for v in data.get("sensory_tags", [])]
    context_tags = [str(v) for v in data.get("context_tags", [])]
    social_tags  = [str(v) for v in data.get("social_tags",  [])]
    place_tags   = [str(v) for v in data.get("place_tags",   [])]

    ai_tags     = [str(v) for v in data.get("tags", [])]
    detail_tags = object_tags + sensory_tags + context_tags + social_tags + place_tags
    final_tags  = list(dict.fromkeys(ai_tags + detail_tags))

    return PreferenceClassification(
        primary_category=str(data["primary_category"]),
        context_categories=[str(v) for v in data.get("context_categories", [])],
        subcategories=[str(v) for v in data.get("subcategories", [])],
        tags=final_tags,
        object_tags=object_tags,
        sensory_tags=sensory_tags,
        context_tags=context_tags,
        social_tags=social_tags,
        place_tags=place_tags,
        preference_signal=str(data["preference_signal"]),
        confidence=round(confidence, 4),
        classifier_source="AI",
    )


# ── Public API ────────────────────────────────────────────────────────────────

def _is_quota_error(exc: Exception) -> bool:
    try:
        from openai import RateLimitError
        if not isinstance(exc, RateLimitError):
            return False
        code = getattr(exc, "code", "") or ""
        return "insufficient_quota" in code or "insufficient_quota" in str(exc)
    except ImportError:
        return False


def postprocess_semantic_classification(
    raw_text: str,
    c: PreferenceClassification,
) -> PreferenceClassification:
    """AI 오분류 보정: 치과 술식 교정 + 일반 시맨틱 힌트 교정."""
    all_pool = set(c.tags) | set(c.context_tags) | set(c.object_tags) | set(c.subcategories)

    # ── 1. 치과 술식 보정 ──────────────────────────────────────────────────────
    dental_found = [t for t in all_pool if t in DENTAL_OBJECT_TERMS]

    new_obj      = list(c.object_tags)
    seen_obj     = set(c.object_tags)
    new_primary  = c.primary_category
    new_ctx_cats = list(c.context_categories)

    if dental_found:
        for t in dental_found:
            if t not in seen_obj:
                new_obj.append(t)
                seen_obj.add(t)
        if new_primary in ("미분류", "상황", "장소"):
            new_primary = "직업"
        for cat in ("임상", "치과"):
            if cat not in new_ctx_cats:
                new_ctx_cats.insert(0, cat)

    # ── 2. 시맨틱 힌트 보정 (미분류/활동/상황 등 weak category일 때 적용) ─────────
    if new_primary in WEAK_PRIMARY_CATEGORIES:
        text_lower = raw_text.lower()
        pool_lower = {t.lower() for t in all_pool}
        # 3개 이상의 쉼표 = 의도적 혼합 리스트 → 보정하지 않음 (활동 유지)
        is_list_input = raw_text.count(",") >= 3
        if not is_list_input:
            for signals, target_cat in _SEMANTIC_HINTS:
                if any(s in text_lower or s in pool_lower for s in signals):
                    new_primary = target_cat
                    break

    # ── 3. 태그 역할 보정: social/context 오염 수정 ────────────────────────────
    new_soc = list(c.social_tags)
    new_ctx = list(c.context_tags)
    seen_obj_set = set(new_obj)
    seen_ctx_set = set(new_ctx)

    # social_tags에서 물건/상황 태그 분리
    corrected_soc: list[str] = []
    for t in new_soc:
        if t in OBJECT_ONLY_TERMS:
            if t not in seen_obj_set:
                new_obj.append(t); seen_obj_set.add(t)
        elif t in CONTEXT_ONLY_TERMS:
            if t not in seen_ctx_set:
                new_ctx.append(t); seen_ctx_set.add(t)
        else:
            corrected_soc.append(t)
    new_soc = corrected_soc

    # context_tags에서 물건 태그 분리
    corrected_ctx: list[str] = []
    for t in new_ctx:
        if t in OBJECT_ONLY_TERMS:
            if t not in seen_obj_set:
                new_obj.append(t); seen_obj_set.add(t)
        else:
            corrected_ctx.append(t)
    new_ctx = corrected_ctx

    # object_tags에서 상황 태그 분리
    corrected_obj: list[str] = []
    for t in new_obj:
        if t in CONTEXT_ONLY_TERMS:
            if t not in seen_ctx_set:
                new_ctx.append(t); seen_ctx_set.add(t)
        else:
            corrected_obj.append(t)
    new_obj = corrected_obj

    changed = (
        new_primary != c.primary_category
        or new_obj   != c.object_tags
        or new_soc   != c.social_tags
        or new_ctx   != c.context_tags
        or new_ctx_cats != c.context_categories
    )
    if not changed:
        return c  # 변경 없음 — 복사 불필요

    merged = list(dict.fromkeys(
        new_obj + c.sensory_tags + new_ctx + new_soc + c.place_tags + c.tags
    ))

    return PreferenceClassification(
        primary_category=new_primary,
        context_categories=list(dict.fromkeys(new_ctx_cats)),
        subcategories=c.subcategories,
        tags=merged,
        object_tags=new_obj,
        sensory_tags=c.sensory_tags,
        context_tags=new_ctx,
        social_tags=new_soc,
        place_tags=c.place_tags,
        preference_signal=c.preference_signal,
        confidence=c.confidence,
        classifier_source=c.classifier_source,
    )


def _normalize_classification(
    c: PreferenceClassification, raw_text: str = ""
) -> PreferenceClassification:
    """태그 정규화 + 시맨틱 오분류 보정을 적용해 PreferenceClassification을 반환."""
    from tastemap.legacy.tag_normalizer import normalize_tags

    obj      = normalize_tags(c.object_tags)
    sen      = normalize_tags(c.sensory_tags)
    ctx      = normalize_tags(c.context_tags)
    soc      = normalize_tags(c.social_tags)
    plc      = normalize_tags(c.place_tags)
    subs     = normalize_tags(c.subcategories)
    ctx_cats = normalize_tags(c.context_categories)
    merged   = list(dict.fromkeys(normalize_tags(c.tags) + obj + sen + ctx + soc + plc))

    normalized = PreferenceClassification(
        primary_category=c.primary_category,
        context_categories=ctx_cats,
        subcategories=subs,
        tags=merged,
        object_tags=obj,
        sensory_tags=sen,
        context_tags=ctx,
        social_tags=soc,
        place_tags=plc,
        preference_signal=c.preference_signal,
        confidence=c.confidence,
        classifier_source=c.classifier_source,
    )
    return postprocess_semantic_classification(raw_text, normalized)


def classify_preference(raw_text: str, rating: int) -> PreferenceClassification:
    if os.environ.get("OPENAI_API_KEY"):
        try:
            result = classify_preference_ai(raw_text, rating)
            return _normalize_classification(result, raw_text)
        except Exception as exc:
            fallback_source = (
                "RULE_BASED_INSUFFICIENT_QUOTA"
                if _is_quota_error(exc)
                else "RULE_BASED_FALLBACK"
            )
    else:
        fallback_source = "RULE_BASED_NO_API_KEY"

    result = classify_preference_rule_based(raw_text, rating, source=fallback_source)
    return _normalize_classification(result, raw_text)
