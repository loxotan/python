from collections import Counter, defaultdict

from tastemap.legacy.models import PreferenceEntry
from tastemap.legacy.tag_normalizer import normalize_tag, normalize_tags


# ── 기존 함수 ─────────────────────────────────────────────────────────────────

def get_top_tags(entries: list[PreferenceEntry], limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for entry in entries:
        counter.update(normalize_tags(entry.tags))
    return counter.most_common(limit)


def get_average_rating_by_category(entries: list[PreferenceEntry]) -> dict[str, float]:
    totals: dict[str, list[int]] = {}
    for entry in entries:
        totals.setdefault(entry.primary_category, []).append(entry.rating)
    return {cat: round(sum(r) / len(r), 2) for cat, r in totals.items()}


def get_top_object_tags(entries: list[PreferenceEntry], limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for entry in entries:
        counter.update(normalize_tags(entry.object_tags))
    return counter.most_common(limit)


def get_top_sensory_tags(entries: list[PreferenceEntry], limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for entry in entries:
        counter.update(normalize_tags(entry.sensory_tags))
    return counter.most_common(limit)


def get_top_context_tags(entries: list[PreferenceEntry], limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for entry in entries:
        counter.update(normalize_tags(entry.context_tags))
    return counter.most_common(limit)


def get_top_social_tags(entries: list[PreferenceEntry], limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for entry in entries:
        counter.update(normalize_tags(entry.social_tags))
    return counter.most_common(limit)


def get_top_place_tags(entries: list[PreferenceEntry], limit: int = 10) -> list[tuple[str, int]]:
    counter: Counter = Counter()
    for entry in entries:
        counter.update(normalize_tags(entry.place_tags))
    return counter.most_common(limit)


def get_average_rating_by_tag(
    entries: list[PreferenceEntry],
    tag_field: str,
    limit: int = 10,
) -> list[tuple[str, float, int]]:
    # Sorted by average rating descending; ties broken by count descending.
    tag_ratings: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        for tag in getattr(entry, tag_field, []):
            tag_ratings[normalize_tag(tag)].append(entry.rating)
    results = [
        (tag, round(sum(r) / len(r), 2), len(r))
        for tag, r in tag_ratings.items()
    ]
    results.sort(key=lambda x: (-x[1], -x[2]))
    return results[:limit]


# ── 기본 통계 개선 ────────────────────────────────────────────────────────────

def get_top_tags_by_rating(
    entries: list[PreferenceEntry],
    ascending: bool = False,
    limit: int = 10,
    min_count: int = 1,
) -> list[tuple[str, float, int]]:
    """combined tags 기준 평균 평점 정렬.
    ascending=False → 높은 평점 순, ascending=True → 낮은 평점 순.
    min_count로 최소 등장 횟수를 필터링할 수 있다.
    반환: [(tag, avg_rating, count), ...]
    """
    tag_ratings: dict[str, list[int]] = defaultdict(list)
    for entry in entries:
        for tag in entry.tags:
            tag_ratings[normalize_tag(tag)].append(entry.rating)
    results = [
        (tag, round(sum(r) / len(r), 2), len(r))
        for tag, r in tag_ratings.items()
        if len(r) >= min_count
    ]
    if ascending:
        results.sort(key=lambda x: (x[1], -x[2]))   # 낮은 평점 오름차순
    else:
        results.sort(key=lambda x: (-x[1], -x[2]))  # 높은 평점 내림차순
    return results[:limit]


# ── 취향 프로필 내부 헬퍼 ──────────────────────────────────────────────────────

def _category_avg_count(
    entries: list[PreferenceEntry], top: int = 5
) -> list[tuple[str, float, int]]:
    cat_ratings: dict[str, list[int]] = defaultdict(list)
    for e in entries:
        cat_ratings[e.primary_category].append(e.rating)
    results = [
        (cat, round(sum(r) / len(r), 2), len(r))
        for cat, r in cat_ratings.items()
    ]
    results.sort(key=lambda x: (-x[1], -x[2]))
    return results[:top]


def _tag_avg_count(
    entries: list[PreferenceEntry], field: str, top: int = 10
) -> list[tuple[str, float, int]]:
    return get_average_rating_by_tag(entries, field, limit=top)


# ── 취향 프로필 생성 ──────────────────────────────────────────────────────────

def build_preference_profile(entries: list[PreferenceEntry]) -> dict:
    """저장된 기록들을 분석해 취향 프로필 데이터를 반환한다."""
    empty: dict = {
        "total_entries": 0,
        "average_rating": 0.0,
        "positive_categories": [],
        "negative_categories": [],
        "top_positive_objects": [],
        "top_positive_sensory": [],
        "top_positive_contexts": [],
        "top_positive_social": [],
        "top_positive_places": [],
        "top_negative_objects": [],
        "top_negative_sensory": [],
    }
    if not entries:
        return empty

    try:
        total = len(entries)
        avg_rating = round(sum(e.rating for e in entries) / total, 1)

        positive = [e for e in entries if e.rating >= 4]
        negative = [e for e in entries if e.rating <= 2]

        return {
            "total_entries": total,
            "average_rating": avg_rating,
            "positive_categories": _category_avg_count(positive),
            "negative_categories": _category_avg_count(negative),
            "top_positive_objects":  _tag_avg_count(positive, "object_tags"),
            "top_positive_sensory":  _tag_avg_count(positive, "sensory_tags"),
            "top_positive_contexts": _tag_avg_count(positive, "context_tags"),
            "top_positive_social":   _tag_avg_count(positive, "social_tags"),
            "top_positive_places":   _tag_avg_count(positive, "place_tags"),
            "top_negative_objects":  _tag_avg_count(negative, "object_tags"),
            "top_negative_sensory":  _tag_avg_count(negative, "sensory_tags"),
        }
    except Exception:
        return empty


# ── 자연어 요약 문장 생성 ─────────────────────────────────────────────────────

def generate_profile_sentences(profile: dict) -> list[str]:
    """프로필 dict를 받아 rule-based 자연어 요약 문장 리스트를 반환한다."""
    total = profile.get("total_entries", 0)

    if total == 0:
        return ["아직 기록이 없어 분석이 불가능합니다."]

    sentences: list[str] = []
    avg = profile.get("average_rating", 0.0)
    sentences.append(
        f"현재 {total}개의 기록을 기준으로 전체 평균 평점은 {avg:.1f}점입니다."
    )

    if total < 3:
        sentences.append(
            "아직 충분한 기록이 없어 뚜렷한 패턴은 제한적입니다."
        )
        return sentences

    pos_cats = profile.get("positive_categories", [])
    if pos_cats:
        names = ", ".join(c[0] for c in pos_cats[:3])
        sentences.append(
            f"높은 평점(4점 이상)을 주는 카테고리는 현재 기록상 {names} 순으로 나타납니다."
        )

    top_obj = profile.get("top_positive_objects", [])
    if top_obj:
        names = ", ".join(t[0] for t in top_obj[:4])
        sentences.append(f"좋아하는 대상으로는 {names} 같은 항목이 반복 관찰됩니다.")

    top_sen = profile.get("top_positive_sensory", [])
    if top_sen:
        names = ", ".join(t[0] for t in top_sen[:4])
        sentences.append(
            f"감각적으로는 {names} 같은 요소에 긍정적으로 반응하는 것으로 보입니다."
        )

    top_soc = profile.get("top_positive_social", [])
    if top_soc:
        names = ", ".join(t[0] for t in top_soc[:3])
        sentences.append(
            f"관계 맥락에서는 {names}와 함께한 경험에서 높은 평점이 관찰됩니다."
        )

    top_ctx = profile.get("top_positive_contexts", [])
    if top_ctx:
        names = ", ".join(t[0] for t in top_ctx[:3])
        sentences.append(f"자주 등장하는 긍정적 상황은 현재 기록상 {names} 등입니다.")

    top_place = profile.get("top_positive_places", [])
    if top_place:
        names = ", ".join(t[0] for t in top_place[:3])
        sentences.append(f"긍정적 경험이 많이 일어나는 장소로는 {names} 등이 관찰됩니다.")

    neg_obj = profile.get("top_negative_objects", [])
    neg_sen = profile.get("top_negative_sensory", [])
    neg_items = [t[0] for t in neg_obj[:2]] + [t[0] for t in neg_sen[:2]]
    if neg_items:
        names = ", ".join(list(dict.fromkeys(neg_items))[:4])
        sentences.append(
            f"낮은 평점(2점 이하) 기록에서는 {names} 같은 요소가 관찰됩니다."
        )

    return sentences


# ── 마인드맵 데이터 생성 ──────────────────────────────────────────────────────

_FIELD_TYPE: dict[str, str] = {
    "object_tags":    "object",
    "sensory_tags":   "sensory",
    "context_tags":   "context",
    "social_tags":    "social",
    "place_tags":     "place",
    "subcategories":  "subcategory",
}


def build_mindmap_data(  # noqa: C901
    entries: list[PreferenceEntry],
    min_count: int = 1,
    include_types: set[str] | None = None,
    use_tag_hierarchy: bool = False,
) -> dict:
    """entries를 분석해 마인드맵 노드/엣지 데이터를 반환한다.

    개선사항:
    - object tag의 domain(태그 고유 카테고리)을 우선 사용 → 소주가 음식 밑에만 붙지 않음
    - sensory_tags를 object 노드에 연결 → "드립커피 → 고소함" 구조
    - place_tags를 첫 번째 object에 연결
    - use_tag_hierarchy=True 시 object_group 계층 생성
    """
    always_included = {"root", "category"}
    allowed = (include_types | always_included) if include_types is not None else None

    node_ratings: dict[str, list[int]] = defaultdict(list)
    node_meta:    dict[str, dict]      = {}
    edge_weight:  dict[tuple[str, str], int] = defaultdict(int)

    if not entries:
        return {
            "nodes": [{"id": "root", "label": "나의 취향", "type": "root",
                       "count": 0, "avg_rating": 0.0}],
            "edges": [],
        }

    # 외부 헬퍼 import
    try:
        from tastemap.legacy.tag_domain import get_tag_domain as _get_domain
    except ImportError:
        _get_domain = lambda t: None  # noqa: E731

    # 계층화 전처리
    qualifying_parents: set[str] = set()
    _get_parent = lambda t: None  # noqa: E731
    if use_tag_hierarchy:
        try:
            from tastemap.legacy.tag_hierarchy import get_parent_tag as _get_parent
            _pch: dict[str, set[str]] = defaultdict(set)
            for entry in entries:
                for raw_tag in entry.object_tags:
                    tag = normalize_tag(raw_tag)
                    p = _get_parent(tag)
                    if p:
                        _pch[p].add(tag)
            qualifying_parents = {p for p, c in _pch.items() if len(c) >= 2}
        except ImportError:
            pass

    root_id = "root"
    node_meta[root_id] = {"label": "나의 취향", "type": "root"}

    # 허용 타입별 플래그 (루프 외부에서 계산해 반복 호출 비용 절감)
    obj_allowed   = allowed is None or "object"       in allowed
    grp_allowed   = allowed is None or "object_group" in allowed
    sen_allowed   = allowed is None or "sensory"      in allowed
    ctx_allowed   = allowed is None or "context"      in allowed
    soc_allowed   = allowed is None or "social"       in allowed
    plc_allowed   = allowed is None or "place"        in allowed
    sub_allowed   = allowed is None or "subcategory"  in allowed

    for entry in entries:
        node_ratings[root_id].append(entry.rating)

        # entry의 primary category (항상 생성)
        entry_cat_id = f"category:{entry.primary_category}"
        node_meta[entry_cat_id] = {"label": entry.primary_category, "type": "category"}
        node_ratings[entry_cat_id].append(entry.rating)
        edge_weight[(root_id, entry_cat_id)] += 1
        _cats_added: set[str] = {entry.primary_category}

        def _cat_id(cat_name: str) -> str:
            """domain category 노드를 생성하고 root→cat 엣지를 추가 (entry당 1회)."""
            cid = f"category:{cat_name}"
            node_meta[cid] = {"label": cat_name, "type": "category"}
            if cat_name not in _cats_added:
                node_ratings[cid].append(entry.rating)
                edge_weight[(root_id, cid)] += 1
                _cats_added.add(cat_name)
            return cid

        # ── object_tags ───────────────────────────────────────────────────────
        effective_obj_ids: list[str] = []

        if obj_allowed:
            for raw_tag in entry.object_tags:
                tag = normalize_tag(raw_tag)

                # domain-aware category routing
                domain   = _get_domain(tag)
                obj_cat  = _cat_id(domain) if domain else entry_cat_id

                # 계층화 처리
                if use_tag_hierarchy and grp_allowed:
                    if tag in qualifying_parents:
                        # 이 태그 자체가 parent group
                        tid = f"object_group:{tag}"
                        node_meta[tid] = {"label": tag, "type": "object_group"}
                        node_ratings[tid].append(entry.rating)
                        edge_weight[(obj_cat, tid)] += 1
                        effective_obj_ids.append(tid)
                    else:
                        p = _get_parent(tag)
                        if p and p in qualifying_parents:
                            gid = f"object_group:{p}"
                            node_meta[gid] = {"label": p, "type": "object_group"}
                            node_ratings[gid].append(entry.rating)
                            edge_weight[(obj_cat, gid)] += 1
                            tid = f"object:{tag}"
                            node_meta[tid] = {"label": tag, "type": "object"}
                            node_ratings[tid].append(entry.rating)
                            edge_weight[(gid, tid)] += 1
                            effective_obj_ids.append(tid)
                        else:
                            tid = f"object:{tag}"
                            node_meta[tid] = {"label": tag, "type": "object"}
                            node_ratings[tid].append(entry.rating)
                            edge_weight[(obj_cat, tid)] += 1
                            effective_obj_ids.append(tid)
                else:
                    tid = f"object:{tag}"
                    node_meta[tid] = {"label": tag, "type": "object"}
                    node_ratings[tid].append(entry.rating)
                    edge_weight[(obj_cat, tid)] += 1
                    effective_obj_ids.append(tid)

        # ── sensory_tags: object가 있으면 각 object에 연결 ─────────────────────
        if sen_allowed:
            for raw_tag in entry.sensory_tags:
                tag = normalize_tag(raw_tag)
                tid = f"sensory:{tag}"
                node_meta[tid] = {"label": tag, "type": "sensory"}
                node_ratings[tid].append(entry.rating)
                if effective_obj_ids:
                    for oid in effective_obj_ids:
                        edge_weight[(oid, tid)] += 1
                else:
                    edge_weight[(entry_cat_id, tid)] += 1

        # ── context_tags: entry category에 연결 ───────────────────────────────
        if ctx_allowed:
            for raw_tag in entry.context_tags:
                tag = normalize_tag(raw_tag)
                tid = f"context:{tag}"
                node_meta[tid] = {"label": tag, "type": "context"}
                node_ratings[tid].append(entry.rating)
                edge_weight[(entry_cat_id, tid)] += 1

        # ── social_tags: entry category에 연결 ───────────────────────────────
        if soc_allowed:
            for raw_tag in entry.social_tags:
                tag = normalize_tag(raw_tag)
                tid = f"social:{tag}"
                node_meta[tid] = {"label": tag, "type": "social"}
                node_ratings[tid].append(entry.rating)
                edge_weight[(entry_cat_id, tid)] += 1

        # ── place_tags: 첫 번째 object 또는 entry category에 연결 ─────────────
        if plc_allowed:
            anchor = effective_obj_ids[0] if effective_obj_ids else entry_cat_id
            for raw_tag in entry.place_tags:
                tag = normalize_tag(raw_tag)
                tid = f"place:{tag}"
                node_meta[tid] = {"label": tag, "type": "place"}
                node_ratings[tid].append(entry.rating)
                edge_weight[(anchor, tid)] += 1

        # ── subcategories: entry category에 연결 ─────────────────────────────
        if sub_allowed:
            for raw_tag in entry.subcategories:
                tag = normalize_tag(raw_tag)
                tid = f"subcategory:{tag}"
                node_meta[tid] = {"label": tag, "type": "subcategory"}
                node_ratings[tid].append(entry.rating)
                edge_weight[(entry_cat_id, tid)] += 1

    valid_nodes: set[str] = set()
    nodes: list[dict] = []
    for nid, meta in node_meta.items():
        ratings = node_ratings[nid]
        count   = len(ratings)
        avg     = round(sum(ratings) / count, 2) if ratings else 0.0
        ntype   = meta["type"]

        if ntype not in always_included and count < min_count:
            continue
        if allowed is not None and ntype not in allowed:
            continue

        valid_nodes.add(nid)
        nodes.append({"id": nid, "label": meta["label"],
                      "type": ntype, "count": count, "avg_rating": avg})

    edges: list[dict] = [
        {"source": src, "target": tgt, "weight": w}
        for (src, tgt), w in edge_weight.items()
        if src in valid_nodes and tgt in valid_nodes
    ]

    return {"nodes": nodes, "edges": edges}


# ── 연관 태그 분석 ────────────────────────────────────────────────────────────

_ALL_TAG_FIELDS: list[str] = [
    "object_tags", "sensory_tags", "context_tags",
    "social_tags", "place_tags", "subcategories", "tags",
]

_FIELD_PRIORITY: dict[str, int] = {
    "object_tags": 1, "sensory_tags": 2, "context_tags": 3,
    "social_tags": 4, "place_tags": 5, "subcategories": 6, "tags": 7,
}


def get_related_tags(
    entries: list[PreferenceEntry],
    query_tag: str,
    limit: int = 20,
    include_fields: list[str] | None = None,
) -> dict:
    """query_tag가 포함된 entry에서 함께 등장한 태그를 빈도순으로 반환한다.

    반환 형식:
    {
        "query": str,
        "matched_entries_count": int,
        "related_tags": [{"tag": str, "count": int, "avg_rating": float, "fields": list[str]}, ...],
        "matched_entries": [{"id", "raw_text", "rating", "primary_category", "created_at"}, ...]
    }
    """
    fields = include_fields or _ALL_TAG_FIELDS
    query  = normalize_tag(query_tag.strip())

    if not query:
        return {"query": query, "matched_entries_count": 0,
                "related_tags": [], "matched_entries": []}

    def _matches(tag_norm: str) -> bool:
        if tag_norm == query:
            return True
        if len(query) >= 2:
            return query in tag_norm or tag_norm in query
        return False

    # related tag → list of ratings (one per matched entry)
    related_ratings: dict[str, list[int]] = defaultdict(list)
    # related tag → set of field names where it appeared
    related_fields:  dict[str, set[str]]  = defaultdict(set)
    matched_entries: list[PreferenceEntry] = []

    for entry in entries:
        # entry 안의 모든 태그를 정규화 + 중복 제거, 어느 field 출신인지 추적
        tag_to_fields: dict[str, set[str]] = {}
        for field in fields:
            for raw in getattr(entry, field, []) or []:
                norm = normalize_tag(raw)
                if norm:
                    tag_to_fields.setdefault(norm, set()).add(field)

        # query와 매칭되는 태그 집합
        matching = {t for t in tag_to_fields if _matches(t)}
        if not matching:
            continue

        matched_entries.append(entry)

        # query와 매칭된 태그를 제외한 나머지를 related 후보로 집계
        for tag, tag_fields in tag_to_fields.items():
            if tag not in matching:
                related_ratings[tag].append(entry.rating)
                related_fields[tag].update(tag_fields)

    # 결과 조립
    related: list[dict] = []
    for tag, ratings in related_ratings.items():
        count = len(ratings)
        avg   = round(sum(ratings) / count, 2) if ratings else 0.0
        sorted_fields = sorted(
            related_fields[tag],
            key=lambda f: _FIELD_PRIORITY.get(f, 99),
        )
        related.append({"tag": tag, "count": count,
                        "avg_rating": avg, "fields": sorted_fields})

    related.sort(key=lambda x: (-x["count"], -x["avg_rating"], x["tag"]))

    return {
        "query":                query,
        "matched_entries_count": len(matched_entries),
        "related_tags":          related[:limit],
        "matched_entries": [
            {
                "id":               e.id,
                "raw_text":         e.raw_text,
                "rating":           e.rating,
                "primary_category": e.primary_category,
                "created_at":       e.created_at,
            }
            for e in matched_entries
        ],
    }


def get_time_perspective_summary(entries: list[PreferenceEntry]) -> dict:
    """past/present/future/unspecified별 기록 수와 평균 평점을 반환한다."""
    groups: dict[str, list[int]] = {"past": [], "present": [], "future": [], "unspecified": []}
    for e in entries:
        tp = getattr(e, "time_perspective", "unspecified") or "unspecified"
        if tp not in groups:
            tp = "unspecified"
        groups[tp].append(e.rating)
    result = {}
    for key, ratings in groups.items():
        count = len(ratings)
        avg   = round(sum(ratings) / count, 2) if ratings else 0.0
        result[key] = {"count": count, "avg_rating": avg}
    return result


def get_searchable_tags(
    entries: list[PreferenceEntry],
    limit: int = 30,
) -> list[tuple[str, int]]:
    """모든 필드를 합산해 검색 가능한 태그를 등장 횟수순으로 반환한다.
    한 entry 안에서 같은 태그가 여러 필드에 있어도 1회로 계산한다.
    """
    tag_counts: Counter = Counter()
    for entry in entries:
        seen: set[str] = set()
        for field in _ALL_TAG_FIELDS:
            for raw in getattr(entry, field, []) or []:
                norm = normalize_tag(raw)
                if norm and norm not in seen:
                    tag_counts[norm] += 1
                    seen.add(norm)
    result = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
    return result[:limit]
