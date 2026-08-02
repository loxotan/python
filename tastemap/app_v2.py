import streamlit as st

from ai_analyzer_v2 import (
    analyze_input, find_related_entries, generate_insight,
    generate_pattern_insight, pattern_confidence,
)
from db_v2 import (
    add_entry, delete_entry, get_entry, init_db, list_entries,
    update_full_entry, update_insight_only, update_pattern_insight,
    update_tags_and_insight,
)
from models_v2 import AnalysisResult, InsightResult, TasteEntry

init_db()

# ── session_state 초기화 ──────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "v2_stage":           "input",   # "input" | "tagging"
    "v2_raw_input":       "",
    "v2_rating":          3,
    "v2_analysis":        None,
    "v2_custom_tags":     [],
    "v2_tag_ver":         0,
    "v2_last_saved":      None,      # TasteEntry | None
    "v2_editing_id":      None,      # int | None
    "v2_edit_custom":     [],        # extra tags added during edit
    "v2_edit_tag_ver":    0,
    "v2_edit_pending":    {},        # {entry_id: AnalysisResult} 재분석 결과 임시 보관
    "v2_cand_tver":       {},        # {entry_id: int} candidate multiselect 버전
    "v2_form_ver":        0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────

def _stars(rating: int) -> str:
    return "★" * rating + "☆" * (5 - rating)


def _tag_chips(tags: list[str], color: str = "#dbeafe", text_color: str = "#1e40af",
               font_size: str = "0.82rem") -> str:
    return " ".join(
        f"<span style='background:{color};color:{text_color};padding:2px 10px;"
        f"border-radius:12px;font-size:{font_size};white-space:nowrap;'>{t}</span>"
        for t in tags
    )


def _run_pattern_flow(entry_id: int) -> None:
    """저장 후 패턴 분석을 실행하고 DB를 업데이트한다."""
    from db_v2 import get_entry as _get, list_entries as _list
    saved = _get(entry_id)
    if saved is None:
        return
    all_entries = _list()
    related = find_related_entries(saved, all_entries)
    conf = pattern_confidence(related)
    if conf in ("medium", "high"):
        result = generate_pattern_insight(saved, related)
        if result.get("pattern_insight"):
            update_pattern_insight(
                entry_id,
                result["pattern_insight"],
                result["pattern_confidence"],
                [e.id for e in related if e.id is not None],
            )


def _debug_insight(r: "InsightResult") -> None:
    """터미널에 추론 중간값을 출력한다 (개발 확인용)."""
    print(f"\n[INSIGHT DEBUG]")
    print(f"  depth    : {r.insight_depth}  mode: {r.insight_mode}")
    print(f"  context  : {r.key_context}")
    print(f"  rule     : {r.preference_rule or '—'}")
    print(f"  contrast : {r.preference_contrast or '—'}")
    print(f"  insight  : {r.one_line_insight}")


def _build_tag_type_map(analysis: "AnalysisResult") -> dict[str, str]:
    m = {t: "source" for t in analysis.source_tags}
    m.update({t: "feature" for t in analysis.feature_tags})
    m.update({t: "preference" for t in analysis.preference_tags})
    return m


def _render_selected_tags(entry: TasteEntry) -> None:
    """선택 태그를 타입별 색상으로 표시. 타입 정보 없는 구 항목도 처리."""
    types = entry.selected_tag_types
    feature   = [t for t in entry.selected_tags if types.get(t) == "feature"]
    pref      = [t for t in entry.selected_tags if types.get(t) == "preference"]
    candidate = [t for t in entry.selected_tags if types.get(t) == "candidate"]
    other     = [t for t in entry.selected_tags
                 if types.get(t) not in ("feature", "preference", "source", "candidate")]

    if feature:
        st.markdown(_tag_chips(feature, "#dbeafe", "#1e40af"), unsafe_allow_html=True)
    if pref:
        st.markdown(_tag_chips(pref, "#d1fae5", "#065f46"), unsafe_allow_html=True)
    if candidate:
        st.markdown(_tag_chips(candidate, "#fef9c3", "#854d0e"), unsafe_allow_html=True)
    if other:
        st.markdown(_tag_chips(other, "#fef3c7", "#92400e"), unsafe_allow_html=True)
    # 타입 정보 없는 구 항목 fallback
    if not feature and not pref and not candidate and not other and entry.selected_tags:
        st.markdown(_tag_chips(entry.selected_tags), unsafe_allow_html=True)





# ── 섹션 1: 입력 폼 ───────────────────────────────────────────────────────────

def _render_input_form() -> None:
    fver = st.session_state["v2_form_ver"]
    st.subheader("새 기록 추가")

    raw_input = st.text_area(
        "무엇을 경험했나요?",
        placeholder="예: 애인과 조용한 카페에서 커피 한 잔  /  팩토리오",
        key=f"v2_input_{fver}",
        help="짧은 단어나 고유명사 하나, 또는 상황을 담은 문장 모두 가능합니다.",
    )

    rating = st.radio(
        "별점",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: _stars(x),
        horizontal=True,
        index=st.session_state["v2_rating"] - 1,
        key=f"v2_rating_radio_{fver}",
    )

    col_analyze, col_clear = st.columns([3, 1])
    with col_analyze:
        analyze_btn = st.button("분석", type="primary", use_container_width=True)
    with col_clear:
        if st.session_state["v2_last_saved"]:
            if st.button("저장 기록 숨기기", use_container_width=True):
                st.session_state["v2_last_saved"] = None
                st.rerun()

    if analyze_btn:
        if not raw_input.strip():
            st.warning("내용을 입력해주세요.")
            return
        with st.spinner("분석 중..."):
            result = analyze_input(raw_input.strip(), rating)
        st.session_state["v2_raw_input"] = raw_input.strip()
        st.session_state["v2_rating"] = rating
        st.session_state["v2_analysis"] = result
        st.session_state["v2_custom_tags"] = []
        st.session_state["v2_tag_ver"] += 1
        ms_key = f"v2_ms_{st.session_state['v2_tag_ver']}"
        st.session_state[ms_key] = list(result.recommended_tags)
        st.session_state["v2_stage"] = "tagging"
        st.session_state["v2_last_saved"] = None
        st.rerun()


# ── 섹션 2: 태그 선택 ─────────────────────────────────────────────────────────

def _render_tagging() -> None:
    analysis = st.session_state["v2_analysis"]
    raw_input = st.session_state["v2_raw_input"]
    rating = st.session_state["v2_rating"]
    tver = st.session_state["v2_tag_ver"]
    ms_key = f"v2_ms_{tver}"

    # ── 분석 결과 헤더 ─────────────────────────────────────────────────────────
    st.subheader("분석 결과")
    col_text, col_star = st.columns([5, 1])
    with col_text:
        st.markdown(f"**{raw_input}**")
    with col_star:
        st.markdown(f"<span style='font-size:1.1rem'>{_stars(rating)}</span>",
                    unsafe_allow_html=True)

    if analysis.needs_more_context:
        st.error(
            f"⚠ 맥락이 부족합니다.\n\n"
            f"{analysis.context_question or '작품명, 장소, 활동 중 무엇인지 알려주면 더 정확하게 분석할 수 있습니다.'}\n\n"
            "태그를 직접 추가해서 저장할 수 있습니다."
        )
    else:
        if analysis.input_type == "sentence":
            st.caption("유형: 문장 입력")
            if analysis.contexts:
                st.markdown("맥락: " + _tag_chips(analysis.contexts, "#f0fdf4", "#166534"),
                            unsafe_allow_html=True)
        else:
            st.caption("유형: 단어 입력")
            parts = [analysis.category.large, analysis.category.medium, analysis.category.small]
            cat_str = " > ".join(p for p in parts if p)
            if cat_str:
                st.caption(f"분류: {cat_str}")

        # knowledge_confidence 안내
        if analysis.knowledge_confidence == "low":
            st.warning("⚠ 입력 대상 정보가 부족해 태그가 부정확할 수 있습니다. 직접 수정해주세요.")
        elif analysis.knowledge_confidence == "medium":
            st.caption("💡 일부 속성은 추정값입니다. 맞지 않는 태그는 해제하세요.")

    # ── 정보 태그 (source) ─────────────────────────────────────────────────────
    if analysis.source_tags:
        st.markdown(
            "📌 **정보** &nbsp;" + _tag_chips(analysis.source_tags, "#f3f4f6", "#374151",
                                               font_size="0.78rem"),
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 구체 속성 / 느낌 안내 ──────────────────────────────────────────────────
    if analysis.feature_tags:
        st.markdown(
            "🔵 **구체 속성** &nbsp;" + _tag_chips(analysis.feature_tags, "#eff6ff", "#1d4ed8"),
            unsafe_allow_html=True,
        )
    if analysis.preference_tags:
        st.markdown(
            "🟢 **느낌/해석** &nbsp;" + _tag_chips(analysis.preference_tags, "#f0fdf4", "#15803d"),
            unsafe_allow_html=True,
        )

    st.caption("아래에서 태그를 선택/해제하세요.")

    # ── 태그 multiselect ───────────────────────────────────────────────────────
    all_options = list(dict.fromkeys(
        analysis.recommended_tags
        + [t for t in st.session_state["v2_custom_tags"] if t not in analysis.recommended_tags]
    ))

    selected = st.multiselect(
        "태그 선택",
        options=all_options,
        label_visibility="collapsed",
        key=ms_key,
    )

    col_tag_input, col_tag_btn = st.columns([4, 1])
    with col_tag_input:
        custom_tag = st.text_input(
            "태그 직접 추가",
            placeholder="추가할 태그 입력",
            label_visibility="collapsed",
            key=f"v2_custom_input_{tver}",
        )
    with col_tag_btn:
        if st.button("추가", key=f"v2_add_tag_{tver}", use_container_width=True):
            tag = custom_tag.strip()
            if tag:
                current_selected = list(st.session_state.get(ms_key, []))
                if tag not in st.session_state["v2_custom_tags"]:
                    st.session_state["v2_custom_tags"] = st.session_state["v2_custom_tags"] + [tag]
                new_ver = tver + 1
                st.session_state["v2_tag_ver"] = new_ver
                new_selected = current_selected if tag in current_selected else current_selected + [tag]
                st.session_state[f"v2_ms_{new_ver}"] = new_selected
                st.rerun()

    # ── 예비 태그 (candidate) ──────────────────────────────────────────────────
    if analysis.candidate_tags:
        st.caption("📎 **예비 태그** — 클릭하면 선택 태그에 추가됩니다.")
        current_ms = st.session_state.get(ms_key, [])
        remaining = [t for t in analysis.candidate_tags if t not in current_ms]
        if remaining:
            cand_cols = st.columns(min(len(remaining), 4))
            for ci, ctag in enumerate(remaining):
                if cand_cols[ci % 4].button(ctag, key=f"cand_{tver}_{ci}", use_container_width=True):
                    new_sel = list(current_ms) + [ctag]
                    if ctag not in st.session_state["v2_custom_tags"]:
                        st.session_state["v2_custom_tags"] = st.session_state["v2_custom_tags"] + [ctag]
                    new_ver = tver + 1
                    st.session_state["v2_tag_ver"] = new_ver
                    st.session_state[f"v2_ms_{new_ver}"] = new_sel
                    st.rerun()

    st.divider()

    col_save, col_back = st.columns([3, 1])
    with col_save:
        save_btn = st.button("저장", type="primary", use_container_width=True)
    with col_back:
        if st.button("다시 입력", use_container_width=True):
            st.session_state["v2_stage"] = "input"
            st.session_state["v2_analysis"] = None
            st.rerun()

    if save_btn:
        if not selected:
            st.warning("태그를 하나 이상 선택하거나 추가해주세요.")
            return
        tag_type_map = _build_tag_type_map(analysis)
        # candidate 태그는 type="candidate"로 표시
        cand_set = set(analysis.candidate_tags)
        selected_tag_types = {
            t: ("candidate" if t in cand_set else tag_type_map.get(t, "custom"))
            for t in selected
        }
        with st.spinner("해석 생성 중..."):
            insight_result = generate_insight(
                raw_input=raw_input, rating=rating,
                input_type=analysis.input_type, category=analysis.category,
                contexts=analysis.contexts, selected_tags=selected,
                selected_tag_types=selected_tag_types, source_tags=analysis.source_tags,
                candidate_tags=analysis.candidate_tags,
            )
        _debug_insight(insight_result)
        entry_id = add_entry(
            raw_input=raw_input, rating=rating, analysis=analysis,
            selected_tags=selected, insight=insight_result,
            selected_tag_types=selected_tag_types,
        )
        _run_pattern_flow(entry_id)
        saved = get_entry(entry_id)
        st.session_state["v2_last_saved"] = saved
        st.session_state["v2_stage"] = "input"
        st.session_state["v2_analysis"] = None
        st.session_state["v2_rating"] = 3
        st.session_state["v2_form_ver"] += 1
        st.rerun()


# ── 저장 확인 카드 ────────────────────────────────────────────────────────────

def _render_last_saved() -> None:
    entry: TasteEntry = st.session_state["v2_last_saved"]
    with st.container(border=True):
        st.success("저장되었습니다.")
        st.markdown(f"**{entry.raw_input}** &nbsp; {_stars(entry.rating)}", unsafe_allow_html=True)
        if entry.source_tags:
            st.markdown(
                _tag_chips(entry.source_tags, "#f3f4f6", "#374151", font_size="0.78rem"),
                unsafe_allow_html=True,
            )
        _render_selected_tags(entry)
        if entry.one_line_insight:
            st.info(entry.one_line_insight)


# ── Entry 카드 ────────────────────────────────────────────────────────────────

def _render_entry_card(entry: TasteEntry) -> None:
    editing = st.session_state["v2_editing_id"] == entry.id

    with st.container(border=True):
        if editing:
            _render_edit_mode(entry)
        else:
            _render_view_mode(entry)


def _render_view_mode(entry: TasteEntry) -> None:
    col_main, col_actions = st.columns([8, 2])

    with col_main:
        st.markdown(
            f"**{entry.raw_input}** &nbsp; <span style='font-size:0.95rem'>{_stars(entry.rating)}</span>",
            unsafe_allow_html=True,
        )
        # 분류 or 맥락
        if entry.input_type == "sentence" and entry.contexts:
            st.caption("맥락: " + ", ".join(entry.contexts))
        elif entry.input_type == "word":
            parts = [entry.category.large, entry.category.medium, entry.category.small]
            cat_str = " > ".join(p for p in parts if p)
            if cat_str:
                st.caption(cat_str)

        if entry.source_tags:
            st.markdown(
                _tag_chips(entry.source_tags, "#f3f4f6", "#374151", font_size="0.78rem"),
                unsafe_allow_html=True,
            )
        _render_selected_tags(entry)

        if entry.one_line_insight:
            st.caption(f"💬 {entry.one_line_insight}")
        if entry.pattern_insight and entry.pattern_confidence in ("medium", "high"):
            st.markdown(
                f"<span style='font-size:0.78rem;color:#7c3aed;'>"
                f"🔁 누적 패턴: {entry.pattern_insight}</span>",
                unsafe_allow_html=True,
            )
        if entry.one_line_insight:
            _bc1, _bc2, _bc3 = st.columns([3, 3, 4])
            with _bc1:
                if st.button("↺ 해석 재생성", key=f"regen_{entry.id}",
                             use_container_width=True):
                    with st.spinner("재생성 중..."):
                        new_insight = generate_insight(
                            raw_input=entry.raw_input,
                            rating=entry.rating,
                            input_type=entry.input_type,
                            category=entry.category,
                            contexts=entry.contexts,
                            selected_tags=entry.selected_tags,
                            selected_tag_types=entry.selected_tag_types,
                            source_tags=entry.source_tags,
                            candidate_tags=entry.candidate_tags,
                        )
                    _debug_insight(new_insight)
                    update_insight_only(entry.id, new_insight)
                    _run_pattern_flow(entry.id)
                    st.rerun()
            with _bc2:
                if st.button("🔄 태그 재추천", key=f"retag_{entry.id}",
                             use_container_width=True):
                    with st.spinner("재분석 중..."):
                        new_analysis = analyze_input(entry.raw_input, entry.rating)
                    new_pending = dict(st.session_state["v2_edit_pending"])
                    new_pending[entry.id] = new_analysis
                    st.session_state["v2_edit_pending"] = new_pending
                    st.session_state["v2_edit_custom"] = []
                    st.session_state["v2_editing_id"] = entry.id
                    new_ever = st.session_state["v2_edit_tag_ver"] + 1
                    st.session_state["v2_edit_tag_ver"] = new_ever
                    ekey = f"v2_edit_ms_{entry.id}_{new_ever}"
                    st.session_state[ekey] = list(new_analysis.recommended_tags)
                    st.rerun()

        if entry.needs_more_context:
            st.markdown(
                "<span style='color:#dc2626;font-size:0.8rem'>⚠ 맥락 정보 부족</span>",
                unsafe_allow_html=True,
            )

        date_str = entry.created_at[:10] if entry.created_at else ""
        if date_str:
            st.caption(f"기록일: {date_str}")

    with col_actions:
        if st.button("수정", key=f"edit_{entry.id}", use_container_width=True):
            st.session_state["v2_editing_id"] = entry.id
            st.session_state["v2_edit_custom"] = []
            st.session_state["v2_edit_tag_ver"] += 1
            # 이전 재분석 결과 초기화
            pending = dict(st.session_state["v2_edit_pending"])
            pending.pop(entry.id, None)
            st.session_state["v2_edit_pending"] = pending
            ekey = f"v2_edit_ms_{entry.id}_{st.session_state['v2_edit_tag_ver']}"
            st.session_state[ekey] = list(entry.selected_tags)
            st.rerun()
        if st.button("삭제", key=f"del_{entry.id}", use_container_width=True):
            delete_entry(entry.id)
            if st.session_state["v2_editing_id"] == entry.id:
                st.session_state["v2_editing_id"] = None
            if (
                st.session_state["v2_last_saved"]
                and st.session_state["v2_last_saved"].id == entry.id
            ):
                st.session_state["v2_last_saved"] = None
            st.rerun()


def _render_edit_mode(entry: TasteEntry) -> None:
    st.caption(f"✏️ 수정 중 (id: {entry.id})")

    new_raw = st.text_area(
        "내용",
        value=entry.raw_input,
        key=f"v2_edit_raw_{entry.id}",
    )

    new_rating = st.radio(
        "별점",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: _stars(x),
        horizontal=True,
        index=entry.rating - 1,
        key=f"v2_edit_rating_{entry.id}",
    )

    # 재분석 결과가 있으면 그 추천 태그를 옵션 기준으로 사용
    pending: AnalysisResult | None = st.session_state["v2_edit_pending"].get(entry.id)
    base_recommended = pending.recommended_tags if pending else entry.recommended_tags

    ever = st.session_state["v2_edit_tag_ver"]
    ekey = f"v2_edit_ms_{entry.id}_{ever}"

    all_options = list(dict.fromkeys(
        base_recommended
        + [t for t in st.session_state["v2_edit_custom"] if t not in base_recommended]
        + [t for t in entry.selected_tags if t not in base_recommended]
    ))

    # "태그 다시 추천받기" 버튼: 현재 입력 기준으로 재분석, 옵션 갱신, 기존 선택 유지
    if st.button("↺ 태그 다시 추천받기", key=f"reanalyze_{entry.id}_{ever}"):
        raw = st.session_state.get(f"v2_edit_raw_{entry.id}", entry.raw_input).strip()
        rating = st.session_state.get(f"v2_edit_rating_{entry.id}", entry.rating)
        if not raw:
            st.warning("내용을 먼저 입력해주세요.")
        else:
            with st.spinner("재분석 중..."):
                new_analysis = analyze_input(raw, rating)
            new_pending = dict(st.session_state["v2_edit_pending"])
            new_pending[entry.id] = new_analysis
            st.session_state["v2_edit_pending"] = new_pending
            st.session_state["v2_edit_custom"] = []
            new_ever = ever + 1
            st.session_state["v2_edit_tag_ver"] = new_ever
            # 선택값을 새 추천 태그로 교체 (기존 선택은 옵션에 남겨 추가 선택 가능)
            st.session_state[f"v2_edit_ms_{entry.id}_{new_ever}"] = list(new_analysis.recommended_tags)
            st.rerun()

    selected = st.multiselect(
        "태그",
        options=all_options,
        label_visibility="collapsed",
        key=ekey,
    )

    col_tag_in, col_tag_add = st.columns([4, 1])
    with col_tag_in:
        custom_tag = st.text_input(
            "태그 추가",
            placeholder="추가할 태그",
            label_visibility="collapsed",
            key=f"v2_edit_custom_input_{entry.id}_{ever}",
        )
    with col_tag_add:
        if st.button("추가", key=f"v2_edit_add_{entry.id}_{ever}", use_container_width=True):
            tag = custom_tag.strip()
            if tag:
                current_selected = list(st.session_state.get(ekey, []))
                if tag not in st.session_state["v2_edit_custom"]:
                    st.session_state["v2_edit_custom"] = st.session_state["v2_edit_custom"] + [tag]
                new_ever = ever + 1
                st.session_state["v2_edit_tag_ver"] = new_ever
                new_selected = current_selected if tag in current_selected else current_selected + [tag]
                st.session_state[f"v2_edit_ms_{entry.id}_{new_ever}"] = new_selected
                st.rerun()

    _warn = st.empty()
    col_save, col_cancel = st.columns([3, 1])
    with col_save:
        if st.button("저장", key=f"save_edit_{entry.id}", type="primary",
                     use_container_width=True):
            if not new_raw.strip():
                _warn.warning("내용을 입력해주세요.")
                return
            if not selected:
                _warn.warning("태그를 선택해주세요.")
                return

            final_analysis = pending if pending else AnalysisResult(
                input_type=entry.input_type,
                category=entry.category,
                contexts=entry.contexts,
                source_tags=entry.source_tags,
                feature_tags=entry.feature_tags,
                preference_tags=entry.preference_tags,
                recommended_tags=entry.recommended_tags,
                needs_more_context=entry.needs_more_context,
                context_question=entry.context_question,
                knowledge_confidence=entry.knowledge_confidence,
            )
            tag_type_map = _build_tag_type_map(final_analysis)
            selected_tag_types = {t: tag_type_map.get(t, "custom") for t in selected}

            with st.spinner("해석 재생성 중..."):
                insight_result = generate_insight(
                    raw_input=new_raw.strip(),
                    rating=new_rating,
                    input_type=final_analysis.input_type,
                    category=final_analysis.category,
                    contexts=final_analysis.contexts,
                    selected_tags=selected,
                    selected_tag_types=selected_tag_types,
                    source_tags=final_analysis.source_tags,
                    candidate_tags=final_analysis.candidate_tags,
                )
            update_full_entry(
                entry.id, new_raw.strip(), new_rating, final_analysis,
                selected, insight_result, selected_tag_types,
            )
            new_pending = dict(st.session_state["v2_edit_pending"])
            new_pending.pop(entry.id, None)
            st.session_state["v2_edit_pending"] = new_pending
            st.session_state["v2_editing_id"] = None
            st.rerun()
    with col_cancel:
        if st.button("취소", key=f"cancel_edit_{entry.id}", use_container_width=True):
            new_pending = dict(st.session_state["v2_edit_pending"])
            new_pending.pop(entry.id, None)
            st.session_state["v2_edit_pending"] = new_pending
            st.session_state["v2_editing_id"] = None
            st.rerun()


# ── 메인 레이아웃 ─────────────────────────────────────────────────────────────

st.title("TasteMap v2")
st.caption("입력 + 별점 → 태그 추천 → 선택 → 한줄 해석")

stage = st.session_state["v2_stage"]

if stage == "input":
    _render_input_form()
elif stage == "tagging":
    _render_tagging()

if stage == "input" and st.session_state["v2_last_saved"]:
    _render_last_saved()

st.divider()
st.subheader("기록 목록")

entries = list_entries()

if not entries:
    st.info("아직 기록이 없습니다. 위에서 첫 경험을 기록해보세요.")
else:
    st.caption(f"전체 {len(entries)}개")
    for entry in entries:
        _render_entry_card(entry)
