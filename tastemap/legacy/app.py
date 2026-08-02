import json
from collections import defaultdict
from datetime import date as _date, datetime, timezone

import streamlit as st

from tastemap.legacy.ai_classifier import classify_preference
from tastemap.legacy.analysis import (
    build_mindmap_data,
    build_preference_profile,
    generate_profile_sentences,
    get_average_rating_by_category,
    get_average_rating_by_tag,
    get_related_tags,
    get_searchable_tags,
    get_time_perspective_summary,
    get_top_context_tags,
    get_top_object_tags,
    get_top_place_tags,
    get_top_sensory_tags,
    get_top_social_tags,
    get_top_tags,
    get_top_tags_by_rating,
)
from tastemap.legacy.db import (
    add_entry, delete_entry, export_entries_for_csv, export_entries_for_json,
    get_db_backup_bytes, init_db, list_entries, update_entry,
)
from tastemap.legacy.visualization import render_mindmap, render_mindmap_graphviz

init_db()

# ── 상수 ──────────────────────────────────────────────────────────────────────
_SOURCE_LABELS: dict[str, str] = {
    "AI": "AI",
    "RULE_BASED_FALLBACK": "규칙 기반 (AI 실패)",
    "RULE_BASED_NO_API_KEY": "규칙 기반",
    "RULE_BASED_INSUFFICIENT_QUOTA": "규칙 기반 (quota 초과)",
}

# ── session_state 초기화 ──────────────────────────────────────────────────────
_SS_DEFAULTS: dict = {
    "entry_form_version":          0,
    "entry_input_count":           1,
    "last_classification":         None,
    "last_saved_entry_id":         None,
    "last_saved_raw_text":         None,
    "last_saved_rating":           None,
    "last_saved_entries":          [],
    "show_quota_warning":          False,
    "editing_entry_id":            None,
    "edit_form_version":           0,
    "last_updated_entry_id":       None,
    "last_updated_classification": None,
    "last_updated_raw_text":       None,
    "last_updated_rating":         None,
    "related_tag_query":           "",      # 연관 태그 검색어
    "related_query_pending":       "",      # 추천 버튼 클릭 시 pending 값
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── UI 헬퍼: 별점 표시 ────────────────────────────────────────────────────────
def render_stars(rating: int) -> str:
    return "★" * rating + "☆" * (5 - rating)


def _stars_html(rating: int) -> str:
    stars = render_stars(rating)
    return (
        f"<span style='white-space:nowrap;font-size:1.1rem;letter-spacing:1px'>"
        f"{stars}</span>"
    )


# ── UI 헬퍼: 분류 결과 expander ───────────────────────────────────────────────
def _render_classification_result(
    title: str,
    entry_id: int,
    raw_text: str,
    rating: int,
    classification,
) -> None:
    st.success(f"{title} (id: {entry_id})")
    with st.expander("분류 결과 보기", expanded=True):
        if raw_text:
            st.markdown(
                f"입력: {raw_text} &nbsp;|&nbsp; 평점: {_stars_html(rating)}",
                unsafe_allow_html=True,
            )
        col1, col2, col3 = st.columns(3)
        col1.metric("주 카테고리", classification.primary_category)
        col2.metric("맥락 카테고리", ", ".join(classification.context_categories) or "없음")
        col3.metric("신뢰도", f"{classification.confidence:.0%}")
        st.write("**서브카테고리:**", ", ".join(classification.subcategories) or "없음")
        st.write("**태그:**", ", ".join(classification.tags) or "없음")
        st.write("**선호 신호:**", classification.preference_signal)
        dt1, dt2, dt3, dt4, dt5 = st.columns(5)
        dt1.caption("대상: " + (", ".join(classification.object_tags) or "—"))
        dt2.caption("감각: " + (", ".join(classification.sensory_tags) or "—"))
        dt3.caption("상황: " + (", ".join(classification.context_tags) or "—"))
        dt4.caption("관계: " + (", ".join(classification.social_tags) or "—"))
        dt5.caption("장소: " + (", ".join(classification.place_tags) or "—"))
        st.caption(
            f"분류 방식: {_SOURCE_LABELS.get(classification.classifier_source, classification.classifier_source)}"
        )
    if classification.primary_category == "미분류":
        st.warning(
            "분류가 애매합니다. 대상이 음식/게임/직업/운동/물건 중 무엇인지 드러나도록 "
            "내용을 조금 더 구체적으로 수정한 뒤 재분류해보세요.  \n"
            "예) '팩토리오' → '팩토리오 게임', '향수 좋음' → '애프터눈 딜라이트 향수 좋음'"
        )


# ── UI 헬퍼: 다중 저장 결과 표시 ─────────────────────────────────────────────
def _render_saved_entries_result(saved_entries: list) -> None:
    if not saved_entries:
        return
    n = len(saved_entries)
    if n == 1:
        item = saved_entries[0]
        if st.session_state.get("show_quota_warning"):
            st.warning("OpenAI API quota is unavailable. Using rule-based classifier.")
        _render_classification_result(
            "저장되었습니다.",
            item["entry_id"],
            item["raw_text"],
            item["rating"],
            item["classification"],
        )
        return

    any_unclassified = any(
        item["classification"].primary_category == "미분류" for item in saved_entries
    )
    st.success(f"{n}개 기록이 저장되었습니다.")
    if st.session_state.get("show_quota_warning"):
        st.warning("OpenAI API quota is unavailable. Using rule-based classifier.")
    with st.expander("저장된 분류 결과 보기", expanded=True):
        for item in saved_entries:
            cls = item["classification"]
            badge = ""
            if cls.primary_category == "미분류":
                badge = " <span style='color:#d62728;font-weight:700;'>⚠ 미분류</span>"
            st.markdown(
                f"**{item['raw_text']}** — {render_stars(item['rating'])} "
                f"| <span style='font-size:0.9rem;color:#555;'>{cls.primary_category}</span>"
                + badge,
                unsafe_allow_html=True,
            )
            if cls.tags:
                st.caption("태그: " + " · ".join(cls.tags[:8]))
            st.divider()
    if any_unclassified:
        st.warning(
            "미분류 항목이 있습니다. 해당 기록을 수정/재분류하면 정확도가 올라갑니다."
        )


# ── UI 헬퍼: 재분류 공통 로직 ────────────────────────────────────────────────
def _do_reclassify_entries(target_entries: list) -> tuple[int, list[str]]:
    """target_entries를 현재 classifier로 재분류하고 DB에 저장한다.
    event_date / event_date_precision / time_perspective는 기존 값을 유지한다.
    반환: (success_count, fail_messages)
    """
    pb  = st.progress(0)
    ok  = 0
    fail: list[str] = []
    total = len(target_entries)
    for i, e in enumerate(target_entries):
        if e.id is None:
            continue
        try:
            new_cls = classify_preference(e.raw_text, e.rating)
            update_entry(
                e.id, e.raw_text, e.rating, new_cls,
                event_date=e.event_date,
                event_date_precision=e.event_date_precision,
                time_perspective=e.time_perspective,
            )
            ok += 1
        except Exception as exc:
            fail.append(f"id={e.id}: {exc}")
        pb.progress((i + 1) / total)
    return ok, fail


# ── UI 헬퍼: 날짜 유틸 ───────────────────────────────────────────────────────
def infer_time_perspective(event_date: str | None) -> str:
    if not event_date:
        return "unspecified"
    try:
        ed = _date.fromisoformat(event_date)
        today = _date.today()
        if ed < today:
            return "past"
        if ed == today:
            return "present"
        return "future"
    except ValueError:
        return "unspecified"


def format_event_date(event_date: str | None, precision: str) -> str:
    if not event_date:
        return ""
    try:
        if precision == "day":
            return event_date[:10]
        if precision == "month":
            return event_date[:7]
        if precision == "year":
            return event_date[:4] + "년"
        return ""
    except Exception:
        return ""


def get_time_perspective_label(time_perspective: str) -> str:
    return {"past": "회상일", "present": "경험일", "future": "기대일"}.get(
        time_perspective, ""
    )


def _render_date_input_ui(ver: int) -> tuple[str | None, str, str]:
    """날짜 expander UI를 렌더링하고 (event_date, precision, perspective) 반환."""
    _event_date_out       = None
    _precision_out        = "none"
    _perspective_out      = "unspecified"

    with st.expander("+ 회상/기대 날짜 붙이기", expanded=False):
        _use = st.checkbox(
            "이 기록에 별도 날짜를 지정합니다.",
            key=f"use_ed_{ver}",
        )
        if _use:
            _prec_label = st.radio(
                "날짜 정확도",
                ["정확한 날짜", "월까지만", "연도만"],
                horizontal=True,
                key=f"ed_prec_{ver}",
            )
            _today = _date.today()
            if _prec_label == "정확한 날짜":
                _sel = st.date_input("날짜", value=_today, key=f"ed_day_{ver}")
                _event_date_out  = _sel.isoformat()
                _precision_out   = "day"
            elif _prec_label == "월까지만":
                _yc, _mc = st.columns(2)
                _yr = _yc.selectbox(
                    "연도", range(2000, 2036),
                    index=_today.year - 2000, key=f"ed_yr_{ver}"
                )
                _mo = _mc.selectbox(
                    "월", range(1, 13),
                    index=_today.month - 1, key=f"ed_mo_{ver}"
                )
                _event_date_out  = f"{_yr:04d}-{_mo:02d}-01"
                _precision_out   = "month"
            else:  # 연도만
                _yr = st.selectbox(
                    "연도", range(2000, 2036),
                    index=_today.year - 2000, key=f"ed_yr2_{ver}"
                )
                _event_date_out  = f"{_yr:04d}-01-01"
                _precision_out   = "year"

            _perspective_out = infer_time_perspective(_event_date_out)
            _lbl = get_time_perspective_label(_perspective_out)
            _fmt = format_event_date(_event_date_out, _precision_out)
            if _lbl and _fmt:
                st.caption(f"{_lbl}로 저장됩니다: {_fmt}")

    return _event_date_out, _precision_out, _perspective_out


# ── UI 헬퍼: 분류 검토 필요 여부 판단 ───────────────────────────────────────
def _needs_classification_review(entry_or_cls) -> bool:
    """미분류 / low confidence / 빈 object 활동 등을 검토 필요로 판단한다."""
    primary    = getattr(entry_or_cls, "primary_category", "") or ""
    confidence = getattr(entry_or_cls, "confidence", 1.0) or 1.0
    obj_tags   = getattr(entry_or_cls, "object_tags", []) or []
    tags       = getattr(entry_or_cls, "tags", []) or []
    return (
        primary in ("미분류", "", None)
        or confidence < 0.45
        or (primary in ("활동", "상황") and not obj_tags and not tags)
    )


# ── UI 헬퍼: entry 카드 렌더링 ────────────────────────────────────────────────
def render_entry_card(entry) -> None:
    """단일 entry를 카드로 렌더링한다. 수정/삭제/재분류/세부 태그 기능 포함."""
    editing = st.session_state["editing_entry_id"] == entry.id

    with st.container(border=True):
        if editing:
            # ── 수정 모드 ──────────────────────────────────────────────────────
            st.caption(f"✏️ 수정 중 (id: {entry.id})")

            edit_text_key   = f"edit_text_{entry.id}_{st.session_state['edit_form_version']}"
            edit_rating_key = f"edit_rating_{entry.id}_{st.session_state['edit_form_version']}"

            edited_text = st.text_area(
                "내용",
                value=entry.raw_text,
                key=edit_text_key,
            )
            selected_rating = st.radio(
                "평점",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: "★" * x + "☆" * (5 - x),
                horizontal=True,
                index=entry.rating - 1,
                key=edit_rating_key,
            )

            # 날짜 정보 수정 UI
            _eev = st.session_state["edit_form_version"]
            _ed_out, _eprec_out, _etp_out = entry.event_date, entry.event_date_precision or "none", entry.time_perspective or "unspecified"
            with st.expander("날짜 정보 수정", expanded=bool(entry.event_date)):
                _use_ed_edit = st.checkbox(
                    "별도 날짜 지정", value=bool(entry.event_date),
                    key=f"edit_use_ed_{entry.id}_{_eev}",
                )
                if _use_ed_edit:
                    _prec_map_inv = {"day": 0, "month": 1, "year": 2, "none": 0}
                    _eprec_idx = _prec_map_inv.get(entry.event_date_precision or "none", 0)
                    _eprec_label = st.radio(
                        "날짜 정확도", ["정확한 날짜", "월까지만", "연도만"],
                        horizontal=True, index=_eprec_idx,
                        key=f"edit_prec_{entry.id}_{_eev}",
                    )
                    _today_e = _date.today()
                    if _eprec_label == "정확한 날짜":
                        try:
                            _def_d = _date.fromisoformat(entry.event_date) if entry.event_date else _today_e
                        except ValueError:
                            _def_d = _today_e
                        _esel = st.date_input("날짜", value=_def_d, key=f"edit_day_{entry.id}_{_eev}")
                        _ed_out, _eprec_out = _esel.isoformat(), "day"
                    elif _eprec_label == "월까지만":
                        _eyc, _emc = st.columns(2)
                        try:
                            _def_yr = int(entry.event_date[:4]) if entry.event_date else _today_e.year
                            _def_mo = int(entry.event_date[5:7]) if entry.event_date else _today_e.month
                        except (ValueError, TypeError, IndexError):
                            _def_yr, _def_mo = _today_e.year, _today_e.month
                        _eyr = _eyc.selectbox("연도", range(2000, 2036), index=_def_yr - 2000, key=f"edit_yr_{entry.id}_{_eev}")
                        _emo = _emc.selectbox("월",   range(1, 13),     index=_def_mo - 1,    key=f"edit_mo_{entry.id}_{_eev}")
                        _ed_out, _eprec_out = f"{_eyr:04d}-{_emo:02d}-01", "month"
                    else:
                        try:
                            _def_yr = int(entry.event_date[:4]) if entry.event_date else _today_e.year
                        except (ValueError, TypeError):
                            _def_yr = _today_e.year
                        _eyr = st.selectbox("연도", range(2000, 2036), index=_def_yr - 2000, key=f"edit_yr2_{entry.id}_{_eev}")
                        _ed_out, _eprec_out = f"{_eyr:04d}-01-01", "year"
                    _etp_out = infer_time_perspective(_ed_out)
                    _elbl = get_time_perspective_label(_etp_out)
                    _efmt = format_event_date(_ed_out, _eprec_out)
                    if _elbl and _efmt:
                        st.caption(f"{_elbl}: {_efmt}")
                else:
                    _ed_out, _eprec_out, _etp_out = None, "none", "unspecified"

            _edit_warning = st.empty()
            # 재분류 후 저장 텍스트가 길어 3:1 비율로 충분한 폭 확보
            col_save, col_cancel = st.columns([3, 1])

            if col_save.button("재분류 후 저장", key=f"save_edit_{entry.id}",
                               use_container_width=True):
                if not edited_text.strip():
                    _edit_warning.warning("내용을 입력해주세요.")
                else:
                    new_cls = classify_preference(edited_text.strip(), selected_rating)
                    update_entry(
                        entry.id, edited_text.strip(), selected_rating, new_cls,
                        event_date=_ed_out,
                        event_date_precision=_eprec_out,
                        time_perspective=_etp_out,
                    )
                    st.session_state["editing_entry_id"]            = None
                    st.session_state["last_updated_entry_id"]       = entry.id
                    st.session_state["last_updated_classification"] = new_cls
                    st.session_state["last_updated_raw_text"]       = edited_text.strip()
                    st.session_state["last_updated_rating"]         = selected_rating
                    st.session_state["last_classification"]         = None
                    st.session_state["edit_form_version"]          += 1
                    st.rerun()

            if col_cancel.button("취소", key=f"cancel_edit_{entry.id}",
                                 use_container_width=True):
                st.session_state["editing_entry_id"] = None
                st.session_state["edit_form_version"] += 1
                st.rerun()

        else:
            # ── 일반 표시 모드 ─────────────────────────────────────────────────
            # 3영역: 텍스트/메타 | 별점/카테고리 | 수정·삭제(세로 버튼)
            date_str = entry.created_at[:10] if entry.created_at else ""
            col_main, col_info, col_actions = st.columns([6.8, 2.0, 1.2])

            with col_main:
                st.write(entry.raw_text)
                meta_parts = []
                if entry.tags:
                    meta_parts.append("태그: " + " · ".join(entry.tags[:8]))
                if date_str:
                    meta_parts.append(f"기록일: {date_str}")
                if meta_parts:
                    st.caption("  |  ".join(meta_parts))

            with col_info:
                st.markdown(_stars_html(entry.rating), unsafe_allow_html=True)
                if _needs_classification_review(entry):
                    st.markdown(
                        "<span style='font-size:0.82rem;color:#d62728;"
                        "font-weight:700;white-space:nowrap;'>⚠ 확인 필요</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='font-size:0.82rem;color:#555;"
                        f"white-space:nowrap;'>{entry.primary_category}</span>",
                        unsafe_allow_html=True,
                    )
                if entry.context_categories:
                    st.caption(", ".join(entry.context_categories))
                # 날짜 표시 (지정된 경우만)
                if entry.event_date:
                    _tp_lbl = get_time_perspective_label(entry.time_perspective)
                    _tp_fmt = format_event_date(entry.event_date, entry.event_date_precision)
                    if _tp_lbl and _tp_fmt:
                        st.caption(f"{_tp_lbl}: {_tp_fmt}")

            with col_actions:
                if st.button("수정", key=f"edit_{entry.id}",
                             use_container_width=True):
                    st.session_state["editing_entry_id"] = entry.id
                    st.rerun()
                if st.button("삭제", key=f"del_{entry.id}",
                             use_container_width=True):
                    delete_entry(entry.id)
                    if st.session_state["editing_entry_id"] == entry.id:
                        st.session_state["editing_entry_id"] = None
                    st.rerun()

            if _needs_classification_review(entry):
                st.caption("내용을 조금 더 구체적으로 수정한 뒤 재분류하면 정확도가 올라갑니다.")

            has_detail = any([
                entry.object_tags, entry.sensory_tags,
                entry.context_tags, entry.social_tags, entry.place_tags,
            ])
            if has_detail:
                with st.expander("세부 태그 보기"):
                    d1, d2, d3, d4, d5 = st.columns(5)
                    with d1:
                        st.caption("**대상**")
                        for t in entry.object_tags:
                            st.write(t)
                    with d2:
                        st.caption("**감각**")
                        for t in entry.sensory_tags:
                            st.write(t)
                    with d3:
                        st.caption("**상황**")
                        for t in entry.context_tags:
                            st.write(t)
                    with d4:
                        st.caption("**관계**")
                        for t in entry.social_tags:
                            st.write(t)
                    with d5:
                        st.caption("**장소**")
                        for t in entry.place_tags:
                            st.write(t)


# ─────────────────────────────────────────────────────────────────────────────
st.title("TasteMap")
st.caption("좋았던 것, 싫었던 것을 기록하세요.")

# ── 새 기록 입력 영역 ──────────────────────────────────────────────────────────
_ver   = st.session_state["entry_form_version"]
_count = st.session_state["entry_input_count"]


def _input_key(idx: int) -> str:
    return f"raw_text_input_{idx}_{_ver}"


# 입력칸 렌더링
for _idx in range(_count):
    _tk = _input_key(_idx)
    if _idx == 0:
        st.text_area(
            "무엇을 경험했나요?",
            key=_tk,
            placeholder="예: 어제 먹은 곱창이 너무 맛있었다",
        )
    else:
        _tc, _td = st.columns([11, 1])
        with _tc:
            st.text_area(
                f"항목 {_idx + 1}",
                key=_tk,
                placeholder=f"항목 {_idx + 1}",
                label_visibility="collapsed",
            )
        with _td:
            st.markdown("<div style='padding-top:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("✕", key=f"del_input_{_idx}_{_ver}", help="이 입력칸 삭제"):
                # 삭제된 칸을 제외하고 나머지를 새 키로 복사
                _remaining = [
                    st.session_state.get(_input_key(_i), "")
                    for _i in range(_count) if _i != _idx
                ]
                _new_ver = _ver + 1
                _new_count = max(1, _count - 1)
                st.session_state["entry_input_count"] = _new_count
                st.session_state["entry_form_version"] = _new_ver
                for _ri, _rv in enumerate(_remaining):
                    if _rv:
                        st.session_state[f"raw_text_input_{_ri}_{_new_ver}"] = _rv
                st.rerun()

# 쉼표 감지 — 입력칸이 1개이고 쉼표 2개 이상일 때 안내
_first_val = st.session_state.get(_input_key(0), "")
if _count == 1 and _first_val.count(",") >= 2:
    st.caption(
        "여러 항목이 한 문장에 들어간 것 같습니다. "
        "+ 항목 추가로 나누어 입력하면 분류 정확도가 올라갑니다."
    )
    if st.button("쉼표 기준으로 입력칸 나누기", key=f"split_comma_{_ver}"):
        _split_items = [s.strip() for s in _first_val.split(",") if s.strip()]
        if len(_split_items) >= 2:
            _new_ver = _ver + 1
            st.session_state["entry_input_count"]  = len(_split_items)
            st.session_state["entry_form_version"] = _new_ver
            for _si, _sv in enumerate(_split_items):
                st.session_state[f"raw_text_input_{_si}_{_new_ver}"] = _sv
            st.rerun()

# "+ 항목 추가" 버튼
if st.button("＋ 항목 추가", key=f"add_input_{_ver}"):
    st.session_state["entry_input_count"] += 1
    st.rerun()

# 날짜 옵션 (접힌 expander — 모든 입력칸에 공통 적용)
_event_date, _event_date_precision, _time_perspective = _render_date_input_ui(_ver)

st.caption("별을 클릭하면 현재 입력된 모든 항목이 저장됩니다.")
_input_warning = st.empty()

star_cols = st.columns(5)
for _star_i, _star_col in enumerate(star_cols, start=1):
    _star_label = "★" * _star_i + "☆" * (5 - _star_i)
    if _star_col.button(_star_label, key=f"rating_star_{_star_i}"):
        # 현재 입력된 모든 칸 수집
        _all_texts = [
            st.session_state.get(_input_key(_idx), "").strip()
            for _idx in range(_count)
        ]
        _non_empty = [t for t in _all_texts if t]

        if not _non_empty:
            _input_warning.warning("내용을 입력해주세요.")
        else:
            _saved = []
            _any_quota = False
            for _text in _non_empty:
                _cls = classify_preference(_text, _star_i)
                _eid = add_entry(
                    _text, _star_i, _cls,
                    event_date=_event_date,
                    event_date_precision=_event_date_precision,
                    time_perspective=_time_perspective,
                )
                _saved.append({
                    "entry_id":             _eid,
                    "raw_text":             _text,
                    "rating":               _star_i,
                    "classification":       _cls,
                    "event_date":           _event_date,
                    "event_date_precision": _event_date_precision,
                    "time_perspective":     _time_perspective,
                })
                if _cls.classifier_source == "RULE_BASED_INSUFFICIENT_QUOTA":
                    _any_quota = True

            st.session_state["last_saved_entries"]          = _saved
            st.session_state["show_quota_warning"]          = _any_quota
            st.session_state["last_updated_classification"] = None
            # 단일 저장 호환성 유지
            if len(_saved) == 1:
                st.session_state["last_saved_entry_id"]  = _saved[0]["entry_id"]
                st.session_state["last_saved_raw_text"]  = _saved[0]["raw_text"]
                st.session_state["last_saved_rating"]    = _saved[0]["rating"]
                st.session_state["last_classification"]  = _saved[0]["classification"]
            else:
                st.session_state["last_classification"] = None

            # 입력칸 초기화
            st.session_state["entry_input_count"]  = 1
            st.session_state["entry_form_version"] += 1
            st.rerun()

# ── 마지막 저장 결과 ──────────────────────────────────────────────────────────
if st.session_state["last_saved_entries"]:
    _render_saved_entries_result(st.session_state["last_saved_entries"])

# ── 최근 수정 결과 ────────────────────────────────────────────────────────────
if st.session_state["last_updated_classification"] is not None:
    _render_classification_result(
        "수정 및 재분류 완료.",
        st.session_state["last_updated_entry_id"],
        st.session_state["last_updated_raw_text"] or "",
        st.session_state["last_updated_rating"] or 0,
        st.session_state["last_updated_classification"],
    )

# 분류 관리와 기록 목록 양쪽에서 사용하므로 여기서 한 번만 로드
entries = list_entries()

# ── 분류 관리 ─────────────────────────────────────────────────────────────────
st.divider()
with st.expander("분류 관리", expanded=False):
    _review_mgmt = [e for e in entries if _needs_classification_review(e)]

    _cm1, _cm2 = st.columns(2)
    _cm1.metric("전체 기록", len(entries))
    _cm2.metric("분류 확인 필요", len(_review_mgmt))

    st.caption(
        "현재 분류 규칙으로 기존 기록을 다시 분류합니다.  \n"
        "이 작업은 기존 카테고리와 태그를 현재 분류 규칙 기준으로 덮어씁니다.  \n"
        "AI 사용 환경이면 각 기록마다 API가 호출되어 비용이 발생할 수 있습니다."
    )

    # ── 분류 확인 필요 기록만 재분류 ─────────────────────────────────────────
    if _review_mgmt:
        if st.button(f"분류 확인 필요 {len(_review_mgmt)}개 재분류"):
            _ok, _fail = _do_reclassify_entries(_review_mgmt)
            st.success(f"분류 확인 필요 {_ok}개 재분류 완료")
            if _fail:
                st.warning(f"{len(_fail)}개 실패")
                for _m in _fail:
                    st.caption(_m)
            st.session_state["last_classification"] = None
            st.session_state["last_updated_classification"] = None
            st.rerun()

    st.divider()

    # ── 전체 재분류 ──────────────────────────────────────────────────────────
    confirm_reclassify = st.checkbox("전체 기록을 재분류하겠습니다.")
    if st.button("전체 기록 재분류 및 저장", disabled=not confirm_reclassify):
        if not entries:
            st.info("재분류할 기록이 없습니다.")
        else:
            _ok2, _fail2 = _do_reclassify_entries(entries)
            st.success(f"{_ok2}개 기록을 재분류했습니다.")
            if _fail2:
                st.warning(f"{len(_fail2)}개 실패:")
                for msg in _fail2:
                    st.caption(msg)
            st.session_state["last_classification"] = None
            st.session_state["last_updated_classification"] = None
            st.rerun()

# ── 데이터 백업/내보내기 ──────────────────────────────────────────────────────
st.divider()
with st.expander("데이터 백업/내보내기", expanded=False):
    st.caption("현재 기록을 CSV, JSON, SQLite DB 파일로 내려받을 수 있습니다.")

    if not entries:
        st.info("아직 내보낼 기록이 없습니다.")
        _export_disabled = True
    else:
        st.caption(f"총 {len(entries)}개의 기록을 내보냅니다.")
        _export_disabled = False

    _ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    _exp_c1, _exp_c2, _exp_c3 = st.columns(3)

    # ── JSON 다운로드 ──────────────────────────────────────────────────────────
    with _exp_c1:
        if not _export_disabled:
            _json_data = json.dumps(
                export_entries_for_json(), ensure_ascii=False, indent=2
            ).encode("utf-8")
            st.download_button(
                label="JSON 다운로드",
                data=_json_data,
                file_name=f"tastemap_entries_{_ts}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.button("JSON 다운로드", disabled=True, use_container_width=True)

    # ── CSV 다운로드 ───────────────────────────────────────────────────────────
    with _exp_c2:
        if not _export_disabled:
            try:
                import pandas as _pd
                _csv_rows = export_entries_for_csv()
                _csv_data = _pd.DataFrame(_csv_rows).to_csv(
                    index=False
                ).encode("utf-8-sig")
                st.download_button(
                    label="CSV 다운로드",
                    data=_csv_data,
                    file_name=f"tastemap_entries_{_ts}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except ImportError:
                st.warning("pandas가 없어 CSV 내보내기를 사용할 수 없습니다.")
        else:
            st.button("CSV 다운로드", disabled=True, use_container_width=True)

    # ── SQLite DB 백업 다운로드 ───────────────────────────────────────────────
    with _exp_c3:
        _db_bytes = get_db_backup_bytes()
        if _db_bytes:
            st.download_button(
                label="DB 백업 다운로드",
                data=_db_bytes,
                file_name=f"tastemap_backup_{_ts}.db",
                mime="application/octet-stream",
                use_container_width=True,
            )
        else:
            st.button("DB 백업 다운로드", disabled=True, use_container_width=True)
            st.caption("DB 파일을 찾을 수 없습니다.")

# ── 기록 목록 ─────────────────────────────────────────────────────────────────
st.divider()
st.subheader("기록 목록")
# (entries already loaded above)

if not entries:
    st.info("아직 기록이 없습니다.")
else:
    # 분류 확인 필요 목록 — 모든 view mode에서 공통 사용
    review_entries = [e for e in entries if _needs_classification_review(e)]

    view_mode = st.radio(
        "표시 방식",
        ["최근 10개", "분류 확인 필요", "회상", "기대", "날짜별 보기", "전체 보기"],
        horizontal=True,
        index=0,
        key="entry_view_mode",
    )

    past_entries   = [e for e in entries if e.time_perspective == "past"]
    future_entries = [e for e in entries if e.time_perspective == "future"]

    if view_mode == "최근 10개":
        display_count = min(10, len(entries))
        _extra_parts = []
        if review_entries: _extra_parts.append(f"분류 확인 필요: {len(review_entries)}개")
        if past_entries:   _extra_parts.append(f"회상: {len(past_entries)}개")
        if future_entries: _extra_parts.append(f"기대: {len(future_entries)}개")
        _extra_str = ("  |  " + " / ".join(_extra_parts)) if _extra_parts else ""
        st.caption(
            f"전체 {len(entries)}개 중 최근 {display_count}개를 표시합니다.{_extra_str}"
        )
        for entry in entries[:10]:
            render_entry_card(entry)
        if len(entries) > 10:
            st.info("전체 기록은 '날짜별 보기' 또는 '전체 보기'에서 확인할 수 있습니다.")

    elif view_mode == "분류 확인 필요":
        if not review_entries:
            st.success("분류 확인이 필요한 기록이 없습니다.")
        else:
            st.warning(
                f"분류 확인이 필요한 기록 {len(review_entries)}개를 표시합니다.  \n"
                "수정 버튼을 눌러 내용을 보완한 뒤 재분류하면 정확도가 올라갑니다."
            )
            for entry in review_entries:
                render_entry_card(entry)

    elif view_mode == "회상":
        if not past_entries:
            st.info("회상 기록이 없습니다.")
        else:
            st.caption(f"회상 기록 {len(past_entries)}개를 표시합니다.")
            for entry in past_entries:
                render_entry_card(entry)

    elif view_mode == "기대":
        if not future_entries:
            st.info("기대 기록이 없습니다.")
        else:
            st.caption(f"기대 기록 {len(future_entries)}개를 표시합니다.")
            for entry in future_entries:
                render_entry_card(entry)

    elif view_mode == "날짜별 보기":
        groups: dict = defaultdict(list)
        for entry in entries:
            date_key = entry.created_at[:10] if entry.created_at else "날짜 없음"
            groups[date_key].append(entry)
        for date_key in sorted(groups.keys(), reverse=True):
            day_entries = groups[date_key]
            day_review = sum(1 for e in day_entries if _needs_classification_review(e))
            if day_review > 0:
                expander_label = (
                    f"{date_key} — {len(day_entries)}개 / 확인 필요 {day_review}개"
                )
            else:
                expander_label = f"{date_key} — {len(day_entries)}개"
            with st.expander(expander_label, expanded=False):
                for entry in day_entries:
                    render_entry_card(entry)

    else:  # 전체 보기
        if len(entries) > 50:
            st.warning("전체 보기는 기록이 많을 경우 화면이 길어질 수 있습니다.")
        _all_extra = []
        if review_entries: _all_extra.append(f"분류 확인 필요: {len(review_entries)}개")
        if past_entries:   _all_extra.append(f"회상: {len(past_entries)}개")
        if future_entries: _all_extra.append(f"기대: {len(future_entries)}개")
        _all_str = ("  |  " + " / ".join(_all_extra)) if _all_extra else ""
        st.caption(f"전체 {len(entries)}개를 표시합니다.{_all_str}")
        for entry in entries:
            render_entry_card(entry)

    # ── 분석 ──────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("취향 분석")

    tab_profile, tab_map, tab_mindmap, tab_related, tab_classic = st.tabs(
        ["나의 취향 프로필", "취향 지도 요약", "마인드맵", "연관 태그", "기본 통계"]
    )

    # ── 탭 1: 나의 취향 프로필 ────────────────────────────────────────────────
    with tab_profile:
        profile   = build_preference_profile(entries)
        sentences = generate_profile_sentences(profile)

        m1, m2 = st.columns(2)
        m1.metric("전체 기록 수", profile["total_entries"])
        m2.metric("전체 평균 평점", f'{profile["average_rating"]:.1f} / 5')

        st.divider()
        st.markdown("**요약**")
        for sentence in sentences:
            st.write(f"- {sentence}")

        st.divider()
        with st.expander("프로필 계산 근거 보기"):
            gb1, gb2 = st.columns(2)

            with gb1:
                st.markdown("**긍정 카테고리 (4점 이상)**")
                for cat, avg, cnt in profile["positive_categories"]:
                    st.write(f"- {cat} — {avg:.1f}점 ({cnt}회)")
                if not profile["positive_categories"]:
                    st.caption("없음")

                st.markdown("**긍정 대상 태그**")
                for tag, avg, cnt in profile["top_positive_objects"]:
                    st.write(f"- {tag} — {avg:.1f}점 ({cnt}회)")
                if not profile["top_positive_objects"]:
                    st.caption("없음")

                st.markdown("**긍정 감각 태그**")
                for tag, avg, cnt in profile["top_positive_sensory"]:
                    st.write(f"- {tag} — {avg:.1f}점 ({cnt}회)")
                if not profile["top_positive_sensory"]:
                    st.caption("없음")

                st.markdown("**긍정 상황 태그**")
                for tag, avg, cnt in profile["top_positive_contexts"]:
                    st.write(f"- {tag} — {avg:.1f}점 ({cnt}회)")
                if not profile["top_positive_contexts"]:
                    st.caption("없음")

            with gb2:
                st.markdown("**긍정 관계 태그**")
                for tag, avg, cnt in profile["top_positive_social"]:
                    st.write(f"- {tag} — {avg:.1f}점 ({cnt}회)")
                if not profile["top_positive_social"]:
                    st.caption("없음")

                st.markdown("**긍정 장소 태그**")
                for tag, avg, cnt in profile["top_positive_places"]:
                    st.write(f"- {tag} — {avg:.1f}점 ({cnt}회)")
                if not profile["top_positive_places"]:
                    st.caption("없음")

                st.markdown("**부정 카테고리 (2점 이하)**")
                for cat, avg, cnt in profile["negative_categories"]:
                    st.write(f"- {cat} — {avg:.1f}점 ({cnt}회)")
                if not profile["negative_categories"]:
                    st.caption("없음")

                st.markdown("**부정 대상 태그**")
                for tag, avg, cnt in profile["top_negative_objects"]:
                    st.write(f"- {tag} — {avg:.1f}점 ({cnt}회)")
                if not profile["top_negative_objects"]:
                    st.caption("없음")

                st.markdown("**부정 감각 태그**")
                for tag, avg, cnt in profile["top_negative_sensory"]:
                    st.write(f"- {tag} — {avg:.1f}점 ({cnt}회)")
                if not profile["top_negative_sensory"]:
                    st.caption("없음")

    # ── 탭 2: 취향 지도 요약 ──────────────────────────────────────────────────
    with tab_map:
        st.markdown("#### 취향 지도 요약")

        map_cols = st.columns(5)
        map_sections = [
            ("좋아하는 대상",      get_top_object_tags(entries)),
            ("좋아하는 감각",      get_top_sensory_tags(entries)),
            ("자주 등장하는 상황", get_top_context_tags(entries)),
            ("자주 등장하는 관계", get_top_social_tags(entries)),
            ("자주 등장하는 장소", get_top_place_tags(entries)),
        ]
        for col, (title, items) in zip(map_cols, map_sections):
            with col:
                st.markdown(f"**{title}**")
                if items:
                    for tag, count in items:
                        st.write(f"{tag} — {count}회")
                else:
                    st.caption("데이터 없음")

        st.divider()
        st.markdown("#### 대상별 평균 평점 Top 10")
        rated = get_average_rating_by_tag(entries, "object_tags", limit=10)
        if rated:
            r_cols = st.columns(2)
            for i, (tag, avg, count) in enumerate(rated):
                r_cols[i % 2].write(f"**{tag}** — {avg:.1f}점 ({count}회)")
        else:
            st.info("평점 데이터가 없습니다.")

    # ── 탭 3: 마인드맵 ────────────────────────────────────────────────────────
    with tab_mindmap:
        st.subheader("취향 마인드맵")
        st.caption("기록된 카테고리와 태그를 네트워크 형태로 보여줍니다.")

        with st.expander("필터 설정", expanded=True):
            min_count = st.slider("최소 등장 횟수", min_value=1, max_value=5, value=1)
            fc1, fc2, fc3 = st.columns(3)
            fc4, fc5, fc6 = st.columns(3)
            show_objects     = fc1.checkbox("대상 표시",        value=True)
            show_sensory     = fc2.checkbox("감각 표시",        value=True)
            show_context     = fc3.checkbox("상황 표시",        value=True)
            show_social      = fc4.checkbox("관계 표시",        value=True)
            show_place       = fc5.checkbox("장소 표시",        value=True)
            show_subcategory = fc6.checkbox("서브카테고리 표시", value=False)

            st.divider()
            use_tag_hierarchy = st.checkbox("관련 태그 묶기", value=False)
            st.caption(
                "켜면 드립커피·아메리카노처럼 같은 상위 태그를 공유하는 세부 태그를 그룹 노드 아래로 묶습니다. "
                "같은 상위 태그를 가진 세부 태그가 2개 이상일 때만 묶음이 생성됩니다.  \n"
                "태그 묶기는 마인드맵 생성 시 적용됩니다. 기존 object_tags가 부정확하면 전체 재분류를 실행하세요."
            )

        include_types: set[str] = {"root", "category"}
        if show_objects:
            include_types.add("object")
            include_types.add("object_group")  # 대상 표시와 연동
        if show_sensory:     include_types.add("sensory")
        if show_context:     include_types.add("context")
        if show_social:      include_types.add("social")
        if show_place:       include_types.add("place")
        if show_subcategory: include_types.add("subcategory")

        mindmap  = build_mindmap_data(
            entries,
            min_count=min_count,
            include_types=include_types,
            use_tag_hierarchy=use_tag_hierarchy,
        )
        mm_nodes = mindmap["nodes"]
        mm_edges = mindmap["edges"]

        st.caption(
            f"entries: {len(entries)}개 | nodes: {len(mm_nodes)}개 | "
            f"edges: {len(mm_edges)}개 | 필터: {', '.join(sorted(include_types))} | "
            f"최소 등장: {min_count}회"
        )

        render_mode = st.radio(
            "렌더링 방식",
            ["pyvis (인터랙티브)", "graphviz (정적)"],
            horizontal=True,
            index=0,
            key="mindmap_render_mode",
        )

        if len(mm_nodes) <= 1:
            st.info("마인드맵을 만들 노드가 부족합니다. 기록을 조금 더 추가해보세요.")
        elif len(mm_edges) == 0:
            st.warning("연결된 태그가 없어 마인드맵 엣지가 생성되지 않았습니다.")
        else:
            if render_mode == "pyvis (인터랙티브)":
                try:
                    render_mindmap(mindmap)
                except Exception as exc:
                    st.warning(f"pyvis 렌더링 실패, graphviz로 전환합니다: {exc}")
                    render_mindmap_graphviz(mindmap)
            else:
                render_mindmap_graphviz(mindmap)

        with st.expander("마인드맵 데이터 보기"):
            if mindmap["nodes"]:
                try:
                    import pandas as pd
                    df_nodes = pd.DataFrame(mindmap["nodes"])[["label", "type", "count", "avg_rating"]]
                    df_edges = (
                        pd.DataFrame(mindmap["edges"])
                        if mindmap["edges"]
                        else pd.DataFrame(columns=["source", "target", "weight"])
                    )
                    st.markdown("**노드 목록**")
                    st.dataframe(df_nodes, width="stretch")
                    st.markdown("**엣지 목록**")
                    st.dataframe(df_edges, width="stretch")
                except ImportError:
                    st.markdown("**노드 목록**")
                    st.table([
                        {"label": n["label"], "type": n["type"],
                         "count": n["count"], "avg_rating": n["avg_rating"]}
                        for n in mindmap["nodes"]
                    ])
                    if mindmap["edges"]:
                        st.markdown("**엣지 목록**")
                        st.table(mindmap["edges"])
            else:
                st.info("데이터 없음")

    # ── 탭 4: 연관 태그 ───────────────────────────────────────────────────────
    with tab_related:
        st.subheader("연관 태그 검색")
        st.caption(
            "특정 태그가 포함된 기록에서 함께 등장한 태그를 빈도순으로 보여줍니다. "
            "semantic search가 아니라 co-occurrence 분석입니다."
        )

        _FIELD_LABELS: dict[str, str] = {
            "object_tags":  "대상",
            "sensory_tags": "감각",
            "context_tags": "상황",
            "social_tags":  "관계",
            "place_tags":   "장소",
            "subcategories": "서브카테고리",
            "tags":          "통합태그",
        }

        # 검색 입력 — pending key 패턴으로 버튼 클릭 후 text_input 자동 채우기
        # pending 값이 있으면 위젯 생성 전에 session_state에 적용 후 초기화
        if "related_query_pending" not in st.session_state:
            st.session_state["related_query_pending"] = ""
        if "related_tag_query" not in st.session_state:
            st.session_state["related_tag_query"] = ""

        if st.session_state["related_query_pending"]:
            st.session_state["related_tag_query"]   = st.session_state["related_query_pending"]
            st.session_state["related_query_pending"] = ""

        rt_c1, rt_c2 = st.columns([4, 1])
        with rt_c1:
            query_input = st.text_input(
                "기준 태그",
                placeholder="예: 내장, 커피, 애인, 노포",
                key="related_tag_query",
            )
        with rt_c2:
            st.markdown("<div style='padding-top:1.9rem'></div>", unsafe_allow_html=True)
            rt_limit = st.slider(
                "결과 수", min_value=5, max_value=50, value=20,
                label_visibility="collapsed",
            )

        # 추천 검색어 (자주 등장하는 태그)
        _searchable = get_searchable_tags(entries, limit=30)
        with st.expander("자주 등장하는 태그 (클릭하면 검색)", expanded=not query_input):
            if _searchable:
                _btn_cols = st.columns(5)
                for _bi, (_btag, _bcnt) in enumerate(_searchable):
                    if _btn_cols[_bi % 5].button(
                        f"{_btag} ({_bcnt})",
                        key=f"rtag_{_bi}_{_btag}",
                        use_container_width=True,
                    ):
                        # text_input 생성 이후이므로 pending key 경유
                        st.session_state["related_query_pending"] = _btag
                        st.rerun()
            else:
                st.info("태그가 없습니다.")

        if not query_input:
            st.info("검색할 태그를 입력하세요.")
        else:
            _result = get_related_tags(entries, query_input, limit=rt_limit)
            _qry    = _result["query"]
            _mcnt   = _result["matched_entries_count"]
            _rtags  = _result["related_tags"]

            if _mcnt == 0:
                st.warning(f"'{_qry}' 태그가 포함된 기록을 찾지 못했습니다.")
            else:
                st.success(f"'{_qry}' 태그가 포함된 기록 {_mcnt}개를 찾았습니다.")

                if _rtags:
                    # 결과 테이블
                    try:
                        import pandas as pd
                        _df_rows = [
                            {
                                "연관 태그": item["tag"],
                                "함께 등장": item["count"],
                                "평균 평점": item["avg_rating"],
                                "유형": " · ".join(
                                    _FIELD_LABELS.get(f, f) for f in item["fields"]
                                ) or "—",
                            }
                            for item in _rtags
                        ]
                        st.dataframe(
                            pd.DataFrame(_df_rows),
                            width="stretch", hide_index=True,
                        )
                    except ImportError:
                        for item in _rtags:
                            st.write(
                                f"**{item['tag']}** — {item['count']}회 "
                                f"/ 평균 {item['avg_rating']}점"
                            )

                    # 유형별 요약
                    st.markdown("#### 유형별 연관 태그")
                    _prio_fields = [
                        "object_tags", "sensory_tags", "context_tags",
                        "social_tags", "place_tags",
                    ]
                    _field_groups: dict[str, list[str]] = defaultdict(list)
                    for item in _rtags:
                        pf = item["fields"][0] if item["fields"] else "tags"
                        _field_groups[pf].append(item["tag"])

                    _bp_cols = st.columns(5)
                    for _ci, _pf in enumerate(_prio_fields):
                        with _bp_cols[_ci]:
                            st.markdown(f"**{_FIELD_LABELS[_pf]}**")
                            for _t in _field_groups.get(_pf, [])[:10]:
                                st.write(_t)
                            if not _field_groups.get(_pf):
                                st.caption("없음")

                    _extra = [
                        f"{t} ({_FIELD_LABELS.get(f, f)})"
                        for f in ("subcategories", "tags")
                        for t in _field_groups.get(f, [])
                    ]
                    if _extra:
                        with st.expander("서브카테고리 / 통합태그"):
                            st.write(" · ".join(_extra))
                else:
                    st.info("연관 태그가 없습니다. 기록이 더 필요할 수 있습니다.")

                # 근거 기록
                with st.expander("검색된 기록 보기"):
                    for _me in _result["matched_entries"][:20]:
                        st.write(f"**{_me['raw_text']}**")
                        _d = _me["created_at"][:10] if _me.get("created_at") else "—"
                        st.caption(
                            f"{_me['primary_category']} | {_me['rating']}점 | {_d}"
                        )

    # ── 탭 5: 기본 통계 ───────────────────────────────────────────────────────
    with tab_classic:
        # 시간축 요약
        _tp_summary = get_time_perspective_summary(entries)
        _tp_labels  = {"past": "회상", "present": "경험일", "future": "기대", "unspecified": "날짜 미지정"}
        _tp_nonzero = {k: v for k, v in _tp_summary.items() if v["count"] > 0}
        if any(k != "unspecified" for k in _tp_nonzero):
            st.markdown("**시간축 요약**")
            for _tp_key in ("past", "present", "future", "unspecified"):
                _tpv = _tp_summary.get(_tp_key, {"count": 0, "avg_rating": 0.0})
                if _tpv["count"] > 0:
                    st.write(
                        f"- {_tp_labels[_tp_key]} — "
                        f"{_tpv['count']}개 / 평균 {_tpv['avg_rating']:.1f}점"
                    )
            st.divider()

        col_tags, col_avg = st.columns(2)

        with col_tags:
            st.markdown("**인기 태그 Top 10** (등장 횟수 기준)")
            top_tags = get_top_tags(entries)
            if top_tags:
                for tag, count in top_tags:
                    st.write(f"**{tag}** — {count}회")
            else:
                st.info("태그가 없습니다.")

        with col_avg:
            st.markdown("**주 카테고리별 평균 평점**")
            avg_by_cat = get_average_rating_by_category(entries)
            if avg_by_cat:
                for cat, avg in sorted(avg_by_cat.items(), key=lambda x: -x[1]):
                    st.write(f"**{cat}** — {avg:.1f} / 5")
            else:
                st.info("데이터가 없습니다.")

        st.divider()
        col_high, col_low = st.columns(2)

        with col_high:
            st.markdown("**평점 높은 태그 Top 10**")
            high_rated = get_top_tags_by_rating(entries, ascending=False, limit=10)
            if high_rated:
                for tag, avg, cnt in high_rated:
                    st.write(f"**{tag}** — {avg:.1f}점 ({cnt}회)")
            else:
                st.info("데이터가 없습니다.")

        with col_low:
            st.markdown("**평점 낮은 태그 Top 10**")
            low_rated = get_top_tags_by_rating(entries, ascending=True, limit=10)
            if low_rated:
                for tag, avg, cnt in low_rated:
                    st.write(f"**{tag}** — {avg:.1f}점 ({cnt}회)")
            else:
                st.info("데이터가 없습니다.")
