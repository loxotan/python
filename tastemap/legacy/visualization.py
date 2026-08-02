from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

_TYPE_COLOR: dict[str, str] = {
    "root":         "#222222",
    "category":     "#4C78A8",
    "object_group": "#F2A65A",
    "object":       "#F58518",
    "sensory":      "#E45756",
    "context":      "#72B7B2",
    "social":       "#54A24B",
    "place":        "#B279A2",
    "subcategory":  "#9D755D",
}

_DOT_COLOR: dict[str, str] = {**_TYPE_COLOR}


# ── pyvis 렌더링 ──────────────────────────────────────────────────────────────

def render_mindmap(mindmap_data: dict, height: int = 760) -> None:
    """pyvis Network으로 Obsidian 스타일 인터랙티브 마인드맵을 렌더링한다.
    generate_html()을 직접 사용해 Windows cp949 인코딩 문제를 우회한다."""
    nodes = mindmap_data.get("nodes", [])
    edges = mindmap_data.get("edges", [])

    if len(nodes) <= 1:
        st.info("마인드맵을 만들 데이터가 아직 부족합니다.")
        return

    net = Network(
        height=f"{height}px",
        width="100%",
        bgcolor="#fafafa",
        font_color="#222222",
        cdn_resources="in_line",
        directed=False,
        notebook=False,
    )

    valid_node_ids: set[str] = set()

    for node in nodes:
        node_id = str(node.get("id", "")).strip()
        if not node_id:
            continue
        label      = str(node.get("label", node_id))
        ntype      = str(node.get("type", ""))
        count      = int(node.get("count", 1) or 1)
        avg_rating = float(node.get("avg_rating", 3.0) or 3.0)

        if ntype == "root":
            size = 42
        elif ntype == "category":
            size = 30
        elif ntype == "object_group":
            size = 20 + min(count * 3, 18)
        else:
            size = 12 + min(count * 3, 18)

        color = _TYPE_COLOR.get(ntype, "#999999")
        if ntype not in ("root", "category") and avg_rating <= 2.0:
            color = "#D62728"

        title = f"유형: {ntype}<br>등장: {count}회<br>평균 평점: {avg_rating}"
        valid_node_ids.add(node_id)
        net.add_node(node_id, label=label, title=title, color=color, size=size)

    for edge in edges:
        src    = str(edge.get("source", "")).strip()
        tgt    = str(edge.get("target", "")).strip()
        weight = int(edge.get("weight", 1) or 1)
        if not src or not tgt:
            continue
        if src not in valid_node_ids or tgt not in valid_node_ids:
            continue
        net.add_edge(src, tgt, value=weight, width=min(1 + weight, 5))

    net.set_options("""
{
  "interaction": {
    "hover": true,
    "tooltipDelay": 120,
    "navigationButtons": true,
    "keyboard": true,
    "dragNodes": true,
    "zoomView": true,
    "dragView": true
  },
  "physics": {
    "enabled": true,
    "stabilization": {
      "enabled": true,
      "iterations": 800,
      "updateInterval": 25
    },
    "barnesHut": {
      "gravitationalConstant": -45000,
      "centralGravity": 0.18,
      "springLength": 130,
      "springConstant": 0.045,
      "damping": 0.18,
      "avoidOverlap": 0.35
    }
  },
  "nodes": {
    "shape": "dot",
    "font": {
      "size": 14,
      "face": "Arial",
      "color": "#222222"
    },
    "borderWidth": 2
  },
  "edges": {
    "color": {
      "color": "rgba(140,140,140,0.35)",
      "highlight": "rgba(80,80,80,0.8)"
    },
    "smooth": {
      "enabled": true,
      "type": "dynamic"
    }
  }
}
""")

    # generate_html()은 파일 I/O 없이 HTML 문자열만 반환 → cp949 문제 없음
    html: str = net.generate_html()

    # body 스타일 패치
    html = html.replace(
        "<body>",
        "<body style='margin:0;padding:0;overflow:hidden;background:#fafafa;'>",
        1,
    )

    has_vis = "vis.Network" in html or "new vis.Network" in html
    st.caption(f"pyvis html 길이: {len(html):,} 자 | vis.Network: {has_vis}")

    components.html(html, height=height + 30, scrolling=False)


# ── graphviz DOT 렌더링 (fallback) ───────────────────────────────────────────

def render_mindmap_graphviz(mindmap_data: dict) -> None:
    """DOT 문자열 기반 정적 마인드맵 — graphviz Python 패키지 불필요."""
    nodes = mindmap_data.get("nodes", [])
    edges = mindmap_data.get("edges", [])

    if len(nodes) <= 1:
        st.info("마인드맵을 만들 데이터가 아직 부족합니다.")
        return

    node_map = {str(n["id"]): n for n in nodes if n.get("id")}

    def _safe(s: str) -> str:
        for ch in (":", " ", "-", "/"):
            s = s.replace(ch, "_")
        return s

    lines = [
        "digraph {",
        "  rankdir=LR;",
        '  node [shape=box style="rounded,filled" fontname="Arial"];',
        '  edge [color="#aaaaaa"];',
    ]

    valid_safe: dict[str, str] = {}
    for nid, node in node_map.items():
        sid   = _safe(nid)
        label = str(node.get("label", nid)).replace('"', "'")
        ntype = str(node.get("type", "tag"))
        count = node.get("count", 1)
        avg   = node.get("avg_rating", 0)
        fill  = _DOT_COLOR.get(ntype, "#dddddd")
        fcol  = "white" if ntype in ("root", "category") else "black"
        lines.append(
            f'  "{sid}" [label="{label}\\n({count}회 / {avg}점)" '
            f'fillcolor="{fill}" fontcolor="{fcol}"];'
        )
        valid_safe[nid] = sid

    for edge in edges:
        src = str(edge.get("source", ""))
        tgt = str(edge.get("target", ""))
        if src not in valid_safe or tgt not in valid_safe:
            continue
        lines.append(f'  "{valid_safe[src]}" -> "{valid_safe[tgt]}";')

    lines.append("}")
    dot_str = "\n".join(lines)

    try:
        st.graphviz_chart(dot_str, width="stretch")
    except Exception as exc:
        st.error(f"graphviz 렌더링 실패: {exc}")
        with st.expander("DOT 소스 보기"):
            st.code(dot_str[:2000], language="dot")
