"""
app.py
------
Stage 4 — Streamlit Dashboard
The interactive UI that drives the entire AI Code Review pipeline.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src.pipeline import PipelineConfig, ReviewPipeline
from src.reviewer import REVIEW_CATEGORIES, SEVERITY_LEVELS, ReviewComment
from src.markdown_reporter import generate_markdown_report

# ---------------------------------------------------------------------------
# App-level config
# ---------------------------------------------------------------------------
load_dotenv()

# Load Streamlit Cloud secrets into env vars (graceful if not present)
try:
    for _key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN"):
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
except Exception:
    pass  # Running locally without secrets.toml — that's fine

st.set_page_config(
    page_title="CodeSight — AI Code Review",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark terminal-meets-editorial aesthetic
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, pre, .stCode, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dark base */
.main { background: #0d0f14; color: #e2e8f0; }
section[data-testid="stSidebar"] { background: #111318 !important; border-right: 1px solid #1e2330; }

/* Severity badges */
.badge { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.08em; }
.badge-critical { background:#3d0f0f; color:#ff6b6b; border:1px solid #ff6b6b44; }
.badge-high     { background:#2d1a00; color:#ff9d42; border:1px solid #ff9d4244; }
.badge-medium   { background:#2a2400; color:#ffd166; border:1px solid #ffd16644; }
.badge-low      { background:#0f2030; color:#63b3ed; border:1px solid #63b3ed44; }
.badge-info     { background:#1a1a2e; color:#b794f4; border:1px solid #b794f444; }

/* Confidence bar */
.conf-bar-wrap { background:#1a1d26; border-radius:4px; height:6px; overflow:hidden; }
.conf-bar      { height:6px; border-radius:4px; transition:width .3s; }
.conf-high { background: linear-gradient(90deg,#48bb78,#38a169); }
.conf-mid  { background: linear-gradient(90deg,#ecc94b,#d69e2e); }
.conf-low  { background: linear-gradient(90deg,#fc8181,#e53e3e); }

/* Comment card */
.review-card {
    background: #141720;
    border: 1px solid #1e2330;
    border-left: 3px solid #4a5568;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 12px;
    position: relative;
}
.review-card.severity-critical { border-left-color: #ff6b6b; }
.review-card.severity-high     { border-left-color: #ff9d42; }
.review-card.severity-medium   { border-left-color: #ffd166; }
.review-card.severity-low      { border-left-color: #63b3ed; }
.review-card.severity-info     { border-left-color: #b794f4; }

.card-title { font-size:15px; font-weight:600; color:#e2e8f0; margin-bottom:6px; }
.card-meta  { font-size:12px; color:#718096; margin-bottom:10px; font-family:'JetBrains Mono',monospace; }
.card-body  { font-size:13px; color:#a0aec0; line-height:1.65; margin-bottom:10px; }
.card-suggestion { background:#0d1117; border-left:2px solid #48bb78; padding:8px 12px; font-size:12px; color:#9ae6b4; font-family:'JetBrains Mono',monospace; border-radius:0 4px 4px 0; }
.verify-label { background:#2d1b00; color:#f6ad55; border:1px solid #f6ad5566; border-radius:4px; padding:2px 8px; font-size:11px; font-weight:700; }

/* Metric card */
.metric-box { background:#141720; border:1px solid #1e2330; border-radius:8px; padding:18px; text-align:center; }
.metric-val { font-size:32px; font-weight:800; font-family:'Syne',sans-serif; }
.metric-lbl { font-size:12px; color:#718096; text-transform:uppercase; letter-spacing:.1em; margin-top:4px; }

/* Section heading */
.section-heading { font-size:18px; font-weight:800; color:#e2e8f0; border-bottom:1px solid #1e2330; padding-bottom:8px; margin:24px 0 16px; letter-spacing:-.01em; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper renderers
# ---------------------------------------------------------------------------

def _severity_badge(severity: str) -> str:
    return f'<span class="badge badge-{severity}">{severity}</span>'


def _confidence_bar(confidence: int) -> str:
    cls = "conf-high" if confidence >= 70 else "conf-mid" if confidence >= 50 else "conf-low"
    return (
        f'<div class="conf-bar-wrap"><div class="conf-bar {cls}" style="width:{confidence}%"></div></div>'
        f'<small style="color:#718096;font-size:11px;">{confidence}% confidence</small>'
    )


def render_comment_card(c: ReviewComment) -> None:
    verify = '<span class="verify-label">⚠ Verify This</span> ' if c.low_confidence else ""
    html = f"""
<div class="review-card severity-{c.severity}">
  <div class="card-title">{verify}{c.title}</div>
  <div class="card-meta">
    {_severity_badge(c.severity)}
    &nbsp;<span class="badge" style="background:#1a1d26;color:#a0aec0;border:1px solid #2d3748;">{c.category}</span>
    &nbsp;📄 {c.file_path}  ·  <span style="font-family:JetBrains Mono,monospace;">{c.node_name}</span>
    {('  · line ' + c.line_hint) if c.line_hint and c.line_hint != 'N/A' else ''}
  </div>
  <div class="card-body">{c.body}</div>
  {"<div class='card-suggestion'>💡 " + c.suggestion + "</div>" if c.suggestion else ""}
  <div style="margin-top:10px">{_confidence_bar(c.confidence)}</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


def render_metrics(comments: list[ReviewComment]) -> None:
    total = len(comments)
    critical = sum(1 for c in comments if c.severity == "critical")
    high = sum(1 for c in comments if c.severity == "high")
    low_conf = sum(1 for c in comments if c.low_confidence)
    avg_conf = int(sum(c.confidence for c in comments) / total) if total else 0

    cols = st.columns(5)
    metrics = [
        (str(total), "Total Issues", "#e2e8f0"),
        (str(critical), "Critical", "#ff6b6b"),
        (str(high), "High Severity", "#ff9d42"),
        (str(low_conf), "Needs Verification", "#f6ad55"),
        (f"{avg_conf}%", "Avg Confidence", "#48bb78"),
    ]
    for col, (val, label, color) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-box"><div class="metric-val" style="color:{color}">{val}</div>'
                f'<div class="metric-lbl">{label}</div></div>',
                unsafe_allow_html=True,
            )


def render_charts(comments: list[ReviewComment]) -> None:
    if not comments:
        return

    df = pd.DataFrame([c.to_dict() for c in comments])

    col1, col2, col3 = st.columns(3)

    with col1:
        sev_counts = df["severity"].value_counts().reindex(SEVERITY_LEVELS, fill_value=0)
        colors = ["#ff6b6b", "#ff9d42", "#ffd166", "#63b3ed", "#b794f4"]
        fig = go.Figure(go.Bar(
            x=sev_counts.index.tolist(),
            y=sev_counts.values.tolist(),
            marker_color=colors,
            text=sev_counts.values.tolist(),
            textposition="outside",
        ))
        fig.update_layout(
            title="Issues by Severity", paper_bgcolor="#141720", plot_bgcolor="#141720",
            font=dict(color="#a0aec0"), margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False, height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_counts = df["category"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=cat_counts.index.tolist(),
            values=cat_counts.values.tolist(),
            hole=0.55,
            textinfo="label+percent",
            textfont_size=11,
        ))
        fig2.update_layout(
            title="Issues by Category", paper_bgcolor="#141720", plot_bgcolor="#141720",
            font=dict(color="#a0aec0"), margin=dict(t=40, b=20, l=20, r=20),
            showlegend=False, height=280,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        fig3 = px.histogram(
            df, x="confidence", nbins=10,
            color_discrete_sequence=["#63b3ed"],
            title="Confidence Distribution",
        )
        fig3.add_vline(x=60, line_dash="dash", line_color="#f6ad55",
                       annotation_text="verify threshold", annotation_position="top right")
        fig3.update_layout(
            paper_bgcolor="#141720", plot_bgcolor="#141720",
            font=dict(color="#a0aec0"), margin=dict(t=40, b=20, l=20, r=20),
            height=280,
        )
        st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

def build_sidebar() -> dict:
    st.sidebar.markdown(
        '<div style="font-family:Syne,sans-serif;font-size:22px;font-weight:800;color:#e2e8f0;'
        'letter-spacing:-.02em;margin-bottom:4px;">🔬 CodeSight</div>'
        '<div style="font-size:12px;color:#718096;margin-bottom:20px;">AI Code Review Agent</div>',
        unsafe_allow_html=True,
    )

    github_url = st.sidebar.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        help="Public repositories only. Private repos require a GitHub token.",
    )

    st.sidebar.markdown(
        '<div style="font-size:11px;color:#718096;margin:-8px 0 6px;">Quick test repos:</div>',
        unsafe_allow_html=True,
    )
    q1, q2, q3 = st.sidebar.columns(3)
    test_repos = [
        ("flask", "https://github.com/pallets/flask"),
        ("requests", "https://github.com/psf/requests"),
        ("fastapi", "https://github.com/tiangolo/fastapi"),
    ]
    for col, (label, url) in zip([q1, q2, q3], test_repos):
        if col.button(label, use_container_width=True, key=f"quick_{label}"):
            st.session_state["quick_url"] = url
            st.rerun()
    if "quick_url" in st.session_state and not github_url:
        github_url = st.session_state.pop("quick_url")

    with st.sidebar.expander("⚙ LLM Settings", expanded=True):
        provider = st.selectbox(
            "Provider", ["anthropic", "openai"],
            help="Claude Sonnet (Anthropic) or GPT-4o-mini (OpenAI)",
        )
        model_defaults = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o-mini",
        }
        model = st.text_input("Model", value=model_defaults[provider])
        api_key = st.text_input(
            f"{'ANTHROPIC' if provider == 'anthropic' else 'OPENAI'} API Key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY", ""),
        )

    with st.sidebar.expander("🎛 Advanced"):
        max_files = st.slider("Max files to review", 5, 60, 20)
        max_nodes = st.slider("Max nodes per file", 5, 30, 15)

    with st.sidebar.expander("🔗 GitHub PR (bonus)", expanded=False):
        gh_token = st.text_input("GitHub Token", type="password",
                                  value=os.environ.get("GITHUB_TOKEN", ""))
        pr_number = st.number_input("PR Number", min_value=1, value=1, step=1)
        post_to_pr = st.checkbox("Post comments to PR after review")

    run = st.sidebar.button("🚀 Run Review", type="primary", use_container_width=True)

    return {
        "github_url": github_url,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "max_files": max_files,
        "max_nodes": max_nodes,
        "gh_token": gh_token,
        "pr_number": int(pr_number),
        "post_to_pr": post_to_pr,
        "run": run,
    }


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Hero header ──────────────────────────────────────────────────
    st.markdown(
        '<div style="padding:24px 0 8px;">'
        '<span style="font-family:Syne,sans-serif;font-size:36px;font-weight:800;'
        'color:#e2e8f0;letter-spacing:-.03em;">Code<span style="color:#63b3ed;">Sight</span></span>'
        '<span style="font-size:14px;color:#718096;margin-left:12px;">Autonomous AI Code Review</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    opts = build_sidebar()

    # ── Session state ────────────────────────────────────────────────
    if "result" not in st.session_state:
        st.session_state.result = None
    if "log" not in st.session_state:
        st.session_state.log = []

    # ── Run pipeline ─────────────────────────────────────────────────
    if opts["run"]:
        if not opts["github_url"]:
            st.sidebar.error("Please enter a GitHub URL.")
            return
        if not opts["api_key"]:
            st.sidebar.error("API key is required.")
            return

        st.session_state.result = None
        st.session_state.log = []

        cfg = PipelineConfig(
            github_url=opts["github_url"],
            provider=opts["provider"],
            model=opts["model"],
            api_key=opts["api_key"],
            max_files=opts["max_files"],
            max_nodes_per_file=opts["max_nodes"],
        )
        pipeline = ReviewPipeline(cfg)

        progress_bar = st.progress(0.0, text="Starting pipeline …")
        status_box = st.empty()
        log_expander = st.expander("📋 Live Log", expanded=False)
        log_placeholder = log_expander.empty()

        try:
            for event in pipeline.run():
                progress_bar.progress(min(event.pct, 1.0), text=event.message)
                status_box.info(f"**{event.stage.upper()}** — {event.message}")
                st.session_state.log.append(f"[{event.stage}] {event.message}")
                log_placeholder.code("\n".join(st.session_state.log[-30:]), language=None)

                if event.stage == "error":
                    st.error(event.message)
                    return
        except Exception as exc:
            st.error(f"Pipeline crashed: {exc}")
            return

        progress_bar.progress(1.0, text="✅ Review complete!")
        st.session_state.result = pipeline.result

        # Optional: post to GitHub PR
        if opts["post_to_pr"] and pipeline.result and opts["gh_token"]:
            with st.spinner("Posting comments to GitHub PR …"):
                try:
                    from src.github_poster import GitHubPRPoster
                    poster = GitHubPRPoster(
                        opts["github_url"], opts["pr_number"], token=opts["gh_token"]
                    )
                    poster.post_review(pipeline.result.all_comments)
                    poster.post_summary_issue_comment(pipeline.result.all_comments)
                    st.success(f"✅ Posted {len(pipeline.result.all_comments)} comments to PR #{opts['pr_number']}")
                except Exception as exc:
                    st.warning(f"PR posting failed: {exc}")

    # ── Render results ───────────────────────────────────────────────
    result = st.session_state.result
    if result is None:
        _render_landing()
        return

    comments = result.all_comments
    if not comments:
        st.success("✅ No issues found! The agent reviewed the repository and found nothing to flag.")
        return

    # Metrics row
    st.markdown('<div class="section-heading">📊 Overview</div>', unsafe_allow_html=True)
    render_metrics(comments)

    # Charts
    render_charts(comments)

    # Filters
    st.markdown('<div class="section-heading">🔍 Review Comments</div>', unsafe_allow_html=True)

    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 2, 1])
    with f_col1:
        sev_filter = st.multiselect(
            "Severity", SEVERITY_LEVELS, default=SEVERITY_LEVELS, key="sev_filter"
        )
    with f_col2:
        cat_filter = st.multiselect(
            "Category", REVIEW_CATEGORIES, default=REVIEW_CATEGORIES, key="cat_filter"
        )
    with f_col3:
        conf_filter = st.selectbox(
            "Confidence", ["All", "High (≥70)", "Medium (50–69)", "Low (<50)", "⚠ Verify This"],
            key="conf_filter",
        )
    with f_col4:
        show_low_conf = st.checkbox("Show 'Verify This' first", value=True)

    # Apply filters
    filtered = [
        c for c in comments
        if c.severity in sev_filter and c.category in cat_filter
    ]
    if conf_filter == "High (≥70)":
        filtered = [c for c in filtered if c.confidence >= 70]
    elif conf_filter == "Medium (50–69)":
        filtered = [c for c in filtered if 50 <= c.confidence < 70]
    elif conf_filter == "Low (<50)":
        filtered = [c for c in filtered if c.confidence < 50]
    elif conf_filter == "⚠ Verify This":
        filtered = [c for c in filtered if c.low_confidence]

    if show_low_conf:
        filtered = sorted(filtered, key=lambda c: (not c.low_confidence, SEVERITY_LEVELS.index(c.severity)))

    st.caption(f"Showing {len(filtered)} of {len(comments)} comments")

    # High confidence section
    high_conf = [c for c in filtered if not c.low_confidence]
    if high_conf:
        st.markdown(
            f'<div style="font-size:14px;font-weight:600;color:#48bb78;margin-bottom:8px;">'
            f'✅ High Confidence ({len(high_conf)})</div>',
            unsafe_allow_html=True,
        )
        for c in high_conf:
            render_comment_card(c)

    # Low confidence section
    low_conf_list = [c for c in filtered if c.low_confidence]
    if low_conf_list:
        st.markdown(
            f'<div style="font-size:14px;font-weight:600;color:#f6ad55;margin:16px 0 8px;">'
            f'⚠️ Verify These ({len(low_conf_list)}) — Low Confidence</div>',
            unsafe_allow_html=True,
        )
        for c in low_conf_list:
            render_comment_card(c)

    # Download
    st.markdown('<div class="section-heading">⬇ Export</div>', unsafe_allow_html=True)
    d_col1, d_col2, d_col3 = st.columns(3)

    with d_col1:
        df_export = pd.DataFrame([c.to_dict() for c in comments])
        csv = df_export.to_csv(index=False)
        st.download_button(
            "📥 Download CSV", csv, file_name=f"{result.repo_name}_review.csv",
            mime="text/csv", use_container_width=True,
        )

    with d_col2:
        json_export = json.dumps([c.to_dict() for c in comments], indent=2)
        st.download_button(
            "📥 Download JSON", json_export, file_name=f"{result.repo_name}_review.json",
            mime="application/json", use_container_width=True,
        )

    with d_col3:
        md_report = generate_markdown_report(
            repo_name=result.repo_name,
            comments=comments,
            metadata=result.metadata,
            errors=result.errors,
        )
        st.download_button(
            "📥 Download Markdown Report", md_report,
            file_name=f"{result.repo_name}_review.md",
            mime="text/markdown", use_container_width=True,
        )

    # Errors
    if result.errors:
        with st.expander(f"⚠️ {len(result.errors)} pipeline errors"):
            for err in result.errors:
                st.error(err)


def _render_landing() -> None:
    st.markdown("""
<div style="text-align:center;padding:60px 0;">
  <div style="font-size:64px;margin-bottom:16px;">🔬</div>
  <div style="font-size:22px;font-weight:800;color:#e2e8f0;margin-bottom:10px;">Ready to review</div>
  <div style="font-size:15px;color:#718096;max-width:480px;margin:0 auto;line-height:1.7;">
    Paste a public GitHub URL in the sidebar, configure your LLM provider,
    and hit <strong style="color:#63b3ed;">Run Review</strong>.
    The agent will clone the repo, parse its AST structure, and generate
    confidence-rated code review comments.
  </div>
  <div style="margin-top:40px;display:flex;justify-content:center;gap:32px;flex-wrap:wrap;">
    <div style="background:#141720;border:1px solid #1e2330;border-radius:8px;padding:20px 28px;max-width:200px;">
      <div style="font-size:28px">📥</div>
      <div style="font-weight:700;color:#e2e8f0;margin:8px 0 4px;">Clone</div>
      <div style="font-size:12px;color:#718096;">GitPython shallow clone</div>
    </div>
    <div style="background:#141720;border:1px solid #1e2330;border-radius:8px;padding:20px 28px;max-width:200px;">
      <div style="font-size:28px">🌳</div>
      <div style="font-weight:700;color:#e2e8f0;margin:8px 0 4px;">Parse</div>
      <div style="font-size:12px;color:#718096;">AST → functions, classes</div>
    </div>
    <div style="background:#141720;border:1px solid #1e2330;border-radius:8px;padding:20px 28px;max-width:200px;">
      <div style="font-size:28px">🤖</div>
      <div style="font-weight:700;color:#e2e8f0;margin:8px 0 4px;">Review</div>
      <div style="font-size:12px;color:#718096;">Claude / GPT-4o-mini JSON</div>
    </div>
    <div style="background:#141720;border:1px solid #1e2330;border-radius:8px;padding:20px 28px;max-width:200px;">
      <div style="font-size:28px">📊</div>
      <div style="font-weight:700;color:#e2e8f0;margin:8px 0 4px;">Report</div>
      <div style="font-size:12px;color:#718096;">Dashboard + CSV/JSON export</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()