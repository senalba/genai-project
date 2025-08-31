import os
import sys
from urllib.parse import urlparse

import streamlit as st

# Make sure we can import from ./src
sys.path.append(os.path.dirname(__file__))

from src.main import (  # noqa: E402
    FEEDS,
    ingest,
    build_embeddings,
    greedy_clusters,
    build_stories,
    summarize_selected,
)

st.set_page_config(page_title="News Impact (Baseline)", layout="wide")
st.title("News Impact — Top Positive/Negative")

# -------- Sidebar controls --------
st.sidebar.header("Controls")

# Feed picker
def _label(u: str) -> str:
    d = urlparse(u).netloc.lower()
    if d.startswith("www."):
        d = d[4:]
    return f"{d}"

default_feeds = FEEDS
selected = st.sidebar.multiselect(
    "RSS feeds",
    options=default_feeds,
    default=default_feeds,
    format_func=_label,
)

# Params
max_items = st.sidebar.number_input("Max items", min_value=100, max_value=1500, value=600, step=50)
max_age_hours = st.sidebar.number_input("Max age (hours)", min_value=6, max_value=168, value=48, step=6)
sim_thr = st.sidebar.slider("Similarity threshold", min_value=0.60, max_value=0.90, value=0.72, step=0.01)
topk = st.sidebar.slider("Top-K per sentiment", min_value=1, max_value=10, value=5, step=1)
min_domains = st.sidebar.slider("Min distinct domains per story", min_value=1, max_value=4, value=2, step=1)

# Summarizer
hf_model = st.sidebar.selectbox(
    "Summarizer model",
    options=[
        "sshleifer/distilbart-cnn-12-6",  # fast
        "facebook/bart-large-cnn",
        "google/pegasus-cnn_dailymail",
    ],
    index=0,
)
hf_device = st.sidebar.selectbox("Device", options=["auto", "mps", "cpu"], index=0)

get_btn = st.sidebar.button("Get current top news", type="primary")

# -------- Helpers for selection/ranking --------
def select_top_with_min(stories, k=5, min_domains_req=2):
    def score(s):
        return (-s.get("mention_count_domains", 0), -s.get("mention_count", 0), s.get("first_seen") or "")

    good = [s for s in stories if s.get("mention_count_domains", 0) >= min_domains_req]
    bad = [s for s in stories if s.get("mention_count_domains", 0) < min_domains_req]

    pos = sorted([s for s in good if s["sentiment"] == "positive"], key=score)[:k]
    neg = sorted([s for s in good if s["sentiment"] == "negative"], key=score)[:k]

    if len(pos) < k:
        pos += sorted([s for s in bad if s["sentiment"] == "positive"], key=score)[: k - len(pos)]
    if len(neg) < k:
        neg += sorted([s for s in bad if s["sentiment"] == "negative"], key=score)[: k - len(neg)]
    return pos, neg

# -------- Main action --------
if get_btn:
    if not selected:
        st.warning("Pick at least one RSS feed.")
        st.stop()

    with st.status("Fetching & processing…", expanded=False) as status:
        # 1) Ingest
        df = ingest(selected, max_items=max_items, max_age_hours=max_age_hours)
        if df.empty:
            status.update(label="No fresh items found.", state="error")
            st.stop()

        # 2) Embed + cluster
        X = build_embeddings(df)
        groups = greedy_clusters(X, thr=sim_thr)

        # 3) Build stories
        stories = build_stories(df, groups)

        # 4) Select Top-K with min-domain constraint
        top_pos, top_neg = select_top_with_min(stories, k=topk, min_domains_req=min_domains)

        # 5) Summarize Top-K (article body)
        summarize_selected(top_pos, summarizer="hf", hf_model=hf_model, hf_device=hf_device)
        summarize_selected(top_neg, summarizer="hf", hf_model=hf_model, hf_device=hf_device)

        status.update(label="Done.", state="complete")

    # -------- Render results --------
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader(f"Top {len(top_pos)} Positive")
        if not top_pos:
            st.info("No positive stories matched the filters.")
        for s in top_pos:
            with st.container(border=True):
                st.markdown(f"#### {s['title']}")
                st.write(s.get("summary_llm") or "")
                st.caption(
                    f"mentions: {s.get('mention_count',0)} · domains: {s.get('mention_count_domains',0)} · "
                    f"first seen: {s.get('first_seen','–')}"
                )
                if s.get("canonical_url"):
                    st.link_button("Open article", s["canonical_url"], use_container_width=True, type="secondary")
                if s.get("domains"):
                    st.caption("Sources: " + ", ".join(s["domains"][:6]))

    with col2:
        st.subheader(f"Top {len(top_neg)} Negative")
        if not top_neg:
            st.info("No negative stories matched the filters.")
        for s in top_neg:
            with st.container(border=True):
                st.markdown(f"#### {s['title']}")
                st.write(s.get("summary_llm") or "")
                st.caption(
                    f"mentions: {s.get('mention_count',0)} · domains: {s.get('mention_count_domains',0)} · "
                    f"first seen: {s.get('first_seen','–')}"
                )
                if s.get("canonical_url"):
                    st.link_button("Open article", s["canonical_url"], use_container_width=True, type="secondary")
                if s.get("domains"):
                    st.caption("Sources: " + ", ".join(s["domains"][:6]))

    # Stats
    with st.expander("Run stats"):
        st.json(
            {
                "items_ingested": int(len(df)),
                "groups": int(len(groups)),
                "feeds_used": [_label(u) for u in selected],
                "similarity_threshold": sim_thr,
                "max_age_hours": max_age_hours,
            }
        )
else:
    st.info("Pick feeds on the left, then click **Get current top news**.")

