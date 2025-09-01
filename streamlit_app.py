# streamline.py
import json
import os
import sys
import html
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

# Make project modules importable
sys.path.append(os.path.dirname(__file__))

from src.main import (  # noqa: E402
    FEEDS,
    ingest,
    build_embeddings,
    greedy_clusters,
    build_stories,
    select_top_overall,
    assemble_report_payload,
    update_titles_cache,
    update_articles_cache,
    load_articles_map,
)

st.set_page_config(page_title="News Impact", layout="wide")
st.title("News Impact — Top Most-Mentioned Stories (distinct outlets)")


# ---------- helpers ----------
def _label(u: str) -> str:
    d = urlparse(u).netloc.lower()
    return d[4:] if d.startswith("www.") else d


@st.cache_data(show_spinner=False)
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def titles_cache_df(path="out/titles_cache.json") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=["title", "url", "source", "domain", "published_at"]
        )
    data = load_json(path)
    items = data.get("items", [])
    return pd.DataFrame(items)


# ---------- sidebar controls ----------
st.sidebar.header("Controls")

selected_feeds = st.sidebar.multiselect(
    "RSS feeds", options=FEEDS, default=FEEDS, format_func=_label
)

max_items = st.sidebar.number_input(
    "Max items (ingest)", min_value=50, max_value=2000, value=600, step=50
)
max_age_hours = st.sidebar.number_input(
    "Max age (hours)", min_value=6, max_value=720, value=48, step=6
)
sim_thr = st.sidebar.slider("Similarity threshold", 0.60, 0.90, 0.72, 0.01)
topk = st.sidebar.slider("Top-N stories", 1, 10, 5, 1)

oa_model = st.sidebar.selectbox(
    "OpenAI model",
    ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"],
    index=0,
)

article_timeout = st.sidebar.number_input("Article timeout (s)", 3, 30, 10, 1)

update_btn = st.sidebar.button("Update cache", type="primary")
summarize_btn = st.sidebar.button("Summarize", type="secondary")

# ---------- tabs ----------
tab_top, tab_titles = st.tabs(["Top stories", "All titles"])

# ---------- Update cache ----------
if update_btn:
    with st.status("Updating cache…", expanded=False) as status:
        # Ingest with fallback
        df = ingest(
            selected_feeds,
            max_items=int(max_items),
            max_age_hours=int(max_age_hours),
            allow_undated=False,
        )
        if df.empty:
            df = ingest(
                selected_feeds,
                max_items=int(max_items),
                max_age_hours=max(168, int(max_age_hours)),
                allow_undated=True,
            )
        if df.empty:
            status.update(label="No items to cache.", state="error")
            st.stop()

        st.sidebar.caption(f"Fetched rows: {len(df)} (after filters)")

        added_titles, total_titles = update_titles_cache(df, out_dir="out")
        st.sidebar.success(f"Titles cache: +{added_titles} (total {total_titles})")

        added_bodies, total_bodies = update_articles_cache(
            df, out_dir="out", timeout=int(article_timeout)
        )
        st.sidebar.success(f"Articles cache: +{added_bodies} (total {total_bodies})")

        status.update(label="Cache updated.", state="complete")


# ---------- Summarize from cache ----------
def render_report(rep: dict):
    items = rep.get("items", [])
    topN = rep.get("topN", 5)  # default to 5
    st.subheader(f"Top {topN} most-mentioned stories (by distinct outlets)")
    for item in items:
        with st.container(border=True):
            st.markdown(f"#### {item.get('summary_title','(no title)')}")
            st.write(item.get("summary", ""))
            stats = item.get("stats", {})
            st.caption(
                f"outlets: {stats.get('distinct_outlets','?')} · "
                f"first seen: {stats.get('first_seen') or '–'}"
            )
            for u in item.get("links", []):
                st.markdown(f"- [{_label(u)}]({u})")


if summarize_btn:
    with st.status(
        "Summarizing from cached titles (OpenAI)…", expanded=False
    ) as status:
        tdf = titles_cache_df("out/titles_cache.json")
        if tdf.empty:
            status.update(
                label="No cached titles. Click **Update cache** first.", state="error"
            )
            st.stop()

        tdf["published_at"] = pd.to_datetime(
            tdf["published_at"], utc=True, errors="coerce"
        )
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=int(max_age_hours))
        adf = tdf[tdf["published_at"] >= cutoff].copy()
        if adf.empty:
            adf = tdf.copy()

        adf["summary_hint"] = ""
        adf["source"] = adf["source"].fillna("")

        X = build_embeddings(adf)
        groups = greedy_clusters(X, thr=float(sim_thr))
        stories = build_stories(adf, groups)

        top = select_top_overall(stories, k=int(topk))

        articles_map = load_articles_map("out")
        payload = assemble_report_payload(
            top,
            summarizer="openai",
            oa_model=oa_model,
            articles_map=articles_map,
        )

        os.makedirs("out", exist_ok=True)
        with open("out/report.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        st.session_state["last_report"] = payload
        status.update(label="Done.", state="complete")

        # with tab_top:
        #     render_report(st.session_state["last_report"])
        st.rerun()

# ---------- Auto-load last report on startup ----------
if "last_report" not in st.session_state:
    rpt_path = "out/report.json"
    if os.path.exists(rpt_path):
        try:
            st.session_state["last_report"] = load_json(rpt_path)
        except Exception:
            pass

if st.session_state.get("last_report"):
    with tab_top:
        render_report(st.session_state["last_report"])

# ---------- All titles tab ----------
with tab_titles:
    cache_path = "out/titles_cache.json"
    if not os.path.exists(cache_path):
        st.info("No titles cache yet. Click **Update cache**.")
    else:
        data = load_json(cache_path)
        items = data.get("items", [])
        if not items:
            st.info("Titles cache is empty.")
        else:
            by_source = {}
            for it in items:
                by_source.setdefault(it.get("source", "(unknown)"), []).append(it)

            srcs = sorted(by_source.keys())
            sel = st.multiselect("Filter outlets", options=srcs, default=srcs)
            for source in sel:
                arr = by_source.get(source, [])
                if not arr:
                    continue
                with st.expander(f"{source} — {len(arr)} titles", expanded=False):
                    for it in sorted(
                        arr, key=lambda x: x.get("published_at") or "", reverse=True
                    ):
                        url = it.get("url", "")
                        title = it.get("title", "")
                        ts = it.get("published_at", "—")
                        dom = it.get("domain", "")
                        st.markdown(
                            f"<div style='margin:6px 0'>"
                            f"<a href='{html.escape(url)}' target='_blank'>{html.escape(title)}</a><br>"
                            f"<span style='color:gray;font-size:0.9em'>{ts} · {dom}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
