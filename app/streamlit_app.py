# streamlit_app.py
import json
import os
import sys
import html
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

try:
    import tldextract
except Exception:
    tldextract = None

# Make project modules importable
# sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from news_impact.core import (
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
    translate_needed_titles,
    translate_needed_bodies,
    get_recent_titles_df,
    openai_select_by_topic,
    openai_topic_summary,
    load_or_generate_topic_summary,
    load_or_translate_topic_summary,
    topic_summary_paths,
    topic_tts_path,
    get_or_generate_topic_tts,
)

st.set_page_config(page_title="News Impact", layout="wide")
st.title("News Impact — Top Most-Mentioned Stories (distinct outlets)")


# ---------- helpers ----------


def registrable_domain(host: str) -> str:
    if not host:
        return ""
    if tldextract:
        ext = tldextract.extract(host)
        return ".".join([p for p in (ext.domain, ext.suffix) if p]) or host
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


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
    "Max items (ingest)", min_value=50, max_value=2000, value=1600, step=50
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
translate_titles = st.sidebar.checkbox("Translate titles to English", value=True)
translate_bodies = st.sidebar.checkbox("Translate article bodies (costly)", value=False)

article_timeout = st.sidebar.number_input("Article timeout (s)", 3, 30, 10, 1)

update_btn = st.sidebar.button("Update cache", type="primary")
summarize_btn = st.sidebar.button("Summarize", type="secondary")

# ---------- tabs ----------
tab_top, tab_titles, tab_war, tab_tech, tab_econ = st.tabs(
    ["Top stories", "All titles", "Russo-Ukrainian War", "Technologies", "Economics"]
)


# ---------- Update cache ----------
if update_btn:
    with st.status("Updating cache…", expanded=False) as status:
        # 1) Ingest
        df = ingest(
            selected_feeds,
            max_items=int(max_items),
            max_age_hours=int(max_age_hours),
            allow_undated=True,  # keep undated UA items
        )
        if df.empty:
            status.update(label="No items to cache.", state="error")
            st.stop()

        # 2) Write caches (titles + article bodies)
        added_titles, total_titles = update_titles_cache(df, out_dir="out")
        added_bodies, total_bodies = update_articles_cache(
            df, out_dir="out", timeout=int(article_timeout)
        )

        ## 3) Translate (idempotent; only new/changed items)
        tx_titles_new = tx_bodies_new = 0
        if translate_titles:
            tx_titles_new, _ = translate_needed_titles(out_dir="out", model=oa_model)
        if translate_bodies:
            tx_bodies_new, _ = translate_needed_bodies(out_dir="out", model=oa_model)

        # 4) Bust Streamlit file cache so the UI sees the updated JSON files
        try:
            load_json.clear()  # clear this cached function
        except Exception:
            pass
        st.cache_data.clear()  # brute-force fallback

        # 5) Sidebar feedback
        st.sidebar.success(f"Titles cache: +{added_titles} (total {total_titles})")
        st.sidebar.success(f"Articles cache: +{added_bodies} (total {total_bodies})")
        st.sidebar.success(f"Translated titles: +{tx_titles_new}")
        st.sidebar.success(f"Translated bodies: +{tx_bodies_new}")

        status.update(label="Cache updated & translated.", state="complete")


# ---------- Summarize from cache ----------
def render_report(rep: dict):
    items = rep.get("items", [])
    topN = rep.get("topN", 5)
    st.subheader(f"Top {topN} most-mentioned stories (by distinct outlets)")

    # Build url -> (title_en|title, registrable domain)
    url_to_title = {}
    url_to_regdom = {}
    try:
        cache = load_json("out/titles_cache.json")
        for it in cache.get("items", []):
            u = it.get("url", "")
            if not u:
                continue
            host = urlparse(u).netloc.lower()
            host = host[4:] if host.startswith("www.") else host
            # prefer English title if available
            title = it.get("title_en") or it.get("title") or ""
            # registrable domain label
            try:
                import tldextract

                ext = tldextract.extract(host)
                reg = ".".join([p for p in (ext.domain, ext.suffix) if p]) or host
            except Exception:
                parts = host.split(".")
                reg = ".".join(parts[-2:]) if len(parts) >= 2 else host
            url_to_title[u] = title
            url_to_regdom[u] = reg
    except Exception:
        pass

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
                host = urlparse(u).netloc.lower()
                host = host[4:] if host.startswith("www.") else host
                reg = url_to_regdom.get(u) or host
                title = url_to_title.get(u) or "(title unavailable)"
                st.markdown(
                    "- "
                    + f"<strong>[{html.escape(reg.upper())}]</strong> "
                    + f"<a href='{html.escape(u)}' target='_blank'>{html.escape(title)}</a>",
                    unsafe_allow_html=True,
                )


def render_topic_tab(container, topic_key: str, topic_label: str):
    """Drop-in: keeps all existing behavior, adds TTS (per-language) with playback & download."""
    with container:
        st.subheader(topic_label)

        # --- Controls ---
        col_lang, col_voice, col_gen, col_limit = st.columns([1.2, 1, 1, 1])
        with col_lang:
            summary_lang = st.radio(
                "Summary language",
                options=["English", "Українська"],
                index=0,
                horizontal=True,
                key=f"lang_{topic_key}",
            )
        with col_voice:
            voice = st.selectbox(
                "TTS voice",
                ["alloy", "verse", "luna", "sparkle"],
                index=0,
                key=f"voice_{topic_key}",
            )
        with col_gen:
            gen = st.button(f"Generate — {topic_label}", key=f"btn_{topic_key}")
        with col_limit:
            max_titles_for_topic = st.number_input(
                "Max titles to send",
                min_value=50,
                max_value=800,
                value=min(int(max_items), 400),
                step=50,
                key=f"lim_{topic_key}",
            )

        # --- Generate summary flow ---
        if gen:
            with st.status(
                f"Selecting and summarizing for '{topic_label}'…", expanded=False
            ) as status:

                en_path, uk_path = topic_summary_paths(topic_label, out_dir="../out")
                en_exists = os.path.exists(en_path) and os.path.getsize(en_path) > 0
                uk_exists = os.path.exists(uk_path) and os.path.getsize(uk_path) > 0

                if en_exists or uk_exists:
                    # Load what we have on disk
                    en_text = ""
                    uk_text = None
                    try:
                        if en_exists:
                            with open(en_path, "r", encoding="utf-8") as f:
                                en_text = f.read()
                        if uk_exists:
                            with open(uk_path, "r", encoding="utf-8") as f:
                                uk_text = f.read()
                    except Exception as e:
                        # If loading fails, fall back to full flow below
                        en_text = ""
                        uk_text = None

                    if en_text:
                        # Ensure English titles exist if you want selection to work better
                        try:
                            if translate_titles:
                                translate_needed_titles(out_dir="out", model=oa_model)
                        except Exception:
                            pass

                        # Build links list if we don't have it in session (cheap: reuse picker)
                        items_records = (
                            st.session_state.get("topic_reports", {})
                            .get(topic_key, {})
                            .get("items")
                        )
                        if not items_records:
                            tdf = get_recent_titles_df(
                                out_dir="out", max_age_hours=int(max_age_hours)
                            )
                            try:
                                picked = openai_select_by_topic(
                                    tdf=tdf,
                                    topic=topic_label,
                                    oa_model=oa_model,
                                    max_titles=int(max_titles_for_topic),
                                )
                                items_records = picked.to_dict(orient="records")
                            except Exception:
                                items_records = []

                        # Save to session (no re-generation)
                        st.session_state.setdefault("topic_reports", {})[topic_key] = {
                            "summary_md": en_text,  # EN already on disk
                            "summary_md_uk": uk_text,  # UA if present on disk
                            "items": items_records or [],
                            "path": en_path,
                            "path_uk": uk_path if uk_exists else None,
                        }
                        status.update(
                            label="Loaded existing summary from disk.", state="complete"
                        )
                        st.rerun()

                try:
                    if translate_titles:
                        translate_needed_titles(out_dir="out", model=oa_model)
                except Exception:
                    pass

                tdf = get_recent_titles_df(
                    out_dir="out", max_age_hours=int(max_age_hours)
                )
                if tdf.empty:
                    status.update(label="No recent titles available.", state="error")
                    st.stop()

                try:
                    picked = openai_select_by_topic(
                        tdf=tdf,
                        topic=topic_label,
                        oa_model=oa_model,
                        max_titles=int(max_titles_for_topic),
                    )
                except RuntimeError as e:
                    status.update(label=str(e), state="error")
                    st.stop()

                if picked.empty:
                    status.update(
                        label="No relevant items found by the model.", state="error"
                    )
                    st.stop()

                articles_map = load_articles_map("out")
                out = load_or_generate_topic_summary(
                    selected_df=picked,
                    articles_map=articles_map,
                    topic=topic_label,
                    oa_model=oa_model,
                    out_dir="out",
                )

                st.session_state.setdefault("topic_reports", {})[topic_key] = {
                    "summary_md": out.get("summary_md", ""),  # EN
                    "summary_md_uk": None,  # UA (lazy)
                    "items": picked.to_dict(orient="records"),
                    "path": out.get("path"),
                }
                status.update(label="Done.", state="complete")

        # --- Render (and translate-on-toggle) ---
        data = st.session_state.get("topic_reports", {}).get(topic_key)
        if data:
            text_en = data.get("summary_md", "") or ""

            if summary_lang == "Українська":
                if not data.get("summary_md_uk") and text_en:
                    with st.status("Translating summary to Ukrainian…", expanded=False):
                        ua_text, ua_path = load_or_translate_topic_summary(
                            topic=topic_label,
                            en_text=text_en,
                            oa_model=oa_model,
                            out_dir="out",
                        )
                        data["summary_md_uk"] = ua_text
                        data["path_uk"] = ua_path
                        st.session_state["topic_reports"][topic_key] = data
                to_show = data.get("summary_md_uk") or text_en
            else:
                en_path, _ = topic_summary_paths(topic_label, out_dir="out")
                if os.path.exists(en_path) and not text_en:
                    try:
                        with open(en_path, "r", encoding="utf-8") as f:
                            data["summary_md"] = f.read()
                            st.session_state["topic_reports"][topic_key] = dataf
                    except Exception:
                        pass
                to_show = data.get("summary_md", "")

            st.markdown(to_show)

            # Links
            st.markdown("**Посилання:**")
            for it in data.get("items", []):
                dom = str(it.get("regdom") or it.get("domain") or "")
                title = it.get("title_en") or it.get("title") or "(no title)"
                url = it.get("url") or ""
                st.markdown(f"- [{dom.upper()}] [{title}]({url})")

            # Downloads (EN/UA summaries)
            en_path, uk_path = topic_summary_paths(topic_label, out_dir="out")
            if os.path.exists(en_path):
                with open(en_path, "rb") as fh:
                    st.download_button(
                        "Download English summary (.txt)",
                        fh,
                        file_name=os.path.basename(en_path),
                        mime="text/plain",
                        use_container_width=True,
                        key=f"dl_en_{topic_key}",
                    )
            if os.path.exists(uk_path):
                with open(uk_path, "rb") as fh:
                    st.download_button(
                        "Завантажити український підсумок (.txt)",
                        fh,
                        file_name=os.path.basename(uk_path),
                        mime="text/plain",
                        use_container_width=True,
                        key=f"dl_uk_{topic_key}",
                    )

            # --- TTS: per-language mp3 (reused if exists), playback + download ---
            lang_code = "uk" if summary_lang == "Українська" else "en"
            tts_path = topic_tts_path(topic=topic_label, lang=lang_code, out_dir="out")

            col_audio_btn, col_audio_dl = st.columns([1, 1])
            with col_audio_btn:
                make_audio = st.button(
                    "🎧 Generate audio",
                    key=f"tts_btn_{topic_key}_{lang_code}",
                    use_container_width=True,
                )

            if make_audio and to_show.strip():
                try:
                    tts_path = get_or_generate_topic_tts(
                        text=to_show,
                        topic=topic_label,
                        lang=lang_code,
                        voice=voice,
                        out_dir="out",
                    )
                    st.session_state.setdefault("topic_reports", {}).setdefault(
                        topic_key, {}
                    )[f"tts_{lang_code}"] = tts_path
                    st.success("Audio ready.")
                except Exception as e:
                    st.error(f"TTS failed: {e}")

            # Prefer session path; otherwise show if file already exists
            sess_tts = (
                st.session_state.get("topic_reports", {})
                .get(topic_key, {})
                .get(f"tts_{lang_code}")
            )
            if sess_tts and os.path.exists(sess_tts):
                tts_path = sess_tts

            if os.path.exists(tts_path):
                try:
                    with open(tts_path, "rb") as fh:
                        st.audio(fh.read(), format="audio/mp3")
                    with col_audio_dl:
                        with open(tts_path, "rb") as fh:
                            st.download_button(
                                "Download audio (.mp3)",
                                fh,
                                file_name=os.path.basename(tts_path),
                                mime="audio/mpeg",
                                use_container_width=True,
                                key=f"tts_dl_{topic_key}_{lang_code}",
                            )
                except Exception as e:
                    st.caption(f"Audio unavailable: {e}")


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
            stories=top,
            oa_model=oa_model,
            articles_map=articles_map,
            topN=int(topk),  # make Top-N explicit
        )

        os.makedirs("out", exist_ok=True)
        with open("out/report.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        st.session_state["last_report"] = payload
        status.update(label="Done.", state="complete")

        # with tab_top:
        #     (st.session_state["last_report"])
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
def render_all_titles_tab(container):
    """Drop-in replacement for the 'All titles' tab — avoids nested expanders."""
    import os, html
    from urllib.parse import urlparse as _urlparse

    with container:
        cache_path = "out/titles_cache.json"
        if not os.path.exists(cache_path):
            st.info("No titles cache yet. Click **Update cache**.")
            return

        data = load_json(cache_path)
        items = data.get("items", [])
        if not items:
            st.info("Titles cache is empty.")
            return

        # registrable domain helper (eTLD+1)
        try:
            import tldextract
        except Exception:
            tldextract = None

        def registrable_domain(host: str) -> str:
            if not host:
                return ""
            if tldextract:
                ext = tldextract.extract(host)
                return ".".join([p for p in (ext.domain, ext.suffix) if p]) or host
            parts = host.split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else host

        # Ensure domain + regdom on each item
        for it in items:
            host = _urlparse(it.get("url", "")).netloc.lower()
            host = host[4:] if host.startswith("www.") else host
            it["domain"] = it.get("domain") or host
            it["regdom"] = registrable_domain(it["domain"])

        # Group by registrable domain
        by_regdom = {}
        for it in items:
            by_regdom.setdefault(it["regdom"], []).append(it)

        regdoms = sorted(by_regdom.keys())
        sel = st.multiselect(
            "Filter outlets (by site)", options=regdoms, default=regdoms
        )

        for reg in sel:
            arr = by_regdom.get(reg, [])
            if not arr:
                continue

            # Group inside this site by full host
            by_host = {}
            for it in arr:
                by_host.setdefault(it["domain"], []).append(it)

            subdomains = sorted(by_host.keys())
            total_titles = sum(len(v) for v in by_host.values())

            # Single expander per registrable domain (no nested expanders)
            with st.expander(
                f"{reg} — {total_titles} title(s) across {len(subdomains)} subdomain(s)",
                expanded=False,
            ):
                # If only root host present → flat list
                if len(subdomains) == 1 and subdomains[0] == reg:
                    bucket = by_host[reg]
                    for it in sorted(
                        bucket, key=lambda x: x.get("published_at") or "", reverse=True
                    ):
                        url = it.get("url", "")
                        title = it.get("title_en") or it.get("title", "")
                        ts = it.get("published_at", "—")
                        st.markdown(
                            f"<div style='margin:6px 0'>"
                            f"<a href='{html.escape(url)}' target='_blank'>{html.escape(title)}</a><br>"
                            f"<span style='color:gray;font-size:0.9em'>{ts}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    # Multi-subdomain: show sections (no inner expanders)
                    ordered = sorted(
                        subdomains, key=lambda d: (d != reg, d)
                    )  # root first
                    for host in ordered:
                        bucket = by_host[host]
                        pretty = f"{reg} (root)" if host == reg else host
                        st.markdown(f"**{pretty} — {len(bucket)}**")
                        with st.container():
                            for it in sorted(
                                bucket,
                                key=lambda x: x.get("published_at") or "",
                                reverse=True,
                            ):
                                url = it.get("url", "")
                                title = it.get("title_en") or it.get("title", "")
                                ts = it.get("published_at", "—")
                                st.markdown(
                                    f"<div style='margin:6px 0'>"
                                    f"<a href='{html.escape(url)}' target='_blank'>{html.escape(title)}</a><br>"
                                    f"<span style='color:gray;font-size:0.9em'>{ts}</span>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )
                        st.divider()


with tab_titles:
    render_all_titles_tab(tab_titles)

with tab_war:
    render_topic_tab(tab_war, topic_key="war", topic_label="Russo-Ukrainian War")

with tab_tech:
    render_topic_tab(tab_tech, topic_key="tech", topic_label="Technologies")

with tab_econ:
    render_topic_tab(tab_econ, topic_key="econ", topic_label="Economics")
