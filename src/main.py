# src/main.py
import argparse
import datetime as dt
import json
import os
import re
from typing import Any, Dict, List, Tuple, cast
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()


import feedparser
import numpy as np
import pandas as pd
import requests
import torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    logging as hf_logging,
)

# ----------------------------- Config -----------------------------
EN_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://www.investing.com/rss/news.rss",
]

UA_FEEDS = [
    ("Ukrainska Pravda (UA)", "https://www.pravda.com.ua/rss/"),
    ("Ukrainska Pravda (EN)", "https://www.pravda.com.ua/eng/rss/"),
    ("Ekonomichna Pravda", "https://www.epravda.com.ua/rss/"),
    ("European Pravda (EN)", "https://www.eurointegration.com.ua/eng/rss/"),
    ("NV.ua – All news", "https://nv.ua/ukr/rss/allnews.xml"),
    ("Korrespondent.net – All", "https://korrespondent.net/rss"),
    ("RBC-Ukraine – All", "https://www.rbc.ua/static/rss/all.news.xml"),
    ("BBC News Україна", "https://www.bbc.com/ukrainian/index.xml"),
    ("24 Канал (24tv)", "https://24tv.ua/rss/all.xml"),
    ("TSN.ua", "https://tsn.ua/rss"),
    ("ZN.UA", "https://zn.ua/rss"),
    ("Kyiv Independent (EN)", "https://kyivindependent.com/feed"),
    ("Kyiv Post (EN)", "https://www.kyivpost.com/feed"),
    ("Hromadske", "https://hromadske.ua/feed"),
    ("Ukrinform (EN top)", "https://www.ukrinform.net/rss/block-lastnews"),
]

FEEDS: List[str] = EN_FEEDS + [u for _, u in UA_FEEDS]

# Quiet HF logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Apple Silicon) NewsImpact/0.1"}


# ----------------------------- Helpers -----------------------------
def _to_dt(t) -> dt.datetime:
    try:
        ts = pd.to_datetime(t, utc=True)
        if pd.isna(ts):
            raise ValueError
        return ts.to_pydatetime()
    except Exception:
        return dt.datetime.now(dt.timezone.utc)


def _domain(u: str) -> str:
    d = urlparse(u).netloc.lower()
    return d[4:] if d.startswith("www.") else d


# ----------------------------- Ingest -----------------------------
def ingest(
    feeds: List[str], max_items: int, max_age_hours: int, allow_undated: bool = False
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            feed_meta = cast(Dict[str, Any], getattr(parsed, "feed", {}) or {})
            entries = cast(List[Dict[str, Any]], getattr(parsed, "entries", []) or [])
            src = feed_meta.get("title") or url
            take = max(50, max_items // max(len(feeds), 1) + 1)
            for e in entries[:take]:
                title = e.get("title")
                if title is None:
                    td = e.get("title_detail")
                    if isinstance(td, dict):
                        title = td.get("value")
                title = title if isinstance(title, str) else str(title)
                link = e.get("link") or ""
                cand = (
                    e.get("published_parsed")
                    or e.get("updated_parsed")
                    or e.get("published")
                    or e.get("updated")
                    or e.get("pubDate")
                    or feed_meta.get("updated_parsed")
                    or feed_meta.get("published_parsed")
                )
                rows.append(
                    {
                        "title": title,
                        "url": str(link),
                        "source": str(src),
                        "published_at": cand,  # parse below
                        "summary_hint": str(
                            e.get("summary") or e.get("description") or ""
                        ),
                    }
                )
        except Exception as ex:
            print(f"[WARN] Failed feed {url}: {ex}")

    df = pd.DataFrame(rows).dropna(subset=["title", "url"])
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df["has_date"] = ~df["published_at"].isna()

    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=max_age_hours)
    dated = df[df["has_date"] & (df["published_at"] >= cutoff)].copy()

    if allow_undated:
        undated = df[~df["has_date"]].copy()
        undated["published_at"] = pd.Timestamp.utcnow()
        df2 = pd.concat([dated, undated], ignore_index=True)
    else:
        df2 = dated

    return (
        df2.sort_values("published_at", ascending=False)
        .head(max_items)
        .reset_index(drop=True)
    )


# ------------------ Embeddings & Clustering ------------------
def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (df["title"].fillna("") + " — " + df["summary_hint"].fillna("")).tolist()
    X = model.encode(texts, normalize_embeddings=True)
    return X


def greedy_clusters(X: np.ndarray, thr: float = 0.72) -> List[List[int]]:
    n = len(X)
    used = np.zeros(n, dtype=bool)
    groups: List[List[int]] = []
    for i in range(n):
        if used[i]:
            continue
        sims = cosine_similarity(X[i : i + 1], X)[0]
        idx = np.where((sims >= thr) & (~used))[0]
        used[idx] = True
        groups.append(idx.tolist())
    return groups


# -------------------------- Sentiment --------------------------
class FinSent:
    def __init__(self):
        self.labels = ["negative", "neutral", "positive"]
        self.tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        self.mdl = AutoModelForSequenceClassification.from_pretrained(
            "ProsusAI/finbert"
        ).eval()

    def __call__(self, text: str) -> str:
        with torch.no_grad():
            inputs = self.tok((text or "")[:512], return_tensors="pt", truncation=True)
            logits = self.mdl(**inputs).logits[0].cpu().numpy()
        return self.labels[int(np.argmax(logits))]


# ----------------------- Article Extraction -----------------------
def fetch_article_text(url: str, timeout: int = 10, max_chars: int = 6000) -> str:
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        if r.status_code != 200 or not r.text:
            return ""
        soup = BeautifulSoup(r.text, "lxml")

        for tag in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]
        ):
            tag.decompose()

        def longest_paragraphs(nodes):
            best = ""
            for node in nodes:
                txt = " ".join(p.get_text(" ", strip=True) for p in node.find_all("p"))
                if len(txt) > len(best):
                    best = txt
            return best

        article_txt = longest_paragraphs(soup.find_all("article"))
        if len(article_txt) < 600:
            article_txt = max(
                (longest_paragraphs([n]) for n in soup.find_all(["div", "section"])),
                key=len,
                default="",
            )

        article_txt = re.sub(r"\s+", " ", article_txt).strip()
        return article_txt[:max_chars]
    except Exception:
        return ""


# ------------------------ OpenAI Summarizer ------------------------
class OpenAISummarizer:
    def __init__(self, model: str = "gpt-4.1-mini", temperature: float = 0.0):
        from openai import OpenAI  # lazy import

        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def __call__(self, pieces: List[str]) -> str:
        blob = " ".join(p for p in pieces if p).strip()
        if not blob:
            return ""
        prompt = (
            "You are a meticulous news deduper and topic normalizer. "
            "Given an article title and body, write a 2–3 sentence neutral summary. "
            "State the event and key facts. Avoid opinions and adjectives."
        )
        msg = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": blob[:6000]},
        ]
        out = self.client.chat.completions.create(
            model=self.model, messages=msg, temperature=self.temperature
        )
        return (out.choices[0].message.content or "").strip()


# ----------------------- Story assembly -----------------------
def build_stories(df: pd.DataFrame, groups: List[List[int]]) -> List[Dict[str, Any]]:
    """Build clusters where each outlet counts once (dedup by domain)."""
    sent = FinSent()
    stories: List[Dict[str, Any]] = []
    for g in groups:
        sub = df.iloc[g].sort_values("published_at")
        sub = sub.assign(domain=sub["url"].map(_domain))

        # keep earliest URL per domain (so each outlet counts once)
        first_by_domain = sub.sort_values("published_at").drop_duplicates(
            subset=["domain"], keep="first"
        )

        domains = first_by_domain["domain"].tolist()
        urls_by_domain = {
            d: u for d, u in zip(first_by_domain["domain"], first_by_domain["url"])
        }

        sample = " ".join(sub["title"].tolist()[:3])
        polarity = sent(sample)

        first_seen = pd.to_datetime(first_by_domain["published_at"].min(), utc=True)
        last_seen = pd.to_datetime(first_by_domain["published_at"].max(), utc=True)

        canonical_url = first_by_domain.sort_values("published_at").iloc[0]["url"]
        latest_title = sub.iloc[-1]["title"]

        stories.append(
            {
                "title": latest_title,
                "summary_llm": None,
                "sentiment": polarity,
                "mention_count_domains": len(domains),
                "domains": sorted(domains),
                "canonical_url": canonical_url,
                "sources": [urls_by_domain[d] for d in sorted(domains)],
                "first_seen": None if pd.isna(first_seen) else first_seen.isoformat(),
                "last_seen": None if pd.isna(last_seen) else last_seen.isoformat(),
            }
        )
    return stories


def select_top_overall(
    stories: List[Dict[str, Any]], k: int = 5
) -> List[Dict[str, Any]]:
    def score(s):
        return (
            -s.get("mention_count_domains", 0),
            -len(s.get("sources", [])),
            s.get("first_seen") or "",
        )

    return sorted(stories, key=score)[:k]


def summarize_selected(
    stories: List[Dict[str, Any]],
    summarizer: str = "openai",
    oa_model: str = "gpt-4.1-mini",
    articles_map: Dict[str, str] | None = None,
) -> None:
    if summarizer != "openai":
        raise ValueError("Only 'openai' summarizer is supported in this build.")
    oa = OpenAISummarizer(model=oa_model)
    for s in stories:
        body = ""
        if articles_map:
            body = articles_map.get(s.get("canonical_url", ""), "")
            if not body:
                for u in s.get("sources", []):
                    body = articles_map.get(u, "")
                    if body:
                        break
        if not body:
            body = fetch_article_text(s.get("canonical_url", ""))
        parts = [s.get("title", "")]
        if body:
            parts.append(body)
        s["summary_llm"] = oa(parts)


def assemble_report_payload(
    stories: List[Dict[str, Any]],
    summarizer: str,
    oa_model: str,
    articles_map: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    summarize_selected(
        stories,
        summarizer=summarizer,
        oa_model=oa_model,
        articles_map=articles_map,
    )
    items = []
    for s in stories:
        items.append(
            {
                "summary_title": s.get("title", ""),
                "summary": s.get("summary_llm") or s.get("title", ""),
                "links": s.get("sources", []),
                "stats": {
                    "distinct_outlets": s.get("mention_count_domains", 0),
                    "first_seen": s.get("first_seen"),
                    "last_seen": s.get("last_seen"),
                },
            }
        )
    return {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "topN": len(items),
        "items": items,
    }


# --- Cache helpers (titles + articles) ---
def _titles_cache_path(out_dir: str = "out") -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "titles_cache.json")


def _articles_cache_path(out_dir: str = "out") -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "articles_cache.json")


def update_titles_cache(df: pd.DataFrame, out_dir: str = "out") -> Tuple[int, int]:
    path = _titles_cache_path(out_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        items = obj.get("items", [])
    else:
        items = []

    by_url = {it["url"]: it for it in items if "url" in it}

    added = 0
    for r in df.itertuples(index=False):
        url = r.url
        if url in by_url:
            cur = by_url[url]
            if pd.notna(r.published_at):
                cur["published_at"] = pd.to_datetime(
                    r.published_at, utc=True, errors="coerce"
                ).isoformat()
            cur["source"] = str(getattr(r, "source", cur.get("source", "")))
        else:
            by_url[url] = {
                "title": r.title,
                "url": r.url,
                "source": str(getattr(r, "source", "")),
                "domain": _domain(r.url),
                "published_at": pd.to_datetime(
                    r.published_at, utc=True, errors="coerce"
                ).isoformat(),
            }
            added += 1

    items = list(by_url.values())
    obj = {
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "total_items": len(items),
        "items": items,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] Titles cache updated: +{added} -> {len(items)} ({path})")
    return added, len(items)


def update_articles_cache(
    df: pd.DataFrame,
    out_dir: str = "out",
    timeout: int = 10,
    max_chars: int = 6000,
) -> Tuple[int, int]:
    path = _articles_cache_path(out_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        by_url = obj.get("by_url", {})
    else:
        by_url = {}

    new = 0
    for url in df["url"].dropna().unique():
        if url in by_url:
            continue
        body = fetch_article_text(url, timeout=timeout, max_chars=max_chars)
        by_url[url] = {
            "body": body,
            "body_len": len(body),
            "fetched_at": pd.Timestamp.utcnow().isoformat() + "Z",
        }
        new += 1

    obj = {
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "by_url": by_url,
        "total_urls": len(by_url),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[OK] Articles cache updated: +{new} -> {len(by_url)} ({path})")
    return new, len(by_url)


def load_articles_map(out_dir: str = "out") -> Dict[str, str]:
    path = _articles_cache_path(out_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    by_url = obj.get("by_url", {})
    return {u: rec.get("body", "") for u, rec in by_url.items()}


# -------------------------- Orchestration --------------------------
def run(
    out_path: str,
    max_items: int,
    sim_thr: float,
    topk: int,
    feeds: List[str],
    summarizer: str,
    oa_model: str,
    max_age_hours: int,
    titles_dir: str,
):
    print(f"[INFO] Ingesting from {len(feeds)} feeds…")
    df = ingest(feeds, max_items, max_age_hours)
    if df.empty:
        raise RuntimeError("No items ingested from feeds (after recency filter).")
    print(f"[INFO] Items ingested: {len(df)}")

    # snapshots + update caches
    dump_titles(df, feeds, out_dir=titles_dir)
    dump_article_texts(df, out_dir=titles_dir, per_source_limit=None, timeout=10)
    update_titles_cache(df, out_dir=titles_dir)
    update_articles_cache(df, out_dir=titles_dir, timeout=10)

    print("[INFO] Embedding…")
    X = build_embeddings(df)

    print(f"[INFO] Clustering (thr={sim_thr})…")
    groups = greedy_clusters(X, thr=sim_thr)
    print(f"[INFO] Story groups: {len(groups)}")

    print("[INFO] Building stories…")
    stories = build_stories(df, groups)

    print(f"[INFO] Selecting Top-{topk} overall…")
    top = select_top_overall(stories, k=topk)

    print(f"[INFO] Summarizing Top-{topk} with OpenAI…")
    articles_map = load_articles_map(out_dir=titles_dir)
    payload = assemble_report_payload(
        top,
        summarizer=summarizer,
        oa_model=oa_model,
        articles_map=articles_map,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {out_path}")


# ------------------------------ Snapshots ------------------------------
def dump_titles(df: pd.DataFrame, feeds: List[str], out_dir: str = "out") -> str:
    date_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    path = os.path.join(out_dir, f"titles_{date_str}.json")
    os.makedirs(out_dir, exist_ok=True)

    by_source = {}
    for source, g in df.groupby("source"):
        items = g.sort_values("published_at", ascending=False)[
            ["title", "url", "published_at"]
        ].assign(domain=g["url"].map(_domain))
        by_source[source] = [
            {
                "title": r.title,
                "url": r.url,
                "domain": r.domain,
                "published_at": pd.to_datetime(r.published_at, utc=True).isoformat(),
            }
            for r in items.itertuples(index=False)
        ]

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "date": date_str,
        "total_items": int(len(df)),
        "feeds": feeds,
        "by_source": by_source,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {path}")
    return path


def dump_article_texts(
    df: pd.DataFrame,
    out_dir: str = "out",
    per_source_limit: int | None = None,
    timeout: int = 10,
    max_chars: int = 6000,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    date_str = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    path = os.path.join(out_dir, f"articles_{date_str}.json")

    by_source: Dict[str, List[Dict[str, str]]] = {}
    for source, g in df.groupby("source", sort=False):
        items = []
        count = 0
        for r in g.sort_values("published_at", ascending=False).itertuples(index=False):
            if per_source_limit and count >= per_source_limit:
                break
            body = fetch_article_text(r.url, timeout=timeout, max_chars=max_chars)
            items.append(
                {
                    "title": r.title,
                    "url": r.url,
                    "published_at": pd.to_datetime(
                        r.published_at, utc=True, errors="coerce"
                    ).isoformat(),
                    "body_len": len(body),
                    "body": body,
                }
            )
            count += 1
        by_source[str(source)] = items

    payload = {
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "date": date_str,
        "total_items": int(len(df)),
        "by_source": by_source,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {path}")
    return path


# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="News Impact: Top-N by distinct outlets (OpenAI summaries)"
    )
    p.add_argument("--out", default="out/report.json", help="Output JSON path")
    p.add_argument(
        "--titles-dir", default="out", help="Where to write snapshots/caches"
    )
    p.add_argument(
        "--max-items", type=int, default=600, help="Max feed items to process"
    )
    p.add_argument("--topk", type=int, default=5, help="Top-N stories overall")
    p.add_argument(
        "--sim-thr",
        type=float,
        default=0.72,
        help="Cosine similarity threshold for clustering",
    )
    p.add_argument("--feeds", nargs="*", default=None, help="Override feed URLs list")
    p.add_argument(
        "--summarizer", choices=["openai"], default="openai", help="Summarizer backend"
    )
    p.add_argument(
        "--oa-model", default="gpt-4.1-mini", help="OpenAI model for summarization"
    )
    p.add_argument(
        "--max-age-hours",
        type=int,
        default=48,
        help="Ignore items older than this many hours",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feeds = args.feeds if args.feeds else FEEDS
    run(
        out_path=args.out,
        max_items=args.max_items,
        sim_thr=args.sim_thr,
        topk=args.topk,
        feeds=feeds,
        summarizer=args.summarizer,
        oa_model=args.oa_model,
        max_age_hours=args.max_age_hours,
        titles_dir=args.titles_dir,
    )
