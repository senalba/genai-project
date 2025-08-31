# src/main.py
import argparse
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, List, cast
from urllib.parse import urlparse

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
    pipeline,
    logging as hf_logging,
)

# ----------------------------- Config -----------------------------
FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    # "https://feeds.reuters.com/reuters/topNews",
    # "https://apnews.com/hub/ap-top-news?utm_source=apnews.com&utm_medium=referral&utm_campaign=ap_rss",
    # "https://www.theguardian.com/world/rss",
    # "http://rss.cnn.com/rss/edition.rss",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    # "https://www.theverge.com/rss/index.xml",
    # "https://feeds.feedburner.com/TechCrunch/",
    "https://www.investing.com/rss/news.rss",
]

# Quiet noisy HF tokenizers-for-fork warning & logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()

# Simple UA for sites that block default clients
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Apple Silicon) NewsImpact/0.1"}


# ----------------------------- Helpers -----------------------------
def safe_head(text: str, n: int = 512) -> str:
    return (text or "")[:n]


def _to_dt(t) -> dt.datetime:
    """Return tz-aware UTC datetime; fallback to 'now' (not epoch)."""
    try:
        ts = pd.to_datetime(t, utc=True)
        if pd.isna(ts):
            raise ValueError
        return ts.to_pydatetime()
    except Exception:
        return dt.datetime.now(dt.timezone.utc)


def _extract_time(entry: Dict[str, Any], feed_meta: Dict[str, Any]) -> dt.datetime:
    cand = (
        entry.get("published_parsed")
        or entry.get("updated_parsed")
        or entry.get("published")
        or entry.get("updated")
        or entry.get("pubDate")
        or feed_meta.get("updated_parsed")
        or feed_meta.get("published_parsed")
    )
    return _to_dt(cand)


def _domain(u: str) -> str:
    d = urlparse(u).netloc.lower()
    return d[4:] if d.startswith("www.") else d


# ----------------------------- Ingest -----------------------------
def ingest(feeds: List[str], max_items: int, max_age_hours: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            feed_meta = cast(Dict[str, Any], getattr(parsed, "feed", {}) or {})
            entries = cast(List[Dict[str, Any]], getattr(parsed, "entries", []) or [])
            src = str(feed_meta.get("title") or url)
            take = max(50, max_items // max(len(feeds), 1) + 1)
            for e in entries[:take]:
                rows.append(
                    {
                        "title": str(e.get("title", "") or "").strip(),
                        "url": str(e.get("link", "") or "").strip(),
                        "source": src,
                        "published_at": _extract_time(e, feed_meta),
                        "summary_hint": str(
                            e.get("summary") or e.get("description") or ""
                        ).strip(),
                    }
                )
        except Exception as ex:
            print(f"[WARN] Failed feed {url}: {ex}")

    df = pd.DataFrame(rows).dropna(subset=["title", "url"])
    if df.empty:
        return df

    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    now = pd.Timestamp.utcnow()
    cutoff = now - pd.Timedelta(hours=max_age_hours)
    df = df[df["published_at"] >= cutoff]
    df = (
        df.sort_values("published_at", ascending=False)
        .head(max_items)
        .reset_index(drop=True)
    )
    return df


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
    """Fetch main article body with light heuristics; skip paywalled/short pages."""
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

        # Prefer dedicated <article>; fallback to densest div/section
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


# ------------------------ HF Summarizer ------------------------
class HFSummarizer:
    def __init__(
        self, model_name: str = "sshleifer/distilbart-cnn-12-6", device: str = "auto"
    ):
        if device == "mps" and torch.backends.mps.is_available():
            dev = "mps"
        elif device == "cpu" or not torch.backends.mps.is_available():
            dev = -1
        else:
            dev = -1
        self.pipe = pipeline("summarization", model=model_name, device=dev)

    def __call__(self, pieces: list[str]) -> str:
        blob = " ".join(p for p in pieces if p).strip()
        if not blob:
            return ""

        # short inputs: don't "summarize" a headline
        if len(blob.split()) < 18:
            return blob

        tok = self.pipe.tokenizer
        ids = tok(blob, truncation=False, return_attention_mask=False)["input_ids"]

        # hard cap to avoid exceeding BART max positions (~1024)
        if len(ids) > 900:
            ids = ids[:900]
            blob = tok.decode(ids, skip_special_tokens=True)

        in_len = len(ids)
        max_new = max(12, min(80, in_len // 2))
        min_len = max(6, min(24, in_len // 6))

        out = self.pipe(
            blob,
            do_sample=False,
            truncation=True,  # ensure tokenizer truncates if needed
            max_new_tokens=max_new,
            min_length=min_len,
            no_repeat_ngram_size=3,
        )
        return out[0]["summary_text"].strip()


# ----------------------- Story assembly -----------------------
def build_stories(df: pd.DataFrame, groups: List[List[int]]) -> List[Dict[str, Any]]:
    sent = FinSent()
    stories: List[Dict[str, Any]] = []
    for g in groups:
        sub = df.iloc[g].sort_values("published_at")
        sample = " ".join(sub["title"].tolist()[:3])
        polarity = sent(sample)

        urls = sub["url"].tolist()
        domains = sorted({_domain(u) for u in urls if u})

        first_seen = pd.to_datetime(sub.iloc[0]["published_at"], utc=True)
        last_seen = pd.to_datetime(sub.iloc[-1]["published_at"], utc=True)

        stories.append(
            {
                "title": sub.iloc[-1]["title"],
                "summary_llm": None,  # filled later
                "sentiment": polarity,
                "mention_count": len(g),  # items
                "mention_count_domains": len(domains),  # distinct domains
                "domains": domains,
                "canonical_url": sub.iloc[0]["url"],  # earliest
                "sources": list(sub["url"].unique()),
                "first_seen": None if pd.isna(first_seen) else first_seen.isoformat(),
                "last_seen": None if pd.isna(last_seen) else last_seen.isoformat(),
            }
        )
    return stories


def select_top(stories: List[Dict[str, Any]], k: int = 5):
    def score(s):
        return (-s["mention_count_domains"], -s["mention_count"], s["first_seen"] or "")

    pos = sorted([s for s in stories if s["sentiment"] == "positive"], key=score)[:k]
    neg = sorted([s for s in stories if s["sentiment"] == "negative"], key=score)[:k]
    return pos, neg


def summarize_selected(
    stories: List[Dict[str, Any]],
    summarizer: str = "hf",
    hf_model: str = "sshleifer/distilbart-cnn-12-6",
    hf_device: str = "auto",
):
    if summarizer == "none":
        for s in stories:
            t = s.get("title", "")
            s["summary_llm"] = t if len(t) <= 200 else (t[:200] + "…")
        return

    if summarizer == "hf":
        hf = HFSummarizer(hf_model, device=hf_device)
        for s in stories:
            body = fetch_article_text(s.get("canonical_url", ""))
            if len(body) < 600:
                # try alternates if canonical is short
                for u in s.get("sources", [])[1:3]:
                    alt = fetch_article_text(u)
                    if len(alt) > len(body):
                        body = alt
                    if len(body) >= 600:
                        break
            parts = [s.get("title", "")]
            if body:
                parts.append(body)
            s["summary_llm"] = hf(parts)
        return

    raise ValueError(f"Unsupported summarizer: {summarizer}")


# -------------------------- Orchestration --------------------------
def run(
    out_path: str,
    max_items: int,
    sim_thr: float,
    topk: int,
    feeds: List[str],
    summarizer: str,
    hf_model: str,
    hf_device: str,
    max_age_hours: int,
):
    print(f"[INFO] Ingesting from {len(feeds)} feeds…")
    df = ingest(feeds, max_items, max_age_hours)
    if df.empty:
        raise RuntimeError("No items ingested from feeds (after recency filter).")
    print(f"[INFO] Items ingested: {len(df)}")

    print("[INFO] Embedding…")
    X = build_embeddings(df)

    print(f"[INFO] Clustering (thr={sim_thr})…")
    groups = greedy_clusters(X, thr=sim_thr)
    print(f"[INFO] Story groups: {len(groups)}")

    print("[INFO] Building stories (no LLM)…")
    stories = build_stories(df, groups)

    print("[INFO] Selecting Top-5 +/- …")
    top_pos, top_neg = select_top(stories, k=topk)

    print(f"[INFO] Summarizing Top-{topk} with {summarizer}…")
    summarize_selected(
        top_pos, summarizer=summarizer, hf_model=hf_model, hf_device=hf_device
    )
    summarize_selected(
        top_neg, summarizer=summarizer, hf_model=hf_model, hf_device=hf_device
    )

    report = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "top_positive": top_pos,
        "top_negative": top_neg,
        "stats": {
            "items_ingested": int(len(df)),
            "groups": int(len(groups)),
            "feeds": feeds,
            "similarity_threshold": sim_thr,
            "summarizer": summarizer,
            "hf_model": hf_model if summarizer == "hf" else None,
            "max_age_hours": max_age_hours,
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] Wrote {out_path}")


# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="News baseline: Top-5 positive/negative stories"
    )
    p.add_argument("--out", default="out/report.json", help="Output JSON path")
    p.add_argument(
        "--max-items", type=int, default=600, help="Max feed items to process"
    )
    p.add_argument("--topk", type=int, default=5, help="Top-K per sentiment")
    p.add_argument(
        "--sim-thr",
        type=float,
        default=0.72,
        help="Cosine similarity threshold for clustering",
    )
    p.add_argument("--feeds", nargs="*", default=None, help="Override feed URLs list")
    p.add_argument(
        "--summarizer",
        choices=["hf", "none"],
        default="hf",
        help="Summarizer to use for Top-K",
    )
    p.add_argument(
        "--hf-model",
        default="sshleifer/distilbart-cnn-12-6",
        help="HF model name for summarization",
    )
    p.add_argument(
        "--hf-device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Device for HF summarizer",
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
        hf_model=args.hf_model,
        hf_device=args.hf_device,
        max_age_hours=args.max_age_hours,
    )
