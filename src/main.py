# src/main.py

import argparse
import datetime as dt
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Tuple, cast
from urllib.parse import urlparse
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # load OPENAI_API_KEY from .env early

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

import logging
from openai import OpenAI

LOG = logging.getLogger("newsimpact")


def setup_logging(level: str = "INFO"):
    """Initialize console logging once."""
    # Don’t double-add handlers if Streamlit reloads modules
    if getattr(setup_logging, "_configured", False):
        LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    setup_logging._configured = True
    LOG.debug("Logging initialized at %s", level)


# ----------------------------- Config -----------------------------
EN_FEEDS = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://www.investing.com/rss/news.rss",
    "https://www.theguardian.com/world/rss",
    "https://feeds.reuters.com/reuters/topNews",
    "https://www.theverge.com/rss/index.xml",
]

UA_FEEDS = [
    ("Ukrainska Pravda (UA)", "https://www.pravda.com.ua/rss/"),
    ("Ukrainska Pravda (EN)", "https://www.pravda.com.ua/eng/rss/"),
    ("Ekonomichna Pravda", "https://www.epravda.com.ua/rss/"),
    ("European Pravda (EN)", "https://www.eurointegration.com.ua/eng/rss/"),
    ("NV.ua – All news", "https://nv.ua/ukr/rss/all.xml"),
    ("RBC-Ukraine – All", "https://www.rbc.ua/static/rss/all.ukr.rss.xml"),
    ("BBC News Україна", "https://www.bbc.com/ukrainian/index.xml"),
    ("24 Канал (24tv)", "https://24tv.ua/rss/all.xml"),
    ("TSN.ua", "https://tsn.ua/rss"),
    ("ZN.UA", "https://zn.ua/rss"),
    ("Kyiv Post (EN)", "https://www.kyivpost.com/feed"),
    ("Ukrinform (EN top)", "https://www.ukrinform.net/rss/block-lastnews"),
    ("Censor.NET (UA)", "https://assets.censor.net/rss/censor.net/rss_uk_news.xml"),
]

FEEDS: List[str] = EN_FEEDS + [u for _, u in UA_FEEDS]

# Quiet HF logs
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
hf_logging.set_verbosity_error()

UA_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Apple Silicon) NewsImpact/0.2"}


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env or export it in the shell "
            "(e.g., export OPENAI_API_KEY=sk-...)"
        )
    return OpenAI(api_key=key)


# ----------------------------- Helpers -----------------------------
def _domain(u: str) -> str:
    d = urlparse(u).netloc.lower()
    return d[4:] if d.startswith("www.") else d


# registrable domain (eTLD+1)
try:
    import tldextract
except Exception:
    tldextract = None


def _slugify_topic(topic: str) -> str:
    s = (topic or "topic").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_\-]+", "", s)
    return s or "topic"


def _date_str_utc() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d")


def registrable_domain(host_or_url: str) -> str:
    host = urlparse(host_or_url).netloc if "://" in host_or_url else host_or_url
    if not host:
        return ""
    ext = tldextract.extract(host)
    return ".".join([p for p in [ext.domain, ext.suffix] if p])


def sha1_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def looks_non_english(text: str) -> bool:
    # quick heuristic: if contains Cyrillic or lots of non-ASCII
    if re.search(r"[\u0400-\u04FF]", text or ""):
        return True
    non_ascii = sum(1 for ch in text or "" if ord(ch) > 127)
    return non_ascii > max(6, len(text) // 8)


# --- ingest ---
def ingest(
    feeds: List[str], max_items: int, max_age_hours: int, allow_undated: bool = True
) -> pd.DataFrame:
    LOG.info(
        "Ingest: feeds=%d, max_items=%d, max_age_hours=%d, allow_undated=%s",
        len(feeds),
        max_items,
        max_age_hours,
        allow_undated,
    )
    rows: List[Dict[str, Any]] = []
    for url in feeds:
        try:
            LOG.debug("Fetching feed: %s", url)
            parsed = feedparser.parse(url, request_headers=UA_HEADERS)
            if not getattr(parsed, "entries", None):
                LOG.debug("Feedparser empty, manual GET fallback: %s", url)
                try:
                    r = requests.get(url, headers=UA_HEADERS, timeout=10)
                    r.raise_for_status()
                    parsed = feedparser.parse(r.content)
                except Exception as ex2:
                    LOG.warning("Manual GET failed for %s: %s", url, ex2)

            feed_meta = cast(Dict[str, Any], getattr(parsed, "feed", {}) or {})
            entries = cast(List[Dict[str, Any]], getattr(parsed, "entries", []) or [])
            src = feed_meta.get("title") or _domain(url) or url
            LOG.debug("Parsed feed: source=%s, entries=%d", src, len(entries))

            take = max(50, max_items // max(len(feeds), 1) + 1)
            for e in entries[:take]:
                title = e.get("title")
                if title is None:
                    td = e.get("title_detail")
                    if isinstance(td, dict):
                        title = td.get("value")
                title = title if isinstance(title, str) else str(title or "")

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
                        "published_at": cand,
                        "summary_hint": str(
                            e.get("summary") or e.get("description") or ""
                        ),
                    }
                )
        except Exception as ex:
            LOG.warning("Failed feed %s: %s", url, ex)

    df = pd.DataFrame(rows).dropna(subset=["title", "url"])
    LOG.debug("Ingest produced %d rows before filtering", len(df))
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

    out = (
        df2.sort_values("published_at", ascending=False)
        .head(max_items)
        .reset_index(drop=True)
    )
    LOG.info("Ingest after filters: dated=%d, final=%d", len(dated), len(out))
    return out


def topic_summary_paths(
    topic: str, out_dir: str = "out", date_str: str | None = None
) -> tuple[str, str]:
    """Return (en_path, uk_path) for given topic/date."""
    os.makedirs(out_dir, exist_ok=True)
    date_str = date_str or _date_str_utc()
    slug = _slugify_topic(topic)
    en_path = os.path.join(out_dir, f"summary_{slug}_{date_str}.en.txt")
    uk_path = os.path.join(out_dir, f"summary_{slug}_{date_str}.uk.txt")
    return en_path, uk_path


def _read_text_if_exists(path: str) -> str | None:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return None


def _write_topic_summary_txt(
    topic: str,
    summary_text: str,
    urls: list[str] | None,
    lang: str,
    out_dir: str = "out",
) -> str:
    """Write summary to dated file with language suffix ('.en.txt' or '.uk.txt')."""
    en_path, uk_path = topic_summary_paths(topic, out_dir=out_dir)
    path = en_path if lang.lower().startswith("en") else uk_path
    os.makedirs(out_dir, exist_ok=True)
    body = []
    body.append(f"Topic: {topic}")
    body.append(f"Date (UTC): {_date_str_utc()}")
    body.append(f"Language: {'EN' if lang.lower().startswith('en') else 'UK'}")
    body.append("")
    body.append(summary_text.strip())
    if urls:
        body.append("")
        body.append("Sources:")
        for u in urls:
            body.append(str(u))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(body).strip() + "\n")
    return path


# Minimal translator using shared OpenAI client
def openai_translate_text(
    text: str, target_lang: str = "Ukrainian", oa_model: str = "gpt-4.1-mini"
) -> str:
    client = get_openai_client()
    if not text.strip():
        return ""
    system = "Translate to the requested target language faithfully and concisely. Keep formatting when reasonable."
    user = f"Target language: {target_lang}\n\nText:\n{text}"
    resp = client.chat.completions.create(
        model=oa_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def load_or_generate_topic_summary(
    selected_df: pd.DataFrame,
    articles_map: dict,
    topic: str,
    oa_model: str = "gpt-4.1-mini",
    out_dir: str = "out",
    max_chars_per_doc: int = 6000,
) -> dict:
    """
    If EN summary file for topic/date exists -> load it.
    Else -> generate via OpenAI, save .en.txt, and return.
    """
    en_path, _ = topic_summary_paths(topic, out_dir=out_dir)
    existing = _read_text_if_exists(en_path)
    urls = selected_df["url"].tolist() if not selected_df.empty else []
    if existing:
        return {"topic": topic, "summary_md": existing, "urls": urls, "path": en_path}

    # Generate (reuse your logic from openai_topic_summary, but do not write twice)
    client = get_openai_client()

    def _body(rec):
        if rec is None:
            return ""
        if isinstance(rec, dict):
            return rec.get("body_en") or rec.get("body") or ""
        return str(rec)

    docs = []
    for _, r in selected_df.iterrows():
        url = r["url"]
        title = str(r.get("title_en") or r.get("title") or "")
        dom = str(r.get("regdom") or r.get("domain") or "")
        body = _body(articles_map.get(url)) or title
        docs.append(
            f"### {dom}: {title}\nURL: {url}\n\n{(body or '')[:max_chars_per_doc]}\n"
        )

    if not docs:
        return {
            "topic": topic,
            "summary_md": "_No relevant documents._",
            "urls": [],
            "path": None,
        }

    system = (
        "You are an expert news analyst. Synthesize a comprehensive, neutral summary."
    )
    user = (
        f"Topic: {topic}\n"
        "Write a cohesive summary (~300–500 words) with key facts/dates, what happened, context, and implications. "
        "End with a 'Sources' section listing the URLs.\n\n" + "\n\n---\n\n".join(docs)
    )

    resp = client.chat.completions.create(
        model=oa_model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    summary_en = (resp.choices[0].message.content or "").strip()

    path = _write_topic_summary_txt(
        topic=topic, summary_text=summary_en, urls=urls, lang="en", out_dir=out_dir
    )
    return {"topic": topic, "summary_md": summary_en, "urls": urls, "path": path}


def load_or_translate_topic_summary(
    topic: str,
    en_text: str,
    oa_model: str = "gpt-4.1-mini",
    out_dir: str = "out",
) -> tuple[str, str | None]:
    """
    If UA file exists -> load and return.
    Else -> translate EN -> save .uk.txt -> return.
    Returns (ua_text, ua_path).
    """
    _, uk_path = topic_summary_paths(topic, out_dir=out_dir)
    existing = _read_text_if_exists(uk_path)
    if existing:
        return existing, uk_path

    ua_text = openai_translate_text(en_text, target_lang="Ukrainian", oa_model=oa_model)
    path = _write_topic_summary_txt(
        topic=topic, summary_text=ua_text, urls=None, lang="uk", out_dir=out_dir
    )
    return ua_text, path


# ------------------ Embeddings & Clustering ------------------
def build_embeddings(df: pd.DataFrame) -> np.ndarray:
    # With translation, titles should be English; still prefer title_en if present.
    model = SentenceTransformer("all-MiniLM-L6-v2")
    title_col = (
        "title_en"
        if "title_en" in df.columns and df["title_en"].notna().any()
        else "title"
    )
    texts = df[title_col].fillna("").astype(str).tolist()
    X = model.encode(texts, normalize_embeddings=True)
    return X


# --- greedy_clusters ---
def greedy_clusters(X: np.ndarray, thr: float = 0.72) -> List[List[int]]:
    n = len(X)
    LOG.debug("Clustering: greedy, thr=%.2f, items=%d", thr, n)
    used = np.zeros(n, dtype=bool)
    groups: List[List[int]] = []
    for i in range(n):
        if used[i]:
            continue
        sims = cosine_similarity(X[i : i + 1], X)[0]
        idx = np.where((sims >= thr) & (~used))[0]
        used[idx] = True
        groups.append(idx.tolist())
    LOG.debug("Clustering produced %d groups", len(groups))
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


# --- fetch_article_text ---
def fetch_article_text(url: str, timeout: int = 20, max_chars: int = 8000) -> str:
    LOG.debug("Fetch article: %s", url)
    try:
        r = requests.get(url, headers=UA_HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code != 200 or not r.text:
            LOG.debug("HTTP %s or empty body for %s", r.status_code, url)
            return ""
        soup = BeautifulSoup(r.text, "lxml")

        for tag in soup(
            ["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]
        ):
            tag.decompose()

        def extract_from(nodes):
            best = ""
            for node in nodes:
                ps = [p.get_text(" ", strip=True) for p in node.find_all("p")]
                txt = " ".join(ps) if ps else node.get_text(" ", strip=True)
                if len(txt) > len(best):
                    best = txt
            return best

        candidates = []
        candidates += soup.find_all("article")
        candidates += soup.select('[role="main"]')
        candidates += soup.select('[itemprop="articleBody"]')
        candidates += soup.select('div[class*="article"], section[class*="article"]')
        candidates += soup.select('div[class*="content"], section[class*="content"]')

        article_txt = extract_from(candidates) or extract_from(soup.find_all("div"))
        article_txt = re.sub(r"\s+", " ", (article_txt or "")).strip()

        used_amp = False
        if len(article_txt) < 500:
            amp = soup.find("link", rel=lambda v: v and "amphtml" in v.lower())
            if amp and amp.get("href"):
                amp_url = amp["href"]
                if amp_url.startswith("//"):
                    amp_url = "https:" + amp_url
                ra = requests.get(amp_url, headers=UA_HEADERS, timeout=timeout)
                if ra.status_code == 200 and ra.text:
                    sa = BeautifulSoup(ra.text, "lxml")
                    for tag in sa(
                        [
                            "script",
                            "style",
                            "noscript",
                            "header",
                            "footer",
                            "nav",
                            "aside",
                            "form",
                        ]
                    ):
                        tag.decompose()
                    amp_candidates = []
                    amp_candidates += sa.find_all("article")
                    amp_candidates += sa.select('[itemprop="articleBody"]')
                    amp_candidates += sa.find_all("div")
                    amp_txt = extract_from(amp_candidates)
                    amp_txt = re.sub(r"\s+", " ", (amp_txt or "")).strip()
                    if len(amp_txt) > len(article_txt):
                        article_txt = amp_txt
                        used_amp = True

        article_txt = (article_txt or "")[:max_chars]
        LOG.debug(
            "Article extracted: chars=%d amp=%s url=%s", len(article_txt), used_amp, url
        )
        return article_txt
    except Exception as ex:
        LOG.debug("Fetch article failed for %s: %s", url, ex)
        return ""


# ------------------------ OpenAI Clients ------------------------
class OpenAIBase:
    def __init__(self):
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)


# OpenAI translator
class OpenAITranslator:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        self.client = get_openai_client()

    def translate(self, texts: list[str]) -> list[str]:
        outs = []
        for t in texts:
            if not t:
                outs.append("")
                continue
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "Translate to English. Keep meaning; do not add info.",
                    },
                    {"role": "user", "content": t[:4000]},
                ],
            )
            outs.append((resp.choices[0].message.content or "").strip())
        return outs


class OpenAISummarizer:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.model = model
        self.client = get_openai_client()

    def summarize(self, pieces: list[str]) -> str:
        blob = " ".join(p for p in pieces if p).strip()
        if not blob:
            return ""
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise news summarizer. Write 2–3 neutral sentences.",
                },
                {"role": "user", "content": blob[:6000]},
            ],
        )
        return (resp.choices[0].message.content or "").strip()


# --- translate_needed_titles ---
def translate_needed_titles(
    out_dir: str, model: str = "gpt-4.1-mini"
) -> Tuple[int, int]:
    path = _titles_cache_path(out_dir)
    if not os.path.exists(path):
        LOG.info("Titles cache absent, nothing to translate.")
        return (0, 0)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    items: List[Dict[str, Any]] = obj.get("items", [])
    need_idx: List[int] = []
    payloads: List[str] = []

    for i, it in enumerate(items):
        title = it.get("title", "") or ""
        src_sha = sha1_text(title)
        ok = (
            it.get("title_en")
            and it.get("title_en_sha1") == src_sha
            and it.get("translator_model") == model
        )
        if ok or not looks_non_english(title):
            if not it.get("title_en"):
                it["title_en"] = title
                it["title_en_sha1"] = src_sha
                it["translator_model"] = model
                it["translated_at"] = pd.Timestamp.utcnow().isoformat() + "Z"
            continue
        need_idx.append(i)
        payloads.append(title)

    LOG.info("Titles needing translation: %d (model=%s)", len(payloads), model)
    if not payloads:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return (0, len(items))

    tx = OpenAITranslator(model=model)
    outs = tx.translate(payloads)

    for j, i in enumerate(need_idx):
        en = outs[j]
        title = items[i].get("title", "")
        items[i]["title_en"] = en or title
        items[i]["title_en_sha1"] = sha1_text(title)
        items[i]["translator_model"] = model
        items[i]["translated_at"] = pd.Timestamp.utcnow().isoformat() + "Z"

    obj["generated_at"] = pd.Timestamp.utcnow().isoformat() + "Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    LOG.info("Titles translated (new): %d / total items: %d", len(need_idx), len(items))
    return (len(need_idx), len(items))


# --- translate_needed_bodies ---
def translate_needed_bodies(
    out_dir: str, model: str = "gpt-4.1-mini", max_chars: int = 6000
) -> Tuple[int, int]:
    path = _articles_cache_path(out_dir)
    if not os.path.exists(path):
        LOG.info("Articles cache absent, nothing to translate.")
        return (0, 0)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    by_url: Dict[str, Dict[str, Any]] = obj.get("by_url", {})
    urls: List[str] = []
    payloads: List[str] = []

    for u, rec in by_url.items():
        body = rec.get("body", "") or ""
        if not body:
            continue
        src_sha = sha1_text(body)
        ok = (
            rec.get("body_en")
            and rec.get("body_en_sha1") == src_sha
            and rec.get("translator_model") == model
        )
        if ok or not looks_non_english(body):
            if not rec.get("body_en"):
                rec["body_en"] = body
                rec["body_en_sha1"] = src_sha
                rec["translator_model"] = model
                rec["translated_at"] = pd.Timestamp.utcnow().isoformat() + "Z"
            continue
        urls.append(u)
        payloads.append(body[:max_chars])

    LOG.info("Bodies needing translation: %d (model=%s)", len(payloads), model)
    if not payloads:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return (0, len(by_url))

    tx = OpenAITranslator(model=model)
    outs = tx.translate(payloads)

    for j, u in enumerate(urls):
        en = outs[j]
        body = by_url[u].get("body", "")
        by_url[u]["body_en"] = en or body
        by_url[u]["body_en_sha1"] = sha1_text(body)
        by_url[u]["translator_model"] = model
        by_url[u]["translated_at"] = pd.Timestamp.utcnow().isoformat() + "Z"

    obj["generated_at"] = pd.Timestamp.utcnow().isoformat() + "Z"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    LOG.info("Bodies translated (new): %d / total urls: %d", len(urls), len(by_url))
    return (len(urls), len(by_url))


# --- build_stories ---
def build_stories(df: pd.DataFrame, groups: List[List[int]]) -> List[Dict[str, Any]]:
    sent = FinSent()
    stories: List[Dict[str, Any]] = []
    for g in groups:
        sub = df.iloc[g].sort_values("published_at")
        title_col = "title_en" if "title_en" in sub.columns else "title"

        sub = sub.assign(host=sub["url"].map(_domain))
        sub = sub.assign(regdom=sub["host"].map(registrable_domain))

        first_by_regdom = sub.sort_values("published_at").drop_duplicates(
            subset=["regdom"], keep="first"
        )

        regdoms = first_by_regdom["regdom"].tolist()
        urls_by_regdom = {
            d: u for d, u in zip(first_by_regdom["regdom"], first_by_regdom["url"])
        }

        sample = " ".join(sub[title_col].astype(str).tolist()[:3])
        polarity = sent(sample)

        first_seen = pd.to_datetime(first_by_regdom["published_at"].min(), utc=True)
        last_seen = pd.to_datetime(first_by_regdom["published_at"].max(), utc=True)

        canonical_url = first_by_regdom.sort_values("published_at").iloc[0]["url"]
        latest_title = sub[title_col].iloc[-1]

        story = {
            "title": latest_title,
            "summary_llm": None,
            "sentiment": polarity,
            "mention_count_domains": len(regdoms),
            "domains": sorted(regdoms),
            "canonical_url": canonical_url,
            "sources": [urls_by_regdom[d] for d in sorted(regdoms)],
            "first_seen": None if pd.isna(first_seen) else first_seen.isoformat(),
            "last_seen": None if pd.isna(last_seen) else last_seen.isoformat(),
        }
        LOG.debug(
            "Story built: outlets=%d, first_seen=%s, last_seen=%s, title=%s",
            story["mention_count_domains"],
            story["first_seen"],
            story["last_seen"],
            str(latest_title)[:80],
        )
        stories.append(story)
    LOG.info("Built %d stories", len(stories))
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


# --- helper for summarization body selection ---
def _get_article_text_for_story(s: Dict[str, Any], articles_map: Dict[str, Any]) -> str:
    def body_of(url: str) -> str:
        rec = articles_map.get(url, "")
        if isinstance(rec, dict):
            return rec.get("body_en") or rec.get("body") or ""
        return rec or ""

    body = body_of(s.get("canonical_url", ""))
    if len(body) < 600:
        for u in s.get("sources", [])[1:3]:
            alt = body_of(u)
            if len(alt) > len(body):
                body = alt
            if len(body) >= 600:
                break
    LOG.debug("Selected article text for summary: chars=%d", len(body))
    return body


def summarize_selected(
    stories: List[Dict[str, Any]],
    oa_model: str = "gpt-4.1-mini",
    articles_map: Dict[str, Any] | None = None,
) -> None:
    sm = OpenAISummarizer(model=oa_model)
    for s in stories:
        title = s.get("title", "")
        body = ""
        if articles_map:
            body = _get_article_text_for_story(s, articles_map)
        if not body:
            body = fetch_article_text(s.get("canonical_url", ""))
        parts = [title]
        if body:
            parts.append(body)
        s["summary_llm"] = sm.summarize(parts)


# --- assemble_report_payload ---
def assemble_report_payload(
    stories: List[Dict[str, Any]],
    oa_model: str,
    articles_map: Dict[str, Any] | None = None,
    topN: int = 5,
) -> Dict[str, Any]:
    LOG.info("Assembling report payload: topN=%d", topN)
    summarize_selected(stories, oa_model=oa_model, articles_map=articles_map)
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
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "topN": topN,
        "items": items,
    }
    LOG.debug("Report payload items=%d", len(items))
    return payload


# --- Cache helpers (titles + articles) ---
def _titles_cache_path(out_dir: str = "out") -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "titles_cache.json")


def _articles_cache_path(out_dir: str = "out") -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "articles_cache.json")


# --- update_titles_cache ---
def update_titles_cache(df: pd.DataFrame, out_dir: str = "out") -> Tuple[int, int]:
    path = _titles_cache_path(out_dir)
    LOG.debug("Updating titles cache at %s", path)
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
            if "domain" not in cur or not cur["domain"]:
                cur["domain"] = _domain(url)
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
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    LOG.info("Titles cache write: +%d -> %d (%s)", added, len(items), path)
    return added, len(items)


# --- update_articles_cache ---
def update_articles_cache(
    df: pd.DataFrame,
    out_dir: str = "out",
    timeout: int = 20,
    max_chars: int = 8000,
    refetch_short: bool = True,
    min_len: int = 400,
) -> Tuple[int, int]:
    path = _articles_cache_path(out_dir)
    LOG.debug("Updating articles cache at %s", path)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        by_url = obj.get("by_url", {})
    else:
        by_url = {}

    new_or_refetched = 0
    for url in df["url"].dropna().unique():
        rec = by_url.get(url)
        should_fetch = rec is None
        if rec and refetch_short:
            cur_len = int(rec.get("body_len", 0) or 0)
            if cur_len < min_len:
                should_fetch = True

        if not should_fetch:
            continue

        LOG.debug(
            "Fetching article body (%s): %s",
            "new" if rec is None else "refetch<min_len",
            url,
        )
        body = fetch_article_text(url, timeout=timeout, max_chars=max_chars)
        by_url[url] = {
            "body": body,
            "body_len": len(body),
            "fetched_at": pd.Timestamp.utcnow().isoformat() + "Z",
        }
        LOG.debug("Saved article body: len=%d url=%s", len(body), url)
        new_or_refetched += 1

    obj = {
        "generated_at": pd.Timestamp.utcnow().isoformat() + "Z",
        "by_url": by_url,
        "total_urls": len(by_url),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    LOG.info(
        "Articles cache write: +/↻%d -> %d (%s)", new_or_refetched, len(by_url), path
    )
    return new_or_refetched, len(by_url)


def load_articles_map(out_dir: str = "out", extended: bool = True) -> Dict[str, Any]:
    """Return {url: record}; record may be 'body' string or dict with 'body','body_en'."""
    path = _articles_cache_path(out_dir)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    by_url = obj.get("by_url", {})
    if extended:
        return by_url
    # legacy map
    return {u: rec.get("body", "") for u, rec in by_url.items()}


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


def get_recent_titles_df(out_dir: str = "out", max_age_hours: int = 48) -> pd.DataFrame:
    path = _titles_cache_path(out_dir)
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=["title", "title_en", "url", "domain", "published_at"]
        )
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    df = pd.DataFrame(obj.get("items", []))
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=int(max_age_hours))
    df = df[df["published_at"] >= cutoff].copy()
    if "title_en" not in df.columns or not df["title_en"].notna().any():
        df["title_en"] = df["title"]
    if "domain" not in df.columns or not df["domain"].notna().any():
        df["domain"] = df["url"].map(_domain)
    df["regdom"] = df["domain"].map(registrable_domain)
    return df.sort_values("published_at", ascending=False).reset_index(drop=True)


def get_recent_titles_df(out_dir: str = "out", max_age_hours: int = 48) -> pd.DataFrame:
    path = _titles_cache_path(out_dir)
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=["title", "title_en", "url", "domain", "published_at"]
        )
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    df = pd.DataFrame(obj.get("items", []))
    if df.empty:
        return df
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=int(max_age_hours))
    df = df[df["published_at"] >= cutoff].copy()
    if "title_en" not in df.columns or not df["title_en"].notna().any():
        df["title_en"] = df["title"]
    if "domain" not in df.columns or not df["domain"].notna().any():
        df["domain"] = df["url"].map(_domain)
    df["regdom"] = df["domain"].map(registrable_domain)
    return df.sort_values("published_at", ascending=False).reset_index(drop=True)


def openai_select_by_topic(
    tdf: pd.DataFrame,
    topic: str,
    oa_model: str = "gpt-4.1-mini",
    max_titles: int = 400,
) -> pd.DataFrame:
    client = get_openai_client()
    rows = tdf.head(int(max_titles)).reset_index(drop=True)

    lines = []
    for i, r in rows.iterrows():
        title = str(r.get("title_en") or r.get("title") or "")[:220]
        dom = str(r.get("regdom") or r.get("domain") or "")
        lines.append(f"{i+1}. [{dom}] {title}")

    system = (
        "You are a precise news curator. Select ONLY items relevant to the given topic. "
        "Return a pure JSON array of integers (IDs), nothing else."
    )
    user = (
        f"Topic: {topic}\n"
        "From the numbered list below, select ALL items that clearly match the topic. "
        "Prefer concrete, on-topic headlines; avoid generic or unrelated items.\n\n"
        + "\n".join(lines)
        + "\n\nOutput: JSON array of selected IDs, e.g. [2,7,13]"
    )

    resp = client.chat.completions.create(
        model=oa_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    txt = (resp.choices[0].message.content or "").strip()
    try:
        sel_ids = json.loads(txt)
        if not isinstance(sel_ids, list):
            sel_ids = []
    except Exception:
        import re

        sel_ids = [int(x) for x in re.findall(r"\d+", txt)]
    sel_ids = {i for i in sel_ids if 1 <= int(i) <= len(rows)}

    picked = rows.iloc[[i - 1 for i in sel_ids]].copy()
    if not picked.empty:
        picked = picked.sort_values("published_at", ascending=False)
        picked = picked.drop_duplicates(subset="regdom", keep="first").reset_index(
            drop=True
        )
    return picked


def openai_topic_summary(
    selected_df: pd.DataFrame,
    articles_map: dict,
    topic: str,
    oa_model: str = "gpt-4.1-mini",
    max_chars_per_doc: int = 6000,
) -> dict:
    client = get_openai_client()

    def _body(rec):
        if rec is None:
            return ""
        if isinstance(rec, dict):
            return rec.get("body_en") or rec.get("body") or ""
        return str(rec)

    docs = []
    urls = []
    for _, r in selected_df.iterrows():
        url = r["url"]
        urls.append(url)
        title = str(r.get("title_en") or r.get("title") or "")
        dom = str(r.get("regdom") or r.get("domain") or "")
        body = _body(articles_map.get(url)) or title
        docs.append(f"### {dom}: {title}\nURL: {url}\n\n{body[:max_chars_per_doc]}\n")

    if not docs:
        return {
            "topic": topic,
            "summary_md": "_Немає релевантних матеріалів._",
            "urls": [],
        }

    system = (
        "You are an expert news analyst. Synthesize a comprehensive, neutral summary.\n"
        "Be factual, avoid repetition, attribute carefully, avoid speculation."
    )
    user = (
        f"Topic: {topic}\n"
        "Using the documents below, write a cohesive summary (≈300–500 words): "
        "ключові факти/дати, що сталося, контекст, наслідки. "
        "Наприкінці додай розділ 'Джерела' зі списком URL.\n\n"
        + "\n\n---\n\n".join(docs)
    )

    resp = client.chat.completions.create(
        model=oa_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    summary = (resp.choices[0].message.content or "").strip()
    return {"topic": topic, "summary_md": summary, "urls": urls}


# --- run (CLI orchestration) ---
def run(
    out_path: str,
    max_items: int,
    sim_thr: float,
    topk: int,
    feeds: List[str],
    oa_model: str,
    max_age_hours: int,
    titles_dir: str,
    translate_bodies: bool,
):
    LOG.info(
        "Run: feeds=%d, topk=%d, sim_thr=%.2f, max_age_hours=%d",
        len(feeds),
        topk,
        sim_thr,
        max_age_hours,
    )

    df = ingest(feeds, max_items, max_age_hours, allow_undated=True)
    if df.empty:
        LOG.error("No items ingested; abort.")
        raise RuntimeError("No items ingested from feeds (after recency filter).")
    LOG.info("Items ingested: %d", len(df))

    dump_titles(df, feeds, out_dir=titles_dir)
    dump_article_texts(df, out_dir=titles_dir, per_source_limit=None, timeout=10)

    added_titles, total_titles = update_titles_cache(df, out_dir=titles_dir)
    added_bodies, total_bodies = update_articles_cache(
        df, out_dir=titles_dir, timeout=10
    )
    LOG.info(
        "Caches updated: titles +%d/%d, articles +/↻%d/%d",
        added_titles,
        total_titles,
        added_bodies,
        total_bodies,
    )

    added_en_titles, total_titles = translate_needed_titles(
        out_dir=titles_dir, model=oa_model
    )
    LOG.info("Title translations (new): %d", added_en_titles)
    if translate_bodies:
        added_en_bodies, total_bodies = translate_needed_bodies(
            out_dir=titles_dir, model=oa_model
        )
        LOG.info("Body translations (new): %d", added_en_bodies)

    tpath = _titles_cache_path(titles_dir)
    with open(tpath, "r", encoding="utf-8") as f:
        tcache = json.load(f)
    items = tcache.get("items", [])
    tdf = pd.DataFrame(items)
    tdf["published_at"] = pd.to_datetime(tdf["published_at"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=int(max_age_hours))
    adf = tdf[tdf["published_at"] >= cutoff].copy()
    if adf.empty:
        adf = tdf.copy()

    if "summary_hint" not in adf.columns:
        adf["summary_hint"] = ""
    if "source" not in adf.columns:
        adf["source"] = adf.get("domain", "")

    LOG.info("Embedding %d titles…", len(adf))
    X = build_embeddings(adf)

    LOG.info("Clustering (thr=%.2f)…", sim_thr)
    groups = greedy_clusters(X, thr=sim_thr)
    LOG.info("Story groups: %d", len(groups))

    LOG.info("Building stories…")
    stories = build_stories(adf, groups)

    LOG.info("Selecting Top-%d overall…", topk)
    top = select_top_overall(stories, k=topk)

    LOG.info("Summarizing Top-%d with %s…", topk, oa_model)
    articles_map = load_articles_map(out_dir=titles_dir, extended=True)
    payload = assemble_report_payload(
        top,
        oa_model=oa_model,
        articles_map=articles_map,
        topN=topk,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    LOG.info("Report written → %s", out_path)


# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="News Impact: cache → translate → cluster → summarize (Top-N by outlets)"
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
        default=0.75,
        help="Cosine similarity threshold for clustering",
    )
    p.add_argument("--feeds", nargs="*", default=None, help="Override feed URLs list")
    p.add_argument(
        "--oa-model",
        default="gpt-4.1-mini",
        help="OpenAI model (translator & summarizer)",
    )
    p.add_argument(
        "--max-age-hours",
        type=int,
        default=48,
        help="Ignore items older than this many hours",
    )
    p.add_argument(
        "--translate-bodies",
        action="store_true",
        help="Also translate article bodies (costly)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console log level",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feeds = args.feeds if args.feeds else FEEDS
    setup_logging(args.log_level)
    run(
        out_path=args.out,
        max_items=args.max_items,
        sim_thr=args.sim_thr,
        topk=args.topk,
        feeds=feeds,
        oa_model=args.oa_model,
        max_age_hours=args.max_age_hours,
        titles_dir=args.titles_dir,
        translate_bodies=args.translate_bodies,
    )
