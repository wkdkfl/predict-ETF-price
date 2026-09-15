# -*- coding: utf-8 -*-
"""Process ALL financial news headlines with FinBERT and aggregate to daily features.

Input:
  - USD/US_financial_news.csv (23,000 rows)
  - UK/UK_financial_news.csv (36,772 rows)
  - USD/finbert/ (local FinBERT model: BertForSequenceClassification)

Pipeline:
  1) Filter out "market close summary" headlines (e.g. "closed at", "ended the day")
  2) Run FinBERT classifier -> [pos, neg, neu] probability per headline
  3) Run BertModel -> [CLS] embedding (768-dim) per headline
  4) Aggregate by date -> 12 sentiment features + mean embedding
  5) Save:
       - {MARKET}/news_sentiment_daily.csv     (12 features per day)
       - {MARKET}/news_embeddings_daily.npy    (T x 768 matrix)
       - {MARKET}/news_embeddings_dates.csv    (T dates matching the npy)
       - {MARKET}/news_concat_daily.csv        (concatenated headlines for TF-IDF later)
       - {MARKET}/news_per_headline.parquet    (raw per-headline preds for traceability)
"""
from __future__ import annotations

import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
FINBERT_DIR = BASE.parent / "USD" / "finbert"  # finbert/ stays at project root (shared, not duplicated)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32 if DEVICE == "cuda" else 32  # CPU can handle this for short headlines
MAX_LEN = 64   # headlines are short; cuts time vs 128
SEED = 42

# label index -> name (from config.json)
ID2LABEL = {0: "positive", 1: "negative", 2: "neutral"}

# Patterns indicating "market close summary" articles (would leak future info)
CLOSE_PATTERNS = re.compile(
    r"\b(closed at|ended (the )?(day|session|week)|"
    r"(close|closes|closing) (higher|lower|up|down|mixed|flat)|"
    r"finished (higher|lower|up|down|mixed)|"
    r"end of (the )?(day|session|week)|"
    r"after the (close|bell)|"
    r"close: |stocks close|market close|wall street close)\b",
    re.IGNORECASE,
)


def load_news(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def filter_close_headlines(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    mask = df["headline"].str.contains(CLOSE_PATTERNS, na=False)
    removed = int(mask.sum())
    return df[~mask].reset_index(drop=True), removed


@torch.no_grad()
def run_finbert(
    df: pd.DataFrame,
    finbert_dir: Path,
    batch_size: int,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs[N,3], embeddings[N,768]) for all headlines.

    Single forward pass: BertForSequenceClassification exposes BertModel via
    `.bert`, and with output_hidden_states=True we recover the [CLS] embedding
    from the encoder's last_hidden_state without a second model.
    """
    print(f"  loading FinBERT from {finbert_dir} ...", flush=True)
    tok = BertTokenizerFast.from_pretrained(str(finbert_dir))
    clf = (
        BertForSequenceClassification
        .from_pretrained(str(finbert_dir), output_hidden_states=True)
        .to(DEVICE).eval()
    )

    n = len(df)
    probs = np.zeros((n, 3), dtype=np.float32)
    embs = np.zeros((n, 768), dtype=np.float32)

    headlines = df["headline"].tolist()
    t0 = time.time()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = headlines[start:end]
        toks = tok(
            batch,
            padding=True, truncation=True, max_length=max_len,
            return_tensors="pt",
        ).to(DEVICE)

        out = clf(**toks)
        # softmax classifier probs
        probs[start:end] = torch.softmax(out.logits, dim=-1).cpu().numpy()
        # [CLS] from last hidden state of the encoder
        cls = out.hidden_states[-1][:, 0, :].cpu().numpy()
        embs[start:end] = cls

        if start % (batch_size * 20) == 0 or end == n:
            elapsed = time.time() - t0
            rate = end / max(elapsed, 1e-9)
            eta = (n - end) / max(rate, 1e-9)
            print(f"    [{end:6d}/{n}]  rate={rate:5.1f} hl/s  eta={eta/60:5.1f} min", flush=True)
    return probs, embs


def aggregate_daily(df: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """12 daily sentiment features."""
    tmp = df.copy()
    tmp["sent_pos"] = probs[:, 0]
    tmp["sent_neg"] = probs[:, 1]
    tmp["sent_neu"] = probs[:, 2]
    tmp["sent_score"] = tmp["sent_pos"] - tmp["sent_neg"]
    tmp["sent_label"] = probs.argmax(axis=1)  # 0=pos, 1=neg, 2=neu

    grp = tmp.groupby("date")
    agg = pd.DataFrame({
        "news_volume":     grp.size(),
        "sent_pos_mean":   grp["sent_pos"].mean(),
        "sent_neg_mean":   grp["sent_neg"].mean(),
        "sent_neu_mean":   grp["sent_neu"].mean(),
        "sent_score_mean": grp["sent_score"].mean(),
        "sent_score_std":  grp["sent_score"].std().fillna(0.0),
        "sent_max_neg":    grp["sent_neg"].max(),
        "sent_pos_ratio":  grp["sent_label"].apply(lambda s: float((s == 0).mean())),
        "sent_neg_ratio":  grp["sent_label"].apply(lambda s: float((s == 1).mean())),
    }).reset_index()

    agg = agg.sort_values("date").reset_index(drop=True)
    # Rolling / derived features
    agg["sent_score_ma5"]   = agg["sent_score_mean"].rolling(5,  min_periods=1).mean()
    agg["sent_score_ma20"]  = agg["sent_score_mean"].rolling(20, min_periods=1).mean()
    agg["sent_surprise"]    = agg["sent_score_mean"] - agg["sent_score_ma5"]
    agg["sent_momentum_3d"] = agg["sent_score_mean"].rolling(3, min_periods=1).mean()
    agg["sent_vol_5d"]      = agg["sent_score_mean"].rolling(5, min_periods=1).std().fillna(0.0)
    agg["news_volume_chg"]  = agg["news_volume"].pct_change().replace([np.inf, -np.inf], 0).fillna(0.0)
    return agg


def build_concat_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily concatenated headlines for downstream TF-IDF."""
    grp = df.groupby("date")["headline"].apply(lambda s: " || ".join(s.tolist()))
    return grp.reset_index().rename(columns={"headline": "headlines_concat"})


def aggregate_embeddings(df: pd.DataFrame, embs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Mean [CLS] embedding per day."""
    dates = df["date"].values
    uniq = sorted(pd.unique(dates))
    out = np.zeros((len(uniq), embs.shape[1]), dtype=np.float32)
    for i, d in enumerate(uniq):
        mask = dates == d
        out[i] = embs[mask].mean(axis=0)
    return out, uniq


def process_market(news_path: Path, out_dir: Path, label: str):
    print(f"\n{'='*70}\n[{label}] processing {news_path.name}\n{'='*70}", flush=True)
    df = load_news(news_path)
    print(f"  loaded {len(df):,} headlines  ({df['date'].min()} .. {df['date'].max()})")

    df, removed = filter_close_headlines(df)
    print(f"  removed {removed:,} 'market close summary' headlines "
          f"({100*removed/(len(df)+removed):.2f}%)  -> {len(df):,} remaining")

    probs, embs = run_finbert(df, FINBERT_DIR, BATCH_SIZE, MAX_LEN)

    # Save per-headline preds (parquet — fast, small)
    per_hl = df.copy()
    per_hl["sent_pos"] = probs[:, 0]
    per_hl["sent_neg"] = probs[:, 1]
    per_hl["sent_neu"] = probs[:, 2]
    per_hl_path = out_dir / "news_per_headline.parquet"
    try:
        per_hl.to_parquet(per_hl_path, index=False)
    except Exception:
        per_hl_path = out_dir / "news_per_headline.csv"
        per_hl.to_csv(per_hl_path, index=False)
    print(f"  saved -> {per_hl_path.name}  ({len(per_hl):,} rows)")

    # Daily sentiment aggregation
    daily = aggregate_daily(df, probs)
    daily_path = out_dir / "news_sentiment_daily.csv"
    daily.to_csv(daily_path, index=False)
    print(f"  saved -> {daily_path.name}  shape={daily.shape}")

    # Daily concatenated headlines
    concat = build_concat_daily(df)
    concat_path = out_dir / "news_concat_daily.csv"
    concat.to_csv(concat_path, index=False)
    print(f"  saved -> {concat_path.name}  shape={concat.shape}")

    # Daily embeddings (mean of [CLS])
    daily_embs, dates = aggregate_embeddings(df, embs)
    np.save(out_dir / "news_embeddings_daily.npy", daily_embs)
    pd.DataFrame({"date": dates}).to_csv(out_dir / "news_embeddings_dates.csv", index=False)
    print(f"  saved -> news_embeddings_daily.npy  shape={daily_embs.shape}")
    print(f"  saved -> news_embeddings_dates.csv  {len(dates):,} dates")


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print(f"device = {DEVICE}   batch = {BATCH_SIZE}   max_len = {MAX_LEN}")
    process_market(BASE / "USD" / "US_financial_news.csv", BASE / "USD", "US")
    process_market(BASE / "UK" / "UK_financial_news.csv",  BASE / "UK",  "UK")
    print("\n[DONE] all markets processed.")


if __name__ == "__main__":
    main()
