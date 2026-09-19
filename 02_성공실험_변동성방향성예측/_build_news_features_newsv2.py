# -*- coding: utf-8 -*-
"""신규 일별 경제 뉴스(공통데이터/) -> FinBERT 피처 -> 기존 research_enhanced 의 뉴스 열 교체 ("newsv2").

왜 필요한가
-----------
`{US,UK}_research_enhanced.csv` 와 `news_embeddings_daily_aligned.npy` 를 만든 병합 단계 스크립트가
저장소에 없다(읽기만 한다). 그래서 기존 산출물의 규약을 그대로 따라 재구성했다.

  * 행 = 달력일(비거래일 포함, 이후 `_trading_days.filter_trading_days` 가 거래일만 남김)
  * 뉴스 열 16개 = 일별 감성 통계 15개 + `news_available`  (열 순서 유지)
  * 임베딩 = 행과 정렬된 (행수 x 768) [CLS] 평균

기존 방식과 달라지는 점 (의도적)
--------------------------------
  1. 전방 보간 없음: 기존에는 2017년 이후 221일(US)/713일(UK)의 값이 다음 뉴스일까지 이월됐고
     2014~2016 은 전부 0 이었다. 신규 데이터는 거래일마다 새 값이다.
  2. 비거래일(주말·공휴일) 뉴스는 **다음 거래일**에 합산한다. 그대로 두면 토·일 뉴스가 거래일 필터에서
     버려진다. 거래일 t 행에는 "직전 거래일 이후 ~ t 일" 에 발행된 뉴스가 들어간다.
  3. 롤링 피처(ma5/ma20/surprise/momentum/vol_5d/news_volume_chg)는 거래일 시계열 위에서 계산한다.
  4. `news_available` = 그 거래일에 실제로 뉴스가 있었는지 (신규 데이터에서는 거의 항상 1).

시점 정합성: 뉴스 열은 기존과 같이 t 행에 shift 없이 들어가고(금융 피처만 t-1), 타깃은 t+1 이후 수익률이므로
t 일까지 발행된 뉴스는 예측 시점에 이미 알려진 정보다.

FinBERT 처리(모델·max_len·배치·장 마감 요약 필터)는 `_build_multi_headline_features.py` 를 그대로 재사용한다.

사용법
------
  python _build_news_features_newsv2.py --validate    # 기존 헤드라인 300건으로 FinBERT 재현성만 점검
  python _build_news_features_newsv2.py               # 전체 생성 (CPU 에서 수십 분)
  python _build_news_features_newsv2.py --reuse-cache # 캐시된 헤드라인별 FinBERT 결과 재사용
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import _build_multi_headline_features as bm  # noqa: E402
from _trading_days import trading_day_mask  # noqa: E402

NEWS_ROOT = BASE.parent / "공통데이터"
CACHE = NEWS_ROOT / "_news_cache"
MARKETS = {
    "US": {"sub": "USD", "news": NEWS_ROOT / "USD" / "US_economic_news_daily.csv"},
    "UK": {"sub": "UK", "news": NEWS_ROOT / "UK" / "UK_economic_news_daily.csv"},
}
AGG_COLS = [
    "news_volume", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean", "sent_score_mean", "sent_score_std",
    "sent_max_neg", "sent_pos_ratio", "sent_neg_ratio", "sent_score_ma5", "sent_score_ma20",
    "sent_surprise", "sent_momentum_3d", "sent_vol_5d", "news_volume_chg",
]
NEWS_COLS = AGG_COLS + ["news_available"]
SFX = "_newsv2"


def validate_against_old(n: int = 300) -> bool:
    """기존에 저장된 헤드라인별 확률과 새 환경의 FinBERT 출력이 일치하는지 점검 (모델·토크나이저 동일성)."""
    old = pd.read_csv(BASE / "USD" / "news_per_headline.csv")
    old = old.sample(n=n, random_state=0).reset_index(drop=True)
    probs, _ = bm.run_finbert(old, bm.FINBERT_DIR, bm.BATCH_SIZE, bm.MAX_LEN)
    diff = np.abs(probs - old[["sent_pos", "sent_neg", "sent_neu"]].values)
    print("  FinBERT 재현성: %d건, 최대 |Δp| = %.2e, 평균 |Δp| = %.2e" % (n, diff.max(), diff.mean()))
    ok = diff.max() < 5e-3
    print("  ->", "일치 (모델·토크나이저 동일)" if ok else "불일치! 모델/토크나이저 확인 필요")
    return ok


def session_dates(cal: pd.Series, is_trading: np.ndarray, dates: pd.Series) -> pd.Series:
    """각 날짜를 (그 날 이후 첫 거래일)로 매핑. 프레임 마지막 거래일 이후 날짜는 NaT."""
    trading = np.sort(cal[is_trading].values)
    idx = np.searchsorted(trading, dates.values, side="left")
    out = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(trading)
    out[ok] = trading[idx[ok]]
    return pd.Series(out, index=dates.index)


def build_market(mc: str, reuse_cache: bool):
    cfg = MARKETS[mc]
    sub = cfg["sub"]
    print(f"\n{'=' * 72}\n[{mc}] {cfg['news'].name}\n{'=' * 72}", flush=True)

    # ---- 1) 헤드라인 적재 + 장 마감 요약 필터 (기존 파이프라인과 동일)
    df = bm.load_news(cfg["news"])
    df = df.sort_values(["date", "day_rank"]).reset_index(drop=True)
    n0 = len(df)
    df, removed = bm.filter_close_headlines(df)
    print(f"  헤드라인 {n0:,}건 ({df['date'].min()} ~ {df['date'].max()}), 장 마감 요약 패턴 제거 {removed}건 -> {len(df):,}건")

    # ---- 2) FinBERT (헤드라인별 확률 + [CLS] 임베딩), 캐시 가능
    cache = CACHE / f"{mc}_headline_finbert{SFX}.npz"
    if reuse_cache and cache.exists():
        z = np.load(cache)
        probs, embs = z["probs"], z["embs"]
        assert len(probs) == len(df), "캐시 행 수 불일치 — --reuse-cache 없이 다시 실행"
        print(f"  캐시 사용: {cache.name}")
    else:
        t0 = time.time()
        probs, embs = bm.run_finbert(df, bm.FINBERT_DIR, bm.BATCH_SIZE, bm.MAX_LEN)
        print(f"  FinBERT 완료 ({(time.time() - t0) / 60:.1f}분)")
        CACHE.mkdir(exist_ok=True)
        np.savez(cache, probs=probs, embs=embs)

    # ---- 3) 시장 프레임(기존 research_enhanced)과 거래일 달력
    frame = pd.read_csv(BASE / sub / f"{mc}_research_enhanced.csv")
    frame["Date"] = pd.to_datetime(frame["Date"])
    assert frame["Date"].is_monotonic_increasing, "research_enhanced 의 Date 가 정렬돼 있지 않음"
    is_tr = trading_day_mask(frame, mc).values
    print(f"  프레임 {len(frame)}행, 거래일 {int(is_tr.sum())}일 ({frame['Date'].min().date()} ~ {frame['Date'].max().date()})")

    # ---- 4) 비거래일 뉴스 -> 다음 거래일 (세션 날짜)
    d = pd.to_datetime(df["date"])
    sess = session_dates(frame["Date"], is_tr, d)
    keep = sess.notna().values
    n_drop = int((~keep).sum())
    df = df.loc[keep].reset_index(drop=True)
    probs, embs = probs[keep], embs[keep]
    df["session_date"] = pd.to_datetime(sess[keep].values).strftime("%Y-%m-%d")
    shifted = int((df["session_date"] != df["date"]).sum())
    print(f"  세션 매핑: 프레임 마지막 거래일 이후 {n_drop}건 제외, 다음 거래일로 이월 {shifted:,}건 ({100 * shifted / len(df):.1f}%)")

    # ---- 5) 세션(거래일) 단위 집계 — 기존 집계 함수 재사용
    tmp = df.copy()
    tmp["date"] = tmp["session_date"]
    daily = bm.aggregate_daily(tmp, probs)                   # 15열, 거래일 시계열 위에서 롤링
    demb, ddates = bm.aggregate_embeddings(tmp, embs)        # 세션 평균 [CLS]
    assert list(daily["date"]) == list(ddates)

    # ---- 6) 달력일 프레임에 정렬 (비거래일/무뉴스 세션은 직전 값 유지 — 어차피 거래일 필터로 제거)
    agg = daily.set_index(pd.to_datetime(daily["date"]))[AGG_COLS]
    aligned = agg.reindex(frame["Date"])
    has_news = aligned["news_volume"].notna().values & is_tr
    aligned = aligned.ffill().fillna(0.0)
    aligned["news_available"] = has_news.astype(float)

    new_frame = frame.copy()
    for c in NEWS_COLS:
        new_frame[c] = aligned[c].values
    non_news = [c for c in frame.columns if c not in NEWS_COLS]
    assert new_frame[non_news].equals(frame[non_news]), "뉴스 외 열이 변경됨"
    assert list(new_frame.columns) == list(frame.columns), "열 순서가 바뀜"

    emb_map = {pd.Timestamp(dt): demb[i] for i, dt in enumerate(ddates)}
    E = np.zeros((len(frame), demb.shape[1]), dtype=np.float32)
    last = np.zeros(demb.shape[1], dtype=np.float32)
    for i, dt in enumerate(frame["Date"]):
        if bool(is_tr[i]) and dt in emb_map:
            last = emb_map[dt]
        E[i] = last

    # ---- 7) 저장
    out = BASE / sub
    # 시장 열은 원본 텍스트를 그대로 보존한다(float 을 다시 쓰면 CSV 왕복에서 1e-14 수준 반올림 차이가 생긴다).
    txt = pd.read_csv(BASE / sub / f"{mc}_research_enhanced.csv", dtype=str, keep_default_na=False)
    for c in NEWS_COLS:
        txt[c] = [repr(float(v)) for v in new_frame[c].values]
    txt.to_csv(out / f"{mc}_research_enhanced{SFX}.csv", index=False)
    np.save(out / f"news_embeddings_daily_aligned{SFX}.npy", E)
    pd.DataFrame({"Date": frame["Date"].dt.strftime("%Y-%m-%d")}).to_csv(
        out / f"news_embeddings_dates_aligned{SFX}.csv", index=False)
    daily.to_csv(out / f"news_sentiment_daily{SFX}.csv", index=False)
    per = df.drop(columns=[c for c in ("year", "month") if c in df.columns]).copy()
    per["sent_pos"], per["sent_neg"], per["sent_neu"] = probs[:, 0], probs[:, 1], probs[:, 2]
    per.to_csv(out / f"news_per_headline{SFX}.csv", index=False, encoding="utf-8-sig")

    n_tr = int(is_tr.sum())
    print(f"  뉴스가 있는 거래일 {int(has_news.sum())} / {n_tr} ({100 * has_news.sum() / n_tr:.1f}%)")
    vol = daily["news_volume"]
    print(f"  거래일당 헤드라인: 평균 {vol.mean():.2f}, 중앙값 {vol.median():.0f}, 최대 {vol.max():.0f}")
    print(f"  감성 점수 평균 {daily['sent_score_mean'].mean():+.4f}, 표준편차 {daily['sent_score_mean'].std():.4f}")
    print(f"  저장: {mc}_research_enhanced{SFX}.csv, news_embeddings_daily_aligned{SFX}.npy (shape {E.shape}), "
          f"news_sentiment_daily{SFX}.csv, news_per_headline{SFX}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="FinBERT 재현성 점검만")
    ap.add_argument("--reuse-cache", action="store_true", help="헤드라인별 FinBERT 캐시 재사용")
    ap.add_argument("--market", choices=["US", "UK"])
    a = ap.parse_args()
    print(f"device = {bm.DEVICE}  batch = {bm.BATCH_SIZE}  max_len = {bm.MAX_LEN}")
    if a.validate:
        sys.exit(0 if validate_against_old() else 1)
    import torch
    torch.manual_seed(bm.SEED)
    np.random.seed(bm.SEED)
    for mc in ([a.market] if a.market else ["US", "UK"]):
        build_market(mc, a.reuse_cache)
    print("\n[DONE]")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
