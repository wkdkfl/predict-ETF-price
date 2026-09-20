# -*- coding: utf-8 -*-
"""뉴스 강도·주제 피처(INT) 생성 — plan/20260920_improvement_plan.md 의 사전 고정 정의.

입력 : 공통데이터/{USD,UK}/{US,UK}_economic_news_candidates.csv  (일별 선별 전 후보 기사 전체)
출력 : 개선실험/data/{US,UK}_news_intensity.csv  (거래일 행)

정의 (실험 결과를 보기 전에 고정)
--------------------------------
  * 장 마감 요약 패턴 기사는 제외(기존 파이프라인과 동일), 같은 URL 은 가장 이른 날짜에만 남긴다.
  * 비거래일(주말·공휴일) 기사는 다음 거래일에 합산 — 기존 newsv2 규칙과 동일. 따라서 거래일 t 의 값은 t 일까지의 뉴스만 반영한다.
  * 일별 원값: 후보 수 n_cand, 관련도 점수 ≥3 인 수 n_hi, 평균 점수 score_mean, 주제 6종 기사 수.
  * 수준 드리프트(연도별 수집량 차이) 제거: 각 값을 '과거 60거래일 평균(당일 제외)' 대비 비율로 바꾼다.
        rel = (x + 1) / (mean_{t-60..t-1}(x) + 1),  초기 20거래일 미만은 1.0
  * 주제 분류는 collect_economic_news.py 의 HIGH_KW 정규식을 그대로 재사용한다(헤드라인 + 요약문).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent / "공통데이터"))

import collect_economic_news as cen  # noqa: E402
from _trading_days import trading_day_mask  # noqa: E402

OUT = BASE / "개선실험" / "data"
MARKETS = {"US": ("USD", "US"), "UK": ("UK", "UK")}

# HIGH_KW 인덱스 -> 주제 (collect_economic_news.HIGH_KW 는 15개)
TOPICS = {
    "policy": [0, 1, 2],          # Fed / BoE·ECB·중앙은행 / 금리·통화정책
    "macro": [3, 4, 5, 12],       # 인플레이션 / GDP·침체 / 고용 / 재정
    "trade_geo": [6, 13],         # 관세·무역·제재 / 브렉시트·위기·부도
    "market": [7, 11],            # 증시 / 채권·금리수익률
    "corp": [8, 14],              # 실적 / 인수합병
    "commod_fx": [9, 10],         # 유가·에너지 / 환율
}
WINDOW, MIN_PERIODS = 60, 20


def session_dates(cal: pd.Series, is_trading: np.ndarray, dates: pd.Series) -> pd.Series:
    """각 날짜를 (그 날 이후 첫 거래일)로 매핑."""
    trading = np.sort(cal[is_trading].values)
    idx = np.searchsorted(trading, dates.values, side="left")
    out = np.full(len(dates), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = idx < len(trading)
    out[ok] = trading[idx[ok]]
    return pd.Series(out, index=dates.index)


def rel(x: pd.Series) -> pd.Series:
    past = x.shift(1).rolling(WINDOW, min_periods=MIN_PERIODS).mean()
    r = (x + 1.0) / (past + 1.0)
    return r.fillna(1.0)


def build(mc: str):
    sub, pref = MARKETS[mc]
    assert len(cen.HIGH_KW) == 15
    cand = pd.read_csv(BASE.parent / "공통데이터" / sub / f"{pref}_economic_news_candidates.csv")
    n0 = len(cand)
    cand = cand[cand["is_close_summary"] == 0]
    cand = cand.sort_values("date").drop_duplicates("url", keep="first").reset_index(drop=True)
    print(f"[{mc}] 후보 {n0:,}건 -> 장 마감 요약 제외·URL 중복 제거 후 {len(cand):,}건")

    text = (cand["headline"].fillna("") + " " + cand["summary"].fillna("")).tolist()
    flags = np.zeros((len(text), 15), dtype=bool)
    for j, p in enumerate(cen.HIGH_KW):
        flags[:, j] = [bool(p.search(t)) for t in text]
    for name, idxs in TOPICS.items():
        cand[f"topic_{name}"] = flags[:, idxs].any(axis=1).astype(int)
    cand["hi"] = (cand["score"] >= cen.THRESH).astype(int)

    frame = pd.read_csv(BASE / sub / f"{mc}_research_enhanced.csv", usecols=["Date", "ETF"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    is_tr = trading_day_mask(frame, mc).values
    sess = session_dates(frame["Date"], is_tr, pd.to_datetime(cand["date"]))
    cand["session"] = sess
    cand = cand[cand["session"].notna()]

    g = cand.groupby("session")
    daily = pd.DataFrame({
        "n_cand": g.size(),
        "n_hi": g["hi"].sum(),
        "score_mean": g["score"].mean(),
        **{f"topic_{k}": g[f"topic_{k}"].sum() for k in TOPICS},
    })
    days = pd.DatetimeIndex(frame.loc[is_tr, "Date"])
    daily = daily.reindex(days).fillna(0.0)
    out = pd.DataFrame({"Date": days})
    out["n_cand"], out["n_hi"] = daily["n_cand"].values, daily["n_hi"].values
    out["n_cand_rel"] = rel(daily["n_cand"]).values
    out["n_hi_rel"] = rel(daily["n_hi"]).values
    out["score_mean_rel"] = rel(daily["score_mean"]).values
    for k in TOPICS:
        out[f"topic_{k}_rel"] = rel(daily[f"topic_{k}"]).values
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / f"{mc}_news_intensity.csv", index=False)
    print(f"  거래일 {len(out)}일, 거래일당 후보 평균 {out['n_cand'].mean():.1f}건 (최소 {int(out['n_cand'].min())}), "
          f"n_hi 평균 {out['n_hi'].mean():.1f}, 주제별 비율 피처 평균 "
          + ", ".join(f"{k} {out[f'topic_{k}_rel'].mean():.2f}" for k in TOPICS))
    print(f"  저장: {OUT / (mc + '_news_intensity.csv')}")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    for mc in ("US", "UK"):
        build(mc)
