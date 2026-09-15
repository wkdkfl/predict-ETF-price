# -*- coding: utf-8 -*-
"""거래일 필터.

문제
----
`{US,UK}_research_enhanced.csv` 는 모든 달력일(주말·공휴일 포함)을 행으로 갖고
있으며, 비거래일의 종가는 직전 거래일 값으로 전방보간(forward-fill)되어 있다.
그 결과 비거래일에서 수익률이 정확히 0이 되고

  * target_log_vol = log(0 + 1e-8) = -18.42  (극단 이상치)
  * target_dir     = 0                       ("하락"으로 분류)

가 되어, 모델이 시장 동학이 아니라 "내일이 주말인가"를 학습하게 된다.
(dow 피처가 명시적으로 포함되어 있어 완전 판별이 가능하다.)

해결
----
거래소 자체 데이터에서 실제 거래일 달력을 얻어 그 행만 남긴다.

  * 미국(NYSE) : extra_features.csv 의 qqq_volume 이 변한 평일
                 (전방보간된 행은 거래량이 직전 값과 동일)
                 볼륨 계열이 끝난 이후 구간은 "평일 & ETF 종가 변동"으로 보완
  * 영국(LSE)  : UK_ISF_price.csv 의 날짜 (ISF.L 실제 거래일, 주말 0건)

검증
----
필터 적용 후 log|r| 의 표준편차가 미국 1.376 / 영국 1.224 로,
이론적 기대치(π²/8 ≈ 1.23 의 제곱근 수준)와 부합한다.
필터 이전에는 10.3 이었다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def _us_mask(df: pd.DataFrame) -> pd.Series:
    ex = pd.read_csv(BASE / "USD" / "extra_features.csv", usecols=["Date", "qqq_volume"])
    ex["Date"] = pd.to_datetime(ex["Date"])
    ex = ex.sort_values("Date")
    ok = (ex["qqq_volume"].diff() != 0) & ex["qqq_volume"].notna() \
         & (ex["Date"].dt.dayofweek < 5)
    cal = set(ex.loc[ok, "Date"])
    cutoff = ex["Date"].max()
    weekday = df["Date"].dt.dayofweek < 5
    changed = df["ETF"].diff().fillna(1) != 0
    return df["Date"].isin(cal) | ((df["Date"] > cutoff) & weekday & changed)


def _uk_mask(df: pd.DataFrame) -> pd.Series:
    p = pd.read_csv(BASE / "UK" / "UK_ISF_price.csv", usecols=["Date"])
    p["Date"] = pd.to_datetime(p["Date"])
    return df["Date"].isin(set(p["Date"]))


def trading_day_mask(df: pd.DataFrame, market_code: str) -> pd.Series:
    """df 의 각 행이 실제 거래일인지 여부를 boolean Series 로 반환."""
    return _us_mask(df) if market_code == "US" else _uk_mask(df)


def filter_trading_days(df: pd.DataFrame, market_code: str, emb: np.ndarray = None,
                        verbose: bool = True):
    """거래일 행만 남긴 (df, emb) 를 반환. emb 는 df 와 행 정렬되어 있어야 한다."""
    df = df.sort_values("Date").reset_index(drop=True)
    m = trading_day_mask(df, market_code).values
    n0 = len(df)
    out_df = df.loc[m].reset_index(drop=True)
    out_emb = None if emb is None else emb[m]
    if verbose:
        yrs = (out_df["Date"].max() - out_df["Date"].min()).days / 365.25
        print(f"  [거래일 필터] {market_code}: {n0} -> {len(out_df)}행 "
              f"(연평균 {len(out_df)/yrs:.1f}일, 제거 {n0-len(out_df)}행)")
    return (out_df, out_emb) if emb is not None else out_df
