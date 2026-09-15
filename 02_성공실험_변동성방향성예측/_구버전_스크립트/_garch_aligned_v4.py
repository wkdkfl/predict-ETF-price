# -*- coding: utf-8 -*-
"""전통 시계열 베이스라인 — ML 파이프라인과 '완전히 동일한' 표본/분할에서 평가.

기존 _garch_baseline.py 는 뉴스·임베딩 결측 제거를 하지 않아 ML 파이프라인보다
표본이 크고 시험 구간도 달랐다(예: 미국 GARCH 시험 453일 2024-03~2025-12 vs
ML 시험 270일 2023-12~2025-01). 따라서 "동일 분할에서 비교"라는 서술이 성립하지
않았다.

이 스크립트는 _run_enhanced_models_v4 의 전처리를 그대로 재현해 살아남은 행과
train/val/test 경계를 얻은 뒤, 그 위에서 HAR-RV·GARCH·GJR-GARCH 를 적합하여
ML 결과와 같은 척도·같은 날짜로 비교한다.

출력: {USD,UK}/results_garch_v4/garch_aligned_{US,UK}.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from _run_enhanced_models_v4 import (  # noqa: E402
    load_market, add_targets_and_ar, build_features, time_split,
    TRAIN_FRAC, VAL_FRAC, EMB_PCA_DIM, LOG_VOL_EPS, SEED,
)

try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False


def aligned_frame(market_code: str) -> pd.DataFrame:
    """ML 파이프라인과 동일한 행 집합/순서를 재현."""
    df, emb = load_market(market_code)
    df = add_targets_and_ar(df)
    i_tr = int(len(df) * TRAIN_FRAC)
    pca = PCA(n_components=EMB_PCA_DIM, random_state=SEED).fit(emb[:i_tr])
    feat, *_ = build_features(df, pca.transform(emb).astype(np.float32))
    feat = feat.dropna(subset=["target_vol", "target_dir"]).dropna().reset_index(drop=True)
    keep = set(feat["Date"])
    out = df[df["Date"].isin(keep)].reset_index(drop=True)
    assert len(out) == len(feat), (len(out), len(feat))
    return out


def har_forecast(d: pd.DataFrame, i_split: int, log_scale: bool):
    a = d["ETF"].astype(float)
    ret = np.log(a).diff()
    absret = ret.abs()
    X = pd.DataFrame({
        "har_d": absret.shift(1),
        "har_w": absret.shift(1).rolling(5, min_periods=1).mean(),
        "har_m": absret.shift(1).rolling(22, min_periods=5).mean(),
    })
    y = np.log(d["target_vol"] + LOG_VOL_EPS) if log_scale else d["target_vol"]
    if log_scale:
        X = np.log(X + LOG_VOL_EPS)
    ok = X.notna().all(axis=1) & y.notna() & np.isfinite(y)
    Xv, yv = X[ok].values, y[ok].values
    idx = np.where(ok.values)[0]
    tr = idx < i_split
    m = LinearRegression().fit(Xv[tr], yv[tr])
    p = m.predict(Xv[~tr])
    return yv[~tr], p


def garch_forecast(d: pd.DataFrame, i_split: int, gjr: bool):
    """확장 윈도우로 재적합하지 않고 train 구간 1회 적합 후 1일 앞 예측을 순차 산출."""
    ret = np.log(d["ETF"].astype(float)).diff().dropna()
    r = ret.values * 100.0
    n_tr = i_split - (len(d) - len(ret))
    o = 1 if gjr else 0
    preds = []
    am = arch_model(r[:n_tr], vol="GARCH", p=1, o=o, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)
    params = res.params
    omega = params["omega"]; alpha = params["alpha[1]"]; beta = params["beta[1]"]
    gamma = params.get("gamma[1]", 0.0)
    # 조건부 분산 순차 갱신 (계수 고정, 관측치는 실현값 사용)
    sigma2 = res.conditional_volatility[-1] ** 2
    eps = r[n_tr - 1]
    for t in range(n_tr, len(r)):
        sigma2 = omega + alpha * eps**2 + gamma * eps**2 * (eps < 0) + beta * sigma2
        preds.append(np.sqrt(sigma2) / 100.0)
        eps = r[t]
    y = np.abs(r[n_tr:]) / 100.0
    return y, np.array(preds)


def run(market_code: str) -> dict:
    print("=" * 62)
    print("  %s — 전통 모형 (ML 파이프라인과 동일 표본/분할)" % market_code)
    d = aligned_frame(market_code)
    n = len(d)
    i_split = int(n * (TRAIN_FRAC + VAL_FRAC))     # Train+Val | Test
    print("  전체 %d행, Train+Val %d / Test %d" % (n, i_split, n - i_split))
    print("  시험 구간 %s ~ %s"
          % (d["Date"].iloc[i_split].date(), d["Date"].iloc[-1].date()))

    out = {"n_total": n, "n_test": n - i_split,
           "test_start": str(d["Date"].iloc[i_split].date()),
           "test_end": str(d["Date"].iloc[-1].date())}

    for tag, log_scale in [("HAR-RV", False), ("HAR-RV_log", True)]:
        y, p = har_forecast(d, i_split, log_scale)
        out[tag] = {"r2": float(r2_score(y, p)),
                    "rmse": float(np.sqrt(mean_squared_error(y, p))),
                    "corr": float(np.corrcoef(y, p)[0, 1])}
        print("  [%-11s] R2=%+.4f  RMSE=%.6f  corr=%.3f"
              % (tag, out[tag]["r2"], out[tag]["rmse"], out[tag]["corr"]))

    if HAVE_ARCH:
        for tag, gjr in [("GARCH(1,1)", False), ("GJR-GARCH(1,1,1)", True)]:
            y, p = garch_forecast(d, i_split, gjr)
            m = min(len(y), len(p))
            y, p = y[:m], p[:m]
            r2_lvl = float(r2_score(y, p))
            yl = np.log(y + LOG_VOL_EPS); pl = np.log(p + LOG_VOL_EPS)
            out[tag] = {"r2": r2_lvl, "r2_log": float(r2_score(yl, pl)),
                        "rmse": float(np.sqrt(mean_squared_error(y, p))),
                        "corr": float(np.corrcoef(y, p)[0, 1])}
            print("  [%-11s] R2=%+.4f  logR2=%+.4f  corr=%.3f"
                  % (tag, r2_lvl, out[tag]["r2_log"], out[tag]["corr"]))

    sub = "USD" if market_code == "US" else "UK"
    od = BASE / sub / "results_garch_v4"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"garch_aligned_{market_code}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("  저장 -> %s" % (od / f"garch_aligned_{market_code}.json"))
    return out


if __name__ == "__main__":
    for m in ["US", "UK"]:
        run(m)
