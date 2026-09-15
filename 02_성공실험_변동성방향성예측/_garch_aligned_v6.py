# -*- coding: utf-8 -*-
"""전통 시계열 베이스라인 — v6 파이프라인과 동일한 표본/분할/타깃에서 평가.

v6 의 주 타깃은 5일 실현변동성 log RV_5 이므로, HAR 계열도 같은 타깃으로 적합한다.
GARCH 계열은 조건부 표준편차의 h일 예측을 집계해 같은 척도로 변환한다.

출력: {USD,UK}/results_v6/traditional_baselines.csv
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from _run_enhanced_models_v4 import (  # noqa: E402
    load_market, add_targets_and_ar, build_features,
    TRAIN_FRAC, VAL_FRAC, SEED, NEWS_SENT_COLS,
)
from _explore_targets_v5 import make_targets  # noqa: E402

PCA_DIM = 15
MAX_MISSING = 0.05
EPS = 1e-8
HORIZONS = [1, 5, 10]

try:
    from arch import arch_model
    HAVE_ARCH = True
except Exception:
    HAVE_ARCH = False


def aligned(mc: str):
    """v6 과 동일한 행 집합을 재현하고 원계열 수익률을 함께 반환."""
    df0, emb = load_market(mc)
    df = add_targets_and_ar(df0)
    tg = make_targets(df0)
    i0 = int(len(df) * TRAIN_FRAC)
    pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(emb[:i0])
    feat, fin_cols, news_cols, ar_cols = build_features(df, pca.transform(emb).astype(np.float32))
    for c in tg.columns:
        feat[c] = tg[c].values
    meta = {"target_vol", "target_log_vol", "target_dir", "next_ret", "Date"}
    tc = [c for c in feat.columns if c.startswith(("vol", "dir"))]
    cand = [c for c in feat.columns if c not in meta and c not in tc]
    f2 = feat.dropna(subset=["target_vol"])
    miss = f2[cand].isna().mean()
    keep = [c for c in cand if miss[c] <= MAX_MISSING]
    df0 = df0.copy()
    df0["ret"] = np.log(df0["ETF"].astype(float)).diff()
    return feat, keep, df0[["Date", "ret"]]


rows = []
for mc in ["US", "UK"]:
    print("=" * 68)
    print("  %s" % mc)
    feat, keep, rets = aligned(mc)
    for h in HORIZONS:
        tname = "vol%d" % h
        s = feat.dropna(subset=[tname]).dropna(subset=keep).reset_index(drop=True)
        s = s[np.isfinite(s[tname])].reset_index(drop=True)
        s = s.merge(rets, on="Date", how="left")
        n = len(s); i_te = int(n * (TRAIN_FRAC + VAL_FRAC))
        y = s[tname].values
        a = s["ret"].abs()

        # ---- HAR: 일/주/월 성분(모두 t 시점까지 관측 가능)으로 log RV_h 예측 ----
        X = pd.DataFrame({
            "d": np.log(a.shift(1) + EPS),
            "w": np.log(a.shift(1).rolling(5, min_periods=1).mean() + EPS),
            "m": np.log(a.shift(1).rolling(22, min_periods=5).mean() + EPS),
        })
        ok = X.notna().all(axis=1).values & np.isfinite(y)
        idx = np.where(ok)[0]
        tr = idx[idx < i_te]; te = idx[idx >= i_te]
        m = LinearRegression().fit(X.values[tr], y[tr])
        r2_har = r2_score(y[te], m.predict(X.values[te]))
        rows.append(dict(market=mc, target=tname, model="HAR-RV", r2=r2_har, n_test=len(te)))
        print("    [%s] HAR-RV            log-R2 = %+.4f" % (tname, r2_har))

        # ---- GARCH / GJR: train 구간 1회 적합 후 조건부 분산 순차 갱신 ----
        if HAVE_ARCH:
            r = s["ret"].values * 100.0
            valid = ~np.isnan(r)
            for gjr, tag in [(False, "GARCH(1,1)"), (True, "GJR-GARCH(1,1,1)")]:
                try:
                    rr = np.where(valid, r, 0.0)
                    res = arch_model(rr[:i_te], vol="GARCH", p=1, o=1 if gjr else 0, q=1,
                                     dist="normal").fit(disp="off", show_warning=False)
                    pr = res.params
                    om, al, be = pr["omega"], pr["alpha[1]"], pr["beta[1]"]
                    ga = pr.get("gamma[1]", 0.0)
                    sig2 = float(res.conditional_volatility[-1] ** 2)
                    eps = rr[i_te - 1]
                    preds = []
                    for t in range(i_te, len(rr)):
                        sig2 = om + al * eps**2 + ga * eps**2 * (eps < 0) + be * sig2
                        # h일 평균 분산 예측 -> RV_h 근사 (계수 고정 하 무조건분산으로 수렴)
                        pers = al + be + ga / 2.0
                        uncond = om / max(1e-12, 1 - pers)
                        acc = 0.0
                        s2 = sig2
                        for _ in range(h):
                            acc += s2
                            s2 = uncond + pers * (s2 - uncond)
                        preds.append(np.sqrt(acc / h) / 100.0)
                        eps = rr[t]
                    p = np.log(np.array(preds) + EPS)
                    yy = y[i_te:]
                    mlen = min(len(p), len(yy))
                    good = np.isfinite(p[:mlen]) & np.isfinite(yy[:mlen])
                    r2g = r2_score(yy[:mlen][good], p[:mlen][good])
                    rows.append(dict(market=mc, target=tname, model=tag, r2=r2g,
                                     n_test=int(good.sum())))
                    print("    [%s] %-17s log-R2 = %+.4f" % (tname, tag, r2g))
                except Exception as e:
                    print("    [%s] %-17s 실패: %s" % (tname, tag, str(e)[:50]))

out = pd.DataFrame(rows)
for mc, sub in [("US", "USD"), ("UK", "UK")]:
    d = BASE / sub / "results_v6"
    d.mkdir(exist_ok=True)
    out[out.market == mc].to_csv(d / "traditional_baselines.csv", index=False)
print("\n저장 완료")
