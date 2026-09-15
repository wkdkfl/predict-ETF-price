# -*- coding: utf-8 -*-
"""v12 — 절약적 모형(parsimonious) 및 FTSE 250 강건성 검정.

동기
----
v11에서 피처를 추가하자 두 시장 모두 성능이 떨어졌다(미국 Full 0.346 -> 0.119).
영국은 모든 설정에서 AR_Only가 Full을 앞섰다. 표본이 2,000행 수준인 환경에서는
피처 축소가 옳은 방향임을 시사한다. HAR 문헌의 표준 모형이 예측변수 3개뿐이라는
점도 같은 방향이다.

검정 1. 피처 개수를 3 -> 25개로 늘려가며 두 시장의 성능 곡선을 그린다.
검정 2. 영국 결과가 ISF 고유의 것인지, FTSE 250(MIDD)에서도 재현되는지 확인한다.

두 검정 모두 결과와 무관하게 보고하며, 시장별로 다른 설정을 고르지 않는다.
"""
from __future__ import annotations

import contextlib, io, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402
from _regime_fixes_v7 import prep  # noqa: E402

SEED, EPS, N_BOOT, BLOCK = 42, 1e-8, 5000, 10
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)

# HAR 문헌의 표준 3성분에 대응하는 최소 피처 (관측 가능 순서대로)
CORE3 = ["absret_lag1", "rolling_absret_5", "rolling_absret_22"]
CORE6 = CORE3 + ["rolling_std_5", "rolling_std_20", "log_absret_lag1"]
CORE10 = CORE6 + ["absret_lag2", "absret_lag3", "rolling_absret_20", "absret_surprise"]


def boot_ci(y, p):
    rng = np.random.RandomState(SEED)
    n = len(y); nb = max(1, n // BLOCK); out = []
    for _ in range(N_BOOT):
        st = rng.randint(0, max(1, n - BLOCK + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]
        out.append(r2_score(y[idx], p[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def evaluate(frames, cols, label):
    CUT_TEST = min(frames[m][0]["Date"].iloc[int(len(frames[m][0]) * 0.85)] for m in frames)
    CUT_TRAIN = min(frames[m][0]["Date"].iloc[int(len(frames[m][0]) * 0.70)] for m in frames)
    Z, Y, PAST, D, MK = {}, {}, {}, {}, {}
    for mc in frames:
        s = frames[mc][0]
        tr = (s["Date"] < CUT_TRAIN).values
        sc = StandardScaler().fit(s[cols].values[tr])
        Z[mc] = sc.transform(s[cols].values)
        Y[mc] = s["vol5"].values
        PAST[mc] = s["rv5_past"].values
        D[mc] = Y[mc] - PAST[mc]
        MK[mc] = {"tr": tr,
                  "va": ((s["Date"] >= CUT_TRAIN) & (s["Date"] < CUT_TEST)).values,
                  "te": (s["Date"] >= CUT_TEST).values}
    res = {}
    for mc in frames:
        o = [k for k in frames if k != mc][0]
        tr, va, te = MK[mc]["tr"], MK[mc]["va"], MK[mc]["te"]
        Xp = np.vstack([Z[mc][tr], Z[o][MK[o]["tr"]]])
        yp = np.concatenate([D[mc][tr], D[o][MK[o]["tr"]]])
        m = lgb.LGBMRegressor(**KW).fit(Xp, yp)
        p_va = m.predict(Z[mc][va]) + PAST[mc][va]
        p_te = m.predict(Z[mc][te]) + PAST[mc][te]
        cal = LinearRegression().fit(p_va.reshape(-1, 1), Y[mc][va])
        p = cal.predict(p_te.reshape(-1, 1))
        y = Y[mc][te]
        lo, hi = boot_ci(y, p)
        res[mc] = (r2_score(y, p), lo, hi, np.corrcoef(y, p)[0, 1])
    return res


if __name__ == "__main__":
    frames = {}
    for mc in ["US", "UK"]:
        with contextlib.redirect_stdout(io.StringIO()):
            frames[mc] = prep(mc)
    ar_common = [c for c in frames["US"][2] if c in frames["UK"][2]]
    full_common = [c for c in frames["US"][1] if c in frames["UK"][1]]

    print("=" * 84)
    print("  검정 1. 피처 개수에 따른 성능 (pooled + 보정, 두 시장 동일)")
    print("  %-22s %-34s %s" % ("피처 세트", "미국 R² [95% CI]", "영국 R² [95% CI]"))
    sets = [("HAR 3성분", [c for c in CORE3 if c in ar_common]),
            ("핵심 6개", [c for c in CORE6 if c in ar_common]),
            ("핵심 10개", [c for c in CORE10 if c in ar_common]),
            ("AR_Only (25개)", ar_common),
            ("Full (71개)", full_common)]
    for name, cols in sets:
        if len(cols) < 2:
            print("  %-22s (피처 부족: %s)" % (name, cols)); continue
        r = evaluate(frames, cols, name)
        def fmt(t):
            return "%+.3f [%+.3f,%+.3f]%s" % (t[0], t[1], t[2], " *" if t[1] > 0 else "  ")
        print("  %-22s %-34s %s  (n=%d)" % (name, fmt(r["US"]), fmt(r["UK"]), len(cols)))
    print("  (* = 95% 신뢰구간이 0을 제외)")
