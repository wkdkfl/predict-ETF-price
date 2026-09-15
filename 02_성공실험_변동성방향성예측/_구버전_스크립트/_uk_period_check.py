# -*- coding: utf-8 -*-
"""영국이 특정 시험 구간에서만 실패하는가, 아니면 전 기간 실패하는가?

확장 윈도우 Walk-Forward 를 fold 별로 펼쳐서 두 시장을 비교한다.
타깃은 차분 형태(레짐 이동에 강건)를 사용하고, 두 시장에 동일하게 적용한다.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from _regime_fixes_v7 import prep, KW  # noqa: E402
import lightgbm as lgb  # noqa: E402

N_FOLDS = 8

for mc, lab in [("US", "미국"), ("UK", "영국")]:
    s, keep, ar_c = prep(mc)
    y = s["vol5"].values
    past = s["rv5_past"].values
    d = y - past
    n = len(s)
    min_tr = int(n * 0.35)
    fs = (n - min_tr) // N_FOLDS
    print("=" * 86)
    print("[%s]  n=%d,  fold 크기 %d" % (lab, n, fs))
    print("  %-26s %9s %9s %9s %9s" % ("시험 구간", "타깃평균", "AR_Only", "Full", "상관(Full)"))
    accs = {"AR_Only": [], "Full": []}
    for k in range(N_FOLDS):
        a = min_tr + k * fs
        b = min(a + fs, n)
        if b <= a:
            continue
        line = []
        for g, cols in [("AR_Only", ar_c), ("Full", keep)]:
            X = s[cols].values
            sc = StandardScaler().fit(X[:a])
            m = lgb.LGBMRegressor(**KW).fit(sc.transform(X[:a]), d[:a])
            p = m.predict(sc.transform(X[a:b])) + past[a:b]
            accs[g].append(r2_score(y[a:b], p))
            line.append((r2_score(y[a:b], p), np.corrcoef(y[a:b], p)[0, 1]))
        print("  %s ~ %s  %+9.2f %+9.3f %+9.3f %9.3f"
              % (s["Date"].iloc[a].date(), s["Date"].iloc[b - 1].date(),
                 y[a:b].mean(), line[0][0], line[1][0], line[1][1]))
    for g in accs:
        arr = np.array(accs[g])
        print("  -> %-8s 평균 %+.3f (표준편차 %.3f), 양수 fold %d/%d"
              % (g, arr.mean(), arr.std(ddof=1), int((arr > 0).sum()), len(arr)))
