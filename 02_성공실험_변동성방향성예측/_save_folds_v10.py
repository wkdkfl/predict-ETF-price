# -*- coding: utf-8 -*-
"""fold별 Walk-Forward 결과 저장 (국면 의존성 표/그림용)"""
from __future__ import annotations
import io, contextlib, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402

SEED = 42
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
N_FOLDS = 8


def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


rows = []
for mc in ["US", "UK"]:
    s, keep, ar_c = build(mc)
    y, past = s["vol5"].values, s["rv5_past"].values
    d = y - past
    n = len(s); mt = int(n * 0.35); fs = (n - mt) // N_FOLDS
    for k in range(N_FOLDS):
        a = mt + k * fs; b = min(a + fs, n)
        if b <= a:
            continue
        rec = {"market": mc, "fold": k + 1,
               "test_start": str(s["Date"].iloc[a].date()),
               "test_end": str(s["Date"].iloc[b - 1].date()),
               "n_train": a, "n_test": b - a,
               "target_mean": float(y[a:b].mean())}
        for g, cols in [("AR_Only", ar_c), ("Full", keep)]:
            X = s[cols].values
            sc = StandardScaler().fit(X[:a])
            p = lgb.LGBMRegressor(**KW).fit(sc.transform(X[:a]), d[:a]) \
                   .predict(sc.transform(X[a:b])) + past[a:b]
            rec[g + "_r2"] = float(r2_score(y[a:b], p))
            rec[g + "_corr"] = float(np.corrcoef(y[a:b], p)[0, 1])
        rows.append(rec)
        print("  %s fold%d %s~%s  평균 %.2f  AR %+.3f  Full %+.3f"
              % (mc, k + 1, rec["test_start"], rec["test_end"], rec["target_mean"],
                 rec["AR_Only_r2"], rec["Full_r2"]))

df = pd.DataFrame(rows)
df.to_csv(BASE / "_final_v10_folds.csv", index=False, encoding="utf-8-sig")
print("\n저장: _final_v10_folds.csv")
for mc in ["US", "UK"]:
    t = df[df.market == mc]
    print("  %s: AR_Only 평균 %+.3f (양수 %d/%d) | Full 평균 %+.3f (양수 %d/%d)"
          % (mc, t.AR_Only_r2.mean(), (t.AR_Only_r2 > 0).sum(), len(t),
             t.Full_r2.mean(), (t.Full_r2 > 0).sum(), len(t)))
