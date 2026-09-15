# -*- coding: utf-8 -*-
"""v6 — 표본 손실 원인 제거 후 재탐색.

발견
----
`eth`(이더리움) 열은 2017-11-10 부터만 존재한다. dropna() 가 행 단위로 동작하므로
이 한 열 때문에 2014-01 ~ 2017-11 구간 전체(미국 967행, 영국 925행)가 삭제되어
학습 표본이 40% 가까이 줄어 있었다. 결측이 과도한 열을 먼저 제거하면
표본이 미국 1,797 -> 2,764, 영국 1,726 -> 2,651 로 회복된다.

여기서는 그 상태에서 (타깃 시계) x (피처군) 을 다시 평가한다.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from _run_enhanced_models_v4 import (  # noqa: E402
    load_market, add_targets_and_ar, build_features,
    TRAIN_FRAC, VAL_FRAC, SEED, NEWS_SENT_COLS,
)
from _explore_targets_v5 import make_targets  # noqa: E402

import lightgbm as lgb  # noqa: E402

PCA_DIM = 15
MAX_MISSING_FRAC = 0.05   # 결측률이 이보다 큰 열은 제거 (행을 살리기 위해)


def prep(market_code: str, pdim=PCA_DIM):
    df0, emb = load_market(market_code)
    df = add_targets_and_ar(df0)
    tg = make_targets(df0)
    i0 = int(len(df) * TRAIN_FRAC)
    pca = PCA(n_components=pdim, random_state=SEED).fit(emb[:i0])
    feat, fin_cols, news_cols, ar_cols = build_features(df, pca.transform(emb).astype(np.float32))
    for c in tg.columns:
        feat[c] = tg[c].values

    meta = {"target_vol", "target_log_vol", "target_dir", "next_ret", "Date"}
    tgt_cols = [c for c in feat.columns if c.startswith(("vol", "dir"))]
    cand = [c for c in feat.columns if c not in meta and c not in tgt_cols]

    f2 = feat.dropna(subset=["target_vol"])
    miss = f2[cand].isna().mean()
    dropped = sorted(miss[miss > MAX_MISSING_FRAC].index)
    keep = [c for c in cand if c not in dropped]
    print("  결측률 >%.0f%% 로 제외한 열 %d개: %s"
          % (100 * MAX_MISSING_FRAC, len(dropped), ", ".join(dropped[:8]) + ("..." if len(dropped) > 8 else "")))

    emb_c = [c for c in keep if c.startswith("emb_pc")]
    sent_c = [c for c in keep if c in NEWS_SENT_COLS]
    ar_c = [c for c in keep if c in ar_cols]
    fin_c = [c for c in keep if c not in emb_c + sent_c + ar_c]
    groups = {"AR_Only": ar_c, "Financial_Only": fin_c, "News_Pure": sent_c + emb_c,
              "AR+News": ar_c + sent_c + emb_c, "Full": keep}
    return feat, groups, keep


def walk_forward(sub, cols, tname, is_dir, n_folds=5):
    n = len(sub); mt = int(n * 0.40); fs = (n - mt) // n_folds
    out = []
    for k in range(n_folds):
        a = mt + k * fs; b = min(a + fs, n)
        if b <= a:
            continue
        tr, te = sub.iloc[:a], sub.iloc[a:b]
        sc = StandardScaler().fit(tr[cols].values)
        Xtr, Xte = sc.transform(tr[cols].values), sc.transform(te[cols].values)
        if is_dir:
            if te[tname].nunique() < 2:
                continue
            m = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                   random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(Xtr, tr[tname].values.astype(int))
            out.append(roc_auc_score(te[tname].values.astype(int),
                                     m.predict_proba(Xte)[:, 1]))
        else:
            m = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                  random_state=SEED, n_jobs=-1, verbose=-1)
            m.fit(Xtr, tr[tname].values)
            out.append(r2_score(te[tname].values, m.predict(Xte)))
    return np.array(out)


for mc in ["US", "UK"]:
    print("=" * 84)
    print("  %s" % mc)
    feat, groups, keep = prep(mc)
    for tname in ["vol1", "vol5", "vol10", "vol20", "dir1", "dir5", "dir20"]:
        is_dir = tname.startswith("dir")
        sub = feat.dropna(subset=[tname]).dropna(subset=keep).reset_index(drop=True)
        sub = sub[np.isfinite(sub[tname])].reset_index(drop=True)
        n = len(sub); i_te = int(n * (TRAIN_FRAC + VAL_FRAC))
        yte = sub[tname].values[i_te:]
        extra = ""
        if is_dir:
            extra = "  다수클래스=%.3f" % max(yte.mean(), 1 - yte.mean())
        print("\n  [%s] 전체 %d행 (학습+검증 %d / 시험 %d)%s"
              % (tname, n, i_te, len(yte), extra))
        print("    %-16s %-12s %s" % ("피처군", "단일분할", "Walk-Forward 평균±sd"))
        for g, cols in groups.items():
            if not cols:
                continue
            sc = StandardScaler().fit(sub[cols].values[:i_te])
            Xtr, Xte = sc.transform(sub[cols].values[:i_te]), sc.transform(sub[cols].values[i_te:])
            if is_dir:
                if len(np.unique(yte)) < 2:
                    continue
                m = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                       random_state=SEED, n_jobs=-1, verbose=-1)
                m.fit(Xtr, sub[tname].values[:i_te].astype(int))
                s = roc_auc_score(yte.astype(int), m.predict_proba(Xte)[:, 1])
            else:
                m = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                      random_state=SEED, n_jobs=-1, verbose=-1)
                m.fit(Xtr, sub[tname].values[:i_te])
                s = r2_score(yte, m.predict(Xte))
            wf = walk_forward(sub, cols, tname, is_dir)
            print("    %-16s %+.3f       %+.3f ± %.3f"
                  % (g, s, wf.mean() if len(wf) else np.nan,
                     wf.std(ddof=1) if len(wf) > 1 else np.nan))
