# -*- coding: utf-8 -*-
"""탐색에서 나온 유망 지점(다일 타깃)의 강건성 검증.

탐색은 (h) x (PCA 차원) x (피처군) 을 훑었으므로, 단일 시험 분할에서의 우위는
선택 편의(selection bias)일 수 있다. 여기서는 다음 세 가지로 재검증한다.

1) Walk-Forward (확장 윈도우 5-fold) — 여러 시험 구간에서 재현되는가
2) 비중첩 평가 — h일 중첩 윈도우 때문에 시험 관측치가 자기상관되므로,
   h 간격으로 솎아낸 부분표본에서도 유지되는가
3) 블록 부트스트랩 CI (block=h) — 중첩을 감안한 신뢰구간
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
H_LIST = [5, 20]
N_BOOT = 2000


def prep(market_code: str, pdim: int = PCA_DIM):
    df0, emb = load_market(market_code)
    df = add_targets_and_ar(df0)
    tg = make_targets(df0)
    i_tr0 = int(len(df) * TRAIN_FRAC)
    pca = PCA(n_components=pdim, random_state=SEED).fit(emb[:i_tr0])
    feat, fin_cols, news_cols, ar_cols = build_features(df, pca.transform(emb).astype(np.float32))
    for c in tg.columns:
        feat[c] = tg[c].values
    base_cols = [c for c in feat.columns
                 if c not in {"target_vol", "target_log_vol", "target_dir", "next_ret", "Date"}
                 and not c.startswith(("vol", "dir"))]
    emb_cols = [c for c in base_cols if c.startswith("emb_pc")]
    sent_cols = [c for c in base_cols if c in NEWS_SENT_COLS]
    ar_all = [c for c in base_cols if c in ar_cols]
    fin_all = [c for c in base_cols if c not in ar_all + sent_cols + emb_cols]
    groups = {
        "AR_Only": ar_all,
        "Financial_Only": fin_all,
        "News_Pure": sent_cols + emb_cols,
        "AR+News": ar_all + sent_cols + emb_cols,
        "Full": base_cols,
    }
    return feat, groups


def clf():
    return lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                              random_state=SEED, n_jobs=-1, verbose=-1)


def block_boot_auc(y, p, block, n_boot=N_BOOT):
    rng = np.random.RandomState(SEED)
    n = len(y); nb = max(1, n // block); out = []
    for _ in range(n_boot):
        st = rng.randint(0, max(1, n - block + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        out.append(roc_auc_score(yy, pp))
    out = np.array(out)
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def walk_forward(sub, cols, tname, n_folds=5):
    n = len(sub); min_tr = int(n * 0.40); fs = (n - min_tr) // n_folds
    aucs = []
    for k in range(n_folds):
        a = min_tr + k * fs; b = min(a + fs, n)
        if b <= a:
            continue
        tr, te = sub.iloc[:a], sub.iloc[a:b]
        if te[tname].nunique() < 2:
            continue
        sc = StandardScaler().fit(tr[cols].values)
        m = clf().fit(sc.transform(tr[cols].values), tr[tname].values.astype(int))
        p = m.predict_proba(sc.transform(te[cols].values))[:, 1]
        aucs.append(roc_auc_score(te[tname].values.astype(int), p))
    return np.array(aucs)


for mc in ["US", "UK"]:
    feat, groups = prep(mc)
    print("=" * 78)
    print("  %s  (PCA=%d)" % (mc, PCA_DIM))
    for h in H_LIST:
        tname = "dir%d" % h
        base_cols = groups["Full"]
        sub = feat.dropna(subset=[tname]).dropna(subset=base_cols).reset_index(drop=True)
        n = len(sub); i_te = int(n * (TRAIN_FRAC + VAL_FRAC))
        yte = sub[tname].values[i_te:].astype(int)
        blm = max(yte.mean(), 1 - yte.mean())
        print("\n  [%s]  시험 n=%d  다수클래스=%.3f  (중첩 윈도우 h=%d -> 유효 관측 ~%d)"
              % (tname, len(yte), blm, h, len(yte) // h))
        print("    %-15s %-10s %-20s %-22s %s"
              % ("피처군", "단일분할", "블록부트스트랩 95%CI", "Walk-Forward(평균±sd)", "비중첩"))
        for gname, cols in groups.items():
            if not cols:
                continue
            sc = StandardScaler().fit(sub[cols].values[:i_te])
            m = clf().fit(sc.transform(sub[cols].values[:i_te]),
                          sub[tname].values[:i_te].astype(int))
            p = m.predict_proba(sc.transform(sub[cols].values[i_te:]))[:, 1]
            auc = roc_auc_score(yte, p)
            lo, hi = block_boot_auc(yte, p, block=h)
            # 비중첩 부분표본
            ynn, pnn = yte[::h], p[::h]
            auc_nn = roc_auc_score(ynn, pnn) if len(np.unique(ynn)) > 1 else np.nan
            wf = walk_forward(sub, cols, tname)
            print("    %-15s %.3f      [%.3f, %.3f]        %.3f ± %.3f          %.3f (n=%d)"
                  % (gname, auc, lo, hi, wf.mean() if len(wf) else np.nan,
                     wf.std(ddof=1) if len(wf) > 1 else np.nan, auc_nn, len(ynn)))
