# -*- coding: utf-8 -*-
"""뉴스에 신호가 있는가? — 이월(forward-fill)되지 않은 '실제 뉴스가 있는 날'만으로 검정.

배경
----
뉴스 고유 날짜는 미국 221일 / 영국 713일뿐이며(2017~2025), 나머지 날짜의 뉴스
피처는 직전 가용일 값이 이월된 것이다. 따라서 전체 표본에서 "뉴스 기여 없음"이
나오는 것은 뉴스의 정보력에 대한 검정이 아니라 데이터 부재에 대한 검정일 수 있다.

이 스크립트는
 (1) 학습/시험 구간별 실제 뉴스 커버리지를 정량화하고
 (2) 뉴스가 실재하는 날만 남긴 부분표본에서 AR_Only vs AR+News 를 비교한다.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

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
MAX_MISSING = 0.05


def real_news_dates(mk_sub: str) -> set:
    p = BASE / mk_sub / "news_per_headline.csv"
    h = pd.read_csv(p, usecols=["date"])
    return set(pd.to_datetime(h["date"], errors="coerce").dropna().dt.normalize())


def prep(mc: str):
    sub = "USD" if mc == "US" else "UK"
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
    emb_c = [c for c in keep if c.startswith("emb_pc")]
    sent_c = [c for c in keep if c in NEWS_SENT_COLS]
    ar_c = [c for c in keep if c in ar_cols]
    feat["has_news"] = feat["Date"].dt.normalize().isin(real_news_dates(sub))
    return feat, keep, ar_c, sent_c + emb_c


def cv_score(sub, cols, tname, is_dir, n_splits=5):
    """TimeSeriesSplit 교차검증 평균 (부분표본이 작아 단일 분할 대신 사용)."""
    X = sub[cols].values
    y = sub[tname].values
    out = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(X):
        if is_dir and len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        kw = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
                  n_jobs=-1, verbose=-1)
        if is_dir:
            m = lgb.LGBMClassifier(**kw).fit(sc.transform(X[tr]), y[tr].astype(int))
            out.append(roc_auc_score(y[te].astype(int), m.predict_proba(sc.transform(X[te]))[:, 1]))
        else:
            m = lgb.LGBMRegressor(**kw).fit(sc.transform(X[tr]), y[tr])
            out.append(r2_score(y[te], m.predict(sc.transform(X[te]))))
    return np.array(out)


for mc in ["US", "UK"]:
    feat, keep, ar_c, news_c = prep(mc)
    print("=" * 78)
    print("  %s" % mc)
    # --- 커버리지 ---
    base = feat.dropna(subset=["target_vol"]).dropna(subset=keep).reset_index(drop=True)
    n = len(base); i_te = int(n * (TRAIN_FRAC + VAL_FRAC))
    tr_cov = base["has_news"][:i_te].mean(); te_cov = base["has_news"][i_te:].mean()
    print("  실제 뉴스가 있는 날의 비율:  학습+검증 %.1f%%  /  시험 %.1f%%"
          % (100 * tr_cov, 100 * te_cov))
    print("  (나머지 날은 직전 가용일의 뉴스 벡터가 이월됨)")

    for tname, is_dir in [("vol5", False), ("dir5", True), ("dir1", True)]:
        s_all = feat.dropna(subset=[tname]).dropna(subset=keep).reset_index(drop=True)
        s_all = s_all[np.isfinite(s_all[tname])].reset_index(drop=True)
        s_news = s_all[s_all["has_news"]].reset_index(drop=True)
        print("\n  [%s]  전체 %d행  /  실제 뉴스일만 %d행" % (tname, len(s_all), len(s_news)))
        for label, ss in [("전체 표본", s_all), ("뉴스 실재일만", s_news)]:
            if len(ss) < 200:
                print("    %-14s 표본 부족(%d행) — 생략" % (label, len(ss)))
                continue
            a = cv_score(ss, ar_c, tname, is_dir)
            b = cv_score(ss, ar_c + news_c, tname, is_dir)
            metric = "AUC" if is_dir else "R2"
            print("    %-14s AR_Only %s=%+.3f ± %.3f   AR+News %s=%+.3f ± %.3f   증분 %+.3f"
                  % (label, metric, a.mean(), a.std(ddof=1),
                     metric, b.mean(), b.std(ddof=1), b.mean() - a.mean()))
