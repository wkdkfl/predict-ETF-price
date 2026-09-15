# -*- coding: utf-8 -*-
"""타깃 확장 탐색 — 1일 앞 |r| 대신 다일 실현변동성(RV)을 예측하면 어떻게 되는가?

배경
----
log|r_{t+1}| = log σ_{t+1} + log|z_{t+1}| 에서 예측 불가능한 항 log|z| 의 분산이
π²/8 ≈ 1.23 으로 신호를 압도한다. 따라서 1일 앞 |r| 의 R² 상한은 통상 0.1~0.3 이며,
실제로 HAR-RV 조차 미국 0.037 / 영국 0.024 에 그쳤다.

HAR 문헌(Corsi 2009 등)의 표준 타깃은 다일 실현변동성이다:
    RV_h(t) = sqrt( (1/h) * Σ_{i=1..h} r²_{t+i} )
h 가 커질수록 평균화로 노이즈가 줄어 예측 가능성이 살아난다.

이 스크립트는 Optuna 없이 고정 하이퍼파라미터로 빠르게 스윕하여
(타깃 h) × (임베딩 PCA 차원) × (피처군) 조합의 성능을 비교한다.
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

import lightgbm as lgb  # noqa: E402

HORIZONS_VOL = [1, 5, 10, 20]
HORIZONS_DIR = [1, 5, 20]
PCA_DIMS = [50, 15, 5]
EPS = 1e-8


def make_targets(df: pd.DataFrame) -> pd.DataFrame:
    """다일 실현변동성 및 누적수익률 방향 타깃 생성 (t 시점 정보로 t+1..t+h 예측)."""
    r = np.log(df["ETF"].astype(float)).diff()
    out = pd.DataFrame(index=df.index)
    r2 = r ** 2
    for h in HORIZONS_VOL:
        # t+1 .. t+h 의 제곱수익률 평균 -> 실현변동성
        fwd = r2.shift(-1).rolling(h, min_periods=h).mean().shift(-(h - 1))
        out[f"vol{h}"] = np.log(np.sqrt(fwd) + EPS)
    for h in HORIZONS_DIR:
        cum = r.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1))
        out[f"dir{h}"] = (cum > 0).astype(float).where(cum.notna())
    return out


def evaluate(market_code: str):
    df0, emb = load_market(market_code)
    df = add_targets_and_ar(df0)
    tg = make_targets(df0)

    rows = []
    for pdim in PCA_DIMS:
        i_tr0 = int(len(df) * TRAIN_FRAC)
        pca = PCA(n_components=pdim, random_state=SEED).fit(emb[:i_tr0])
        feat, fin_cols, news_cols, ar_cols = build_features(
            df, pca.transform(emb).astype(np.float32))
        for c in tg.columns:
            feat[c] = tg[c].values

        base_cols = [c for c in feat.columns
                     if c not in {"target_vol", "target_log_vol", "target_dir",
                                  "next_ret", "Date"} and not c.startswith(("vol", "dir"))]
        emb_cols = [c for c in base_cols if c.startswith("emb_pc")]
        news_all = [c for c in base_cols if c in NEWS_SENT_COLS] + emb_cols
        ar_all = [c for c in base_cols if c in ar_cols]
        groups = {"AR_Only": ar_all, "Full": base_cols}

        for tname in [f"vol{h}" for h in HORIZONS_VOL] + [f"dir{h}" for h in HORIZONS_DIR]:
            sub = feat.dropna(subset=[tname]).dropna(subset=base_cols).reset_index(drop=True)
            sub = sub[np.isfinite(sub[tname])]
            n = len(sub)
            i_te = int(n * (TRAIN_FRAC + VAL_FRAC))
            is_dir = tname.startswith("dir")
            h = int(tname[3:]) if is_dir else int(tname[3:])
            for gname, cols in groups.items():
                sc = StandardScaler().fit(sub[cols].values[:i_te])
                Xtr = sc.transform(sub[cols].values[:i_te])
                Xte = sc.transform(sub[cols].values[i_te:])
                ytr = sub[tname].values[:i_te]
                yte = sub[tname].values[i_te:]
                if is_dir:
                    if len(np.unique(yte)) < 2:
                        continue
                    m = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.6,
                                           reg_lambda=5.0, random_state=SEED,
                                           n_jobs=-1, verbose=-1)
                    m.fit(Xtr, ytr.astype(int))
                    p = m.predict_proba(Xte)[:, 1]
                    score = roc_auc_score(yte.astype(int), p)
                    base = max(yte.mean(), 1 - yte.mean())
                    acc = ((p > 0.5).astype(int) == yte.astype(int)).mean()
                    rows.append(dict(market=market_code, pca=pdim, target=tname, h=h,
                                     group=gname, metric="AUC", score=score,
                                     acc=acc, baseline=base, n_test=len(yte)))
                else:
                    m = lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.6,
                                          reg_lambda=5.0, random_state=SEED,
                                          n_jobs=-1, verbose=-1)
                    m.fit(Xtr, ytr)
                    p = m.predict(Xte)
                    rows.append(dict(market=market_code, pca=pdim, target=tname, h=h,
                                     group=gname, metric="R2", score=r2_score(yte, p),
                                     acc=np.nan, baseline=np.nan, n_test=len(yte)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allr = []
    for mc in ["US", "UK"]:
        print("=" * 74)
        print("  %s" % mc)
        r = evaluate(mc)
        allr.append(r)
        for metric in ["R2", "AUC"]:
            sub = r[r.metric == metric]
            if sub.empty:
                continue
            print("\n  [%s]" % ("변동성 log-R²" if metric == "R2" else "방향성 AUC"))
            pv = sub.pivot_table(index=["target", "group"], columns="pca", values="score")
            print(pv.round(3).to_string())
            if metric == "AUC":
                bl = sub.groupby("target")["baseline"].first()
                print("   (다수클래스 기준: %s)" % ", ".join("%s=%.3f" % (k, v) for k, v in bl.items()))
    out = pd.concat(allr)
    out.to_csv(BASE / "_explore_targets_v5_results.csv", index=False)
    print("\n저장: _explore_targets_v5_results.csv")
