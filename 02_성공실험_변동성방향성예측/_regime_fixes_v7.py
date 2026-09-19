# -*- coding: utf-8 -*-
"""v7 — 레짐 이동에 대한 표준 처방 3종 비교 (두 시장에 동일 적용).

진단
----
영국 시험 구간(2024-03~2025-12)의 log RV5 평균이 학습 구간보다 0.30 낮다
(미국은 0.02 차이). HAR 모형의 시험구간 예측 편향이 영국 +0.123 / 미국 -0.007 로,
영국의 음(-)의 R² 는 상관관계 부재가 아니라 수준(level) 편향에서 비롯된다.

처방
----
  A. 절편 재보정 : 검증 구간의 평균 잔차만큼 예측값을 평행이동
  B. 차분 타깃   : log RV5(t+1..t+5) - log RV5(t-4..t) 를 예측한 뒤 되돌림
                   (수준 이동에 불변, HAR 문헌의 표준 변형)
  C. 롤링 재적합 : 확장 윈도우로 250일마다 재적합

세 처방 모두 미래 정보를 쓰지 않으며(재보정은 검증 구간, 재적합은 과거만 사용),
미국·영국에 동일하게 적용해 시장별 취사선택을 배제한다.
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
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
from _news_variant import make_pca  # noqa: E402

PCA_DIM, MAX_MISS, EPS, H = 15, 0.05, 1e-8, 5
KW = dict(n_estimators=500, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED, n_jobs=-1, verbose=-1)


def prep(mc):
    df0, emb = load_market(mc)
    df = add_targets_and_ar(df0)
    tg = make_targets(df0)
    i0 = int(len(df) * TRAIN_FRAC)
    pca = make_pca(PCA_DIM, SEED).fit(emb[:i0])
    feat, fin, news, ar = build_features(df, pca.transform(emb).astype(np.float32))
    for c in tg.columns:
        feat[c] = tg[c].values
    meta = {"target_vol", "target_log_vol", "target_dir", "next_ret", "Date"}
    tc = [c for c in feat.columns if c.startswith(("vol", "dir"))]
    cand = [c for c in feat.columns if c not in meta and c not in tc]
    f2 = feat.dropna(subset=["target_vol"])
    miss = f2[cand].isna().mean()
    keep = [c for c in cand if miss[c] <= MAX_MISS]
    ar_c = [c for c in keep if c in ar]
    # 과거 5일 실현변동성 (차분 타깃의 기준선, t 시점 관측 가능)
    r = np.log(df0["ETF"].astype(float)).diff()
    feat["rv5_past"] = np.log(np.sqrt((r**2).rolling(H, min_periods=H).mean()) + EPS).values
    s = feat.dropna(subset=["vol5", "rv5_past"]).dropna(subset=keep).reset_index(drop=True)
    s = s[np.isfinite(s["vol5"]) & np.isfinite(s["rv5_past"])].reset_index(drop=True)
    return s, keep, ar_c


def fit_pred(Xtr, ytr, Xte):
    return lgb.LGBMRegressor(**KW).fit(Xtr, ytr).predict(Xte)



if __name__ == "__main__":
    for mc, lab in [("US", "미국"), ("UK", "영국")]:
        s, keep, ar_c = prep(mc)
        n = len(s); i_tr = int(n * TRAIN_FRAC); i_te = int(n * (TRAIN_FRAC + VAL_FRAC))
        y = s["vol5"].values
        past = s["rv5_past"].values
        yte = y[i_te:]
        print("=" * 74)
        print("[%s]  n=%d  학습 %d / 검증 %d / 시험 %d" % (lab, n, i_tr, i_te - i_tr, n - i_te))
        print("  타깃 평균  학습 %.3f -> 시험 %.3f  (차이 %+.3f)"
              % (y[:i_te].mean(), yte.mean(), yte.mean() - y[:i_te].mean()))

        for gname, cols in [("AR_Only", ar_c), ("Full", keep)]:
            X = s[cols].values
            sc = StandardScaler().fit(X[:i_te])
            Xtr_all, Xte = sc.transform(X[:i_te]), sc.transform(X[i_te:])
            sc2 = StandardScaler().fit(X[:i_tr])
            Xtr, Xva = sc2.transform(X[:i_tr]), sc2.transform(X[i_tr:i_te])

            # 기준 (v6 방식)
            p0 = fit_pred(Xtr_all, y[:i_te], Xte)
            r0 = r2_score(yte, p0)

            # A. 절편 재보정 (검증 구간 잔차 평균)
            mA = lgb.LGBMRegressor(**KW).fit(Xtr, y[:i_tr])
            bias = (mA.predict(Xva) - y[i_tr:i_te]).mean()
            pA = fit_pred(Xtr_all, y[:i_te], Xte) - bias
            rA = r2_score(yte, pA)

            # B. 차분 타깃
            d = y - past
            pB = fit_pred(Xtr_all, d[:i_te], Xte) + past[i_te:]
            rB = r2_score(yte, pB)

            # C. 롤링 재적합 (250일마다)
            pC = np.empty(len(yte))
            step = 250
            for st in range(0, len(yte), step):
                en = min(st + step, len(yte))
                cut = i_te + st
                sc3 = StandardScaler().fit(X[:cut])
                pC[st:en] = fit_pred(sc3.transform(X[:cut]), y[:cut],
                                     sc3.transform(X[cut:i_te + en]))
            rC = r2_score(yte, pC)

            # B+C 결합
            pBC = np.empty(len(yte))
            for st in range(0, len(yte), step):
                en = min(st + step, len(yte))
                cut = i_te + st
                sc3 = StandardScaler().fit(X[:cut])
                pBC[st:en] = fit_pred(sc3.transform(X[:cut]), d[:cut],
                                      sc3.transform(X[cut:i_te + en])) + past[cut:i_te + en]
            rBC = r2_score(yte, pBC)

            print("  [%s]  기준 %+.3f | A.재보정 %+.3f | B.차분타깃 %+.3f | C.롤링재적합 %+.3f | B+C %+.3f"
                  % (gname, r0, rA, rB, rC, rBC))
            print("        상관계수: 기준 %.3f  B %.3f  B+C %.3f"
                  % (np.corrcoef(yte, p0)[0, 1], np.corrcoef(yte, pB)[0, 1],
                     np.corrcoef(yte, pBC)[0, 1]))
