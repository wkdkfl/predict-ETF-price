# -*- coding: utf-8 -*-
"""v17 방향성 — 중첩 워크포워드로 모형·파라미터·피처군 탐색 (28 후보).

사전 고정 프로토콜: ../plan/20261007_direction_v17_nested_plan.md
선택은 각 외부 fold 의 학습 구간 안에서만 이루어지므로 외부 OOS AUC 는 탐색으로 부풀려지지 않는다.

  PCA_SOLVER=full RESULT_DIR=방향성개선_v17 python _direction_v17_nested.py
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402
from _news_variant import RES  # noqa: E402
from _direction_v16_walkforward import (build, train_rows, boot_auc, holm, SEED, N_FOLDS,  # noqa: E402
                                        START_Q, EMBARGO, LGB_KW, SHALLOW_KW, PAPER_TEST_START)

N_INNER, INNER_SPAN, TOP_K = 3, 0.5, 3
MID_KW = dict(n_estimators=500, max_depth=3, learning_rate=0.01, min_child_samples=100,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
              random_state=SEED, n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
CLASSIFIERS = {
    "logit_C0.001": lambda: LogisticRegression(C=0.001, max_iter=2000, solver="lbfgs"),
    "logit_C0.01": lambda: LogisticRegression(C=0.01, max_iter=2000, solver="lbfgs"),
    "logit_C0.1": lambda: LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs"),
    "lgbm_shallow": lambda: lgb.LGBMClassifier(**SHALLOW_KW),
    "lgbm_mid": lambda: lgb.LGBMClassifier(**MID_KW),
    "lgbm_v10": lambda: lgb.LGBMClassifier(**LGB_KW),
    "rf_d4": lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=50,
                                            max_features="sqrt", random_state=SEED, n_jobs=4),
}


# 두 시장 합동 학습 후 시장별 예측 확률
def fit_predict(frames, feats, clf, tr, te):
    """pooled 학습 후 시장별 예측 확률."""
    Xp = np.vstack([frames[m][feats].values[tr[m]] for m in frames])
    yp = np.concatenate([frames[m]["dir5"].values[tr[m]] for m in frames]).astype(int)
    sc = StandardScaler().fit(Xp)
    mdl = CLASSIFIERS[clf]().fit(sc.transform(Xp), yp)
    return {m: mdl.predict_proba(sc.transform(frames[m][feats].values[te[m]]))[:, 1] for m in frames}


# 외부 학습 구간 안 내부 워크포워드로 후보별 AUC 계산
def inner_scores(frames, featsets, tr):
    """외부 학습 구간 뒤쪽 INNER_SPAN 을 N_INNER 구간으로 나눈 내부 워크포워드 AUC (시장별 평균)."""
    dates = np.sort(np.concatenate([frames[m]["Date"].values[tr[m]] for m in frames]))
    lo = int(len(dates) * (1 - INNER_SPAN)); step = (len(dates) - lo) // N_INNER
    edges = [pd.Timestamp(dates[lo + j * step]) for j in range(N_INNER)] + [pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)]
    sc = {(f, c): {m: [] for m in frames} for f in featsets for c in CLASSIFIERS}
    for j in range(N_INNER):
        itr = {m: np.intersect1d(tr[m], train_rows(s, edges[j])) for m, s in frames.items()}
        ite = {m: tr[m][(s["Date"].values[tr[m]] >= np.datetime64(edges[j]))
                        & (s["Date"].values[tr[m]] < np.datetime64(edges[j + 1]))]
               for m, s in frames.items()}
        for f, feats in featsets.items():
            for c in CLASSIFIERS:
                p = fit_predict(frames, feats, c, itr, ite)
                for m, s in frames.items():
                    sc[(f, c)][m].append(roc_auc_score(s["dir5"].values[ite[m]].astype(int), p[m]))
    return {k: {m: float(np.mean(v[m])) for m in frames} for k, v in sc.items()}


# 중첩 워크포워드로 후보를 고르고 다음 fold 에서 평가
def main():
    US, keep_us, ar_us = build("US")
    UK, keep_uk, ar_uk = build("UK")
    frames = {"US": US, "UK": UK}
    from _run_enhanced_models_v4 import NEWS_SENT_COLS
    full_c = [c for c in keep_us if c in keep_uk]
    emb_c = [c for c in full_c if c.startswith("emb_pc")]
    sent_c = [c for c in full_c if c in NEWS_SENT_COLS]
    ar_c = [c for c in full_c if c in ar_us and c in ar_uk]
    fin_clean = [c for c in full_c if c not in emb_c + sent_c + ar_c and not c.startswith("sent_")]
    featsets = {"AR": ar_c, "FIN_clean": fin_clean, "AR+FIN_clean": ar_c + fin_clean, "Full": full_c}

    us_dates = US["Date"].values
    n = len(us_dates); a0 = int(n * START_Q); fs = (n - a0) // N_FOLDS
    edges = [pd.Timestamp(us_dates[a0 + k * fs]) for k in range(N_FOLDS)] + [pd.Timestamp.max]

    oos = {m: {"Date": [], "y": [], "fold": [], "best": [], "top3": []} for m in frames}
    sel_rows = []
    # fold 별 후보 선택과 외부 평가
    for k in range(N_FOLDS):
        tr = {m: train_rows(s, edges[k]) for m, s in frames.items()}
        te = {m: np.where(((s["Date"] >= edges[k]) & (s["Date"] < edges[k + 1])).values)[0]
              for m, s in frames.items()}
        scores = inner_scores(frames, featsets, tr)
        cache = {}
        for m, s in frames.items():
            ranked = sorted(scores, key=lambda key: -scores[key][m])
            for key in ranked[:TOP_K]:
                if key not in cache:
                    cache[key] = fit_predict(frames, featsets[key[0]], key[1], tr, te)
            oos[m]["best"].append(cache[ranked[0]][m])
            oos[m]["top3"].append(np.mean([cache[key][m] for key in ranked[:TOP_K]], axis=0))
            oos[m]["Date"].append(s["Date"].values[te[m]])
            oos[m]["y"].append(s["dir5"].values[te[m]].astype(int))
            oos[m]["fold"].append(np.full(len(te[m]), k + 1))
            y_f = s["dir5"].values[te[m]].astype(int)
            for r, key in enumerate(ranked[:TOP_K]):
                sel_rows.append(dict(fold=k + 1, market=m, rank=r + 1, featset=key[0], clf=key[1],
                                     inner_auc=scores[key][m],
                                     outer_auc=roc_auc_score(y_f, cache[key][m])))
            print("fold %d %s  선택: %-13s %-13s 내부 %.3f -> 외부 %.3f"
                  % (k + 1, m, ranked[0][0], ranked[0][1], scores[ranked[0]][m],
                     roc_auc_score(y_f, cache[ranked[0]][m])), flush=True)

    # 선택 규칙별 OOS 지표 계산
    rows, pred_frames = [], []
    for m in frames:
        o = {key: np.concatenate(v) for key, v in oos[m].items()}
        pred_frames.append(pd.DataFrame({"market": m, **o}))
        paper = pd.to_datetime(o["Date"]) >= PAPER_TEST_START
        for rule in ["best", "top3"]:
            b = boot_auc(o["y"], o[rule])
            folds = [roc_auc_score(o["y"][o["fold"] == f], o[rule][o["fold"] == f])
                     for f in range(1, N_FOLDS + 1)]
            rows.append(dict(market=m, rule=rule, n_oos=len(o["y"]), auc=roc_auc_score(o["y"], o[rule]),
                             ci_low=b["ci_low"], ci_high=b["ci_high"], p_gt_half=b["p_gt_half"],
                             mean_fold_auc=float(np.mean(folds)),
                             folds_above_half=int(sum(a > 0.5 for a in folds)),
                             auc_paper_test_ref=roc_auc_score(o["y"][paper], o[rule][paper])))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p_gt_half"].values)

    print("\n" + "=" * 96)
    print("  중첩 워크포워드 OOS (28 후보, 선택은 과거 자료 안에서만)")
    for _, r in res.iterrows():
        print("  %-4s %-5s n=%d  AUC %.3f [%.3f, %.3f]  Holm p=%.3f  fold>0.5 %d/%d  fold평균 %.3f  논문시험(참고) %.3f"
              % (r.market, r.rule, r.n_oos, r.auc, r.ci_low, r.ci_high, r.p_holm,
                 r.folds_above_half, N_FOLDS, r.mean_fold_auc, r.auc_paper_test_ref))

    res.to_csv(RES / "v17_nested_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sel_rows).to_csv(RES / "v17_nested_selections.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_frames).to_csv(RES / "v17_nested_predictions.csv", index=False, encoding="utf-8-sig")
    print("\n저장: %s" % RES)


if __name__ == "__main__":
    main()
