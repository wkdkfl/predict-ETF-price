# -*- coding: utf-8 -*-
"""v6 최종 파이프라인 — 논문 게재용 수치 산출.

v4 대비 변경점
--------------
1. 거래일 필터 (v4에서 도입) 유지
2. 결측 과다 열 제거로 표본 회복 (eth 등: 미국 1,797 -> 3,011행)
3. 예측 타깃 확장: 1일 |r| 외에 5일·10일 실현변동성 RV_h 추가
   RV_h(t) = sqrt( (1/h) * Σ_{i=1..h} r²_{t+i} ),  타깃 = log RV_h
4. 방향성도 1일·5일 누적 두 시계
5. 5개 피처군 어블레이션 + DM 검정 + 블록 부트스트랩 + Walk-Forward

출력: {USD,UK}/results_v6/
"""
from __future__ import annotations

import json, math, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from _run_enhanced_models_v4 import (  # noqa: E402
    load_market, add_targets_and_ar, build_features,
    TRAIN_FRAC, VAL_FRAC, SEED, NEWS_SENT_COLS,
    xgb_reg_optuna, lgbm_reg_optuna, rf_reg, ridge_reg, stacking_reg,
    xgb_cls_optuna, lgbm_cls_optuna, rf_cls, logit_cls, stacking_cls,
    diebold_mariano, dm_classification, block_bootstrap_metric,
)
from _explore_targets_v5 import make_targets  # noqa: E402

import lightgbm as lgb  # noqa: E402

PCA_DIM = 15
MAX_MISSING_FRAC = 0.05
OPTUNA_TRIALS = 50
VOL_TARGETS = ["vol1", "vol5", "vol10"]
DIR_TARGETS = ["dir1", "dir5"]
GROUP_ORDER = ["AR_Only", "Financial_Only", "News_Pure", "AR+News", "Full"]


def prepare(market_code: str):
    df0, emb = load_market(market_code)
    df = add_targets_and_ar(df0)
    tg = make_targets(df0)
    i0 = int(len(df) * TRAIN_FRAC)
    pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(emb[:i0])
    print("  PCA(%d) explained: %.3f" % (PCA_DIM, pca.explained_variance_ratio_.sum()))
    feat, fin_cols, news_cols, ar_cols = build_features(df, pca.transform(emb).astype(np.float32))
    for c in tg.columns:
        feat[c] = tg[c].values

    meta = {"target_vol", "target_log_vol", "target_dir", "next_ret", "Date"}
    tcols = [c for c in feat.columns if c.startswith(("vol", "dir"))]
    cand = [c for c in feat.columns if c not in meta and c not in tcols]
    f2 = feat.dropna(subset=["target_vol"])
    miss = f2[cand].isna().mean()
    dropped = sorted(miss[miss > MAX_MISSING_FRAC].index)
    keep = [c for c in cand if c not in dropped]
    print("  결측 과다로 제외한 열 %d개 -> 사용 피처 %d개" % (len(dropped), len(keep)))

    emb_c = [c for c in keep if c.startswith("emb_pc")]
    sent_c = [c for c in keep if c in NEWS_SENT_COLS]
    ar_c = [c for c in keep if c in ar_cols]
    fin_c = [c for c in keep if c not in emb_c + sent_c + ar_c]
    groups = {"AR_Only": ar_c, "Financial_Only": fin_c, "News_Pure": sent_c + emb_c,
              "AR+News": ar_c + sent_c + emb_c, "Full": keep}
    print("     AR %d / 금융 %d / 감성 %d / 임베딩 %d"
          % (len(ar_c), len(fin_c), len(sent_c), len(emb_c)))
    return feat, groups, keep


def split(sub):
    n = len(sub)
    return int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))


def frame_for(feat, keep, tname):
    s = feat.dropna(subset=[tname]).dropna(subset=keep).reset_index(drop=True)
    return s[np.isfinite(s[tname])].reset_index(drop=True)


def wf(sub, cols, tname, is_dir, n_folds=5):
    n = len(sub); mt = int(n * 0.40); fs = (n - mt) // n_folds
    rows = []
    for k in range(n_folds):
        a = mt + k * fs; b = min(a + fs, n)
        if b <= a:
            continue
        tr, te = sub.iloc[:a], sub.iloc[a:b]
        sc = StandardScaler().fit(tr[cols].values)
        Xtr, Xte = sc.transform(tr[cols].values), sc.transform(te[cols].values)
        kw = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
                  n_jobs=-1, verbose=-1)
        if is_dir:
            if te[tname].nunique() < 2:
                continue
            m = lgb.LGBMClassifier(**kw).fit(Xtr, tr[tname].values.astype(int))
            s = roc_auc_score(te[tname].values.astype(int), m.predict_proba(Xte)[:, 1])
        else:
            m = lgb.LGBMRegressor(**kw).fit(Xtr, tr[tname].values)
            s = r2_score(te[tname].values, m.predict(Xte))
        rows.append({"fold": k + 1, "train_size": len(tr), "test_size": len(te),
                     "score": s, "test_start": str(te["Date"].iloc[0].date()),
                     "test_end": str(te["Date"].iloc[-1].date())})
    return pd.DataFrame(rows)


def run_market(mc: str):
    print("=" * 72)
    print("  %s -- V6 FINAL" % mc)
    feat, groups, keep = prepare(mc)
    sub_dir = "USD" if mc == "US" else "UK"
    out = BASE / sub_dir / "results_v6"
    out.mkdir(exist_ok=True)

    model_rows, abl_rows, dm_rows, boot_rows, wf_rows = [], [], [], [], []

    # ---------------- 변동성 ----------------
    for tname in VOL_TARGETS:
        s = frame_for(feat, keep, tname)
        i_tr, i_te = split(s)
        y = s[tname].values.astype(np.float32)
        print("\n  [%s] n=%d  train %d / val %d / test %d  (%s ~ %s)"
              % (tname, len(s), i_tr, i_te - i_tr, len(s) - i_te,
                 s["Date"].iloc[i_te].date(), s["Date"].iloc[-1].date()))
        # (a) 모델 비교 — Full 피처
        cols = groups["Full"]
        sc = StandardScaler().fit(s[cols].values[:i_tr])
        Xtr, Xva, Xte = (sc.transform(s[cols].values[:i_tr]),
                         sc.transform(s[cols].values[i_tr:i_te]),
                         sc.transform(s[cols].values[i_te:]))
        ytr, yva, yte = y[:i_tr], y[i_tr:i_te], y[i_te:]
        preds = {}
        m, p, _, _ = xgb_reg_optuna(Xtr, ytr, Xva, yva, Xte, yte, OPTUNA_TRIALS)
        preds["XGBoost"] = p; model_rows.append(dict(target=tname, model="XGBoost", **m))
        m, p, _, _ = lgbm_reg_optuna(Xtr, ytr, Xva, yva, Xte, yte, OPTUNA_TRIALS)
        preds["LightGBM"] = p; model_rows.append(dict(target=tname, model="LightGBM", **m))
        m, p, _ = rf_reg(Xtr, ytr, Xte, yte)
        preds["RandomForest"] = p; model_rows.append(dict(target=tname, model="RandomForest", **m))
        m, p = ridge_reg(Xtr, ytr, Xte, yte)
        preds["Ridge"] = p; model_rows.append(dict(target=tname, model="Ridge", **m))
        m, p = stacking_reg(Xtr, ytr, Xva, yva, Xte, yte, OPTUNA_TRIALS)
        preds["Stacking"] = p; model_rows.append(dict(target=tname, model="Stacking", **m))
        for k, v in preds.items():
            print("     %-13s R2=%+.4f" % (k, r2_score(yte, v)))
            pt, lo, hi = block_bootstrap_metric(yte, v, r2_score, block_size=10)
            boot_rows.append(dict(target=tname, model=k, point=pt, ci_low=lo, ci_high=hi))

        # (b) 어블레이션 — LightGBM 고정
        ab_pred = {}
        for g in GROUP_ORDER:
            gc = groups[g]
            if not gc:
                continue
            sc2 = StandardScaler().fit(s[gc].values[:i_te])
            m2 = lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                   random_state=SEED, n_jobs=-1, verbose=-1)
            m2.fit(sc2.transform(s[gc].values[:i_te]), y[:i_te])
            pp = m2.predict(sc2.transform(s[gc].values[i_te:]))
            ab_pred[g] = pp
            abl_rows.append(dict(target=tname, group=g, n_feat=len(gc),
                                 r2=r2_score(yte, pp)))
            w = wf(s, gc, tname, False)
            wf_rows.append(dict(target=tname, group=g, mean=w["score"].mean(),
                                sd=w["score"].std(ddof=1), n_folds=len(w)))
            print("     [ABL] %-15s R2=%+.4f   WF=%+.4f ± %.3f"
                  % (g, r2_score(yte, pp), w["score"].mean(), w["score"].std(ddof=1)))
        for g in GROUP_ORDER:
            if g == "Full" or g not in ab_pred:
                continue
            d, pv = diebold_mariano(yte, ab_pred[g], ab_pred["Full"])
            dm_rows.append(dict(target=tname, baseline=g, vs="Full", dm=d, p=pv))

    # ---------------- 방향성 ----------------
    for tname in DIR_TARGETS:
        s = frame_for(feat, keep, tname)
        i_tr, i_te = split(s)
        y = s[tname].values.astype(int)
        bl = max(y[i_te:].mean(), 1 - y[i_te:].mean())
        print("\n  [%s] n=%d  test %d  다수클래스=%.3f" % (tname, len(s), len(y) - i_te, bl))
        cols = groups["Full"]
        sc = StandardScaler().fit(s[cols].values[:i_tr])
        Xtr, Xva, Xte = (sc.transform(s[cols].values[:i_tr]),
                         sc.transform(s[cols].values[i_tr:i_te]),
                         sc.transform(s[cols].values[i_te:]))
        ytr, yva, yte = y[:i_tr], y[i_tr:i_te], y[i_te:]
        preds = {}
        m, p, _, _ = xgb_cls_optuna(Xtr, ytr, Xva, yva, Xte, yte, OPTUNA_TRIALS)
        preds["XGBoost"] = p; model_rows.append(dict(target=tname, model="XGBoost", baseline=bl, **m))
        m, p, _, _ = lgbm_cls_optuna(Xtr, ytr, Xva, yva, Xte, yte, OPTUNA_TRIALS)
        preds["LightGBM"] = p; model_rows.append(dict(target=tname, model="LightGBM", baseline=bl, **m))
        m, p, _ = rf_cls(Xtr, ytr, Xte, yte)
        preds["RandomForest"] = p; model_rows.append(dict(target=tname, model="RandomForest", baseline=bl, **m))
        m, p = logit_cls(Xtr, ytr, Xte, yte)
        preds["Logit"] = p; model_rows.append(dict(target=tname, model="Logit", baseline=bl, **m))
        m, p = stacking_cls(Xtr, ytr, Xva, yva, Xte, yte, OPTUNA_TRIALS)
        preds["Stacking"] = p; model_rows.append(dict(target=tname, model="Stacking", baseline=bl, **m))
        for k, v in preds.items():
            print("     %-13s AUC=%.4f" % (k, roc_auc_score(yte, v)))
            pt, lo, hi = block_bootstrap_metric(yte.astype(float), v, roc_auc_score,
                                                block_size=max(10, int(tname[3:])))
            boot_rows.append(dict(target=tname, model=k, point=pt, ci_low=lo, ci_high=hi))

        ab_pred = {}
        for g in GROUP_ORDER:
            gc = groups[g]
            if not gc:
                continue
            sc2 = StandardScaler().fit(s[gc].values[:i_te])
            m2 = lgb.LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                    random_state=SEED, n_jobs=-1, verbose=-1)
            m2.fit(sc2.transform(s[gc].values[:i_te]), y[:i_te])
            pp = m2.predict_proba(sc2.transform(s[gc].values[i_te:]))[:, 1]
            ab_pred[g] = pp
            abl_rows.append(dict(target=tname, group=g, n_feat=len(gc),
                                 auc=roc_auc_score(yte, pp),
                                 acc=accuracy_score(yte, (pp > 0.5).astype(int)),
                                 baseline=bl))
            w = wf(s, gc, tname, True)
            wf_rows.append(dict(target=tname, group=g, mean=w["score"].mean(),
                                sd=w["score"].std(ddof=1), n_folds=len(w)))
            print("     [ABL] %-15s AUC=%.4f   WF=%.4f ± %.3f"
                  % (g, roc_auc_score(yte, pp), w["score"].mean(), w["score"].std(ddof=1)))
        for g in GROUP_ORDER:
            if g == "Full" or g not in ab_pred:
                continue
            d, pv = dm_classification(yte.astype(float), ab_pred[g], ab_pred["Full"])
            dm_rows.append(dict(target=tname, baseline=g, vs="Full", dm=d, p=pv))

    for nm, rows in [("model_comparison", model_rows), ("ablation", abl_rows),
                     ("dm_tests", dm_rows), ("bootstrap_ci", boot_rows),
                     ("walk_forward", wf_rows)]:
        pd.DataFrame(rows).to_csv(out / f"{nm}.csv", index=False)
    print("\n  [저장] %s" % out)


if __name__ == "__main__":
    for m in ["US", "UK"]:
        run_market(m)
