# -*- coding: utf-8 -*-
"""
AR_Only Ablation Experiment
============================
기존 Ablation (Financial_Only / News_Only / Full) 에
'AR_Only' 그룹을 추가하여 뉴스의 순수 기여를 분리한다.

피처 그룹 정의
--------------
  AR_Only      : ret_lag, rolling_absret, leverage 등 가격 기반 자기회귀 피처만
  Financial_Only : 외부 시장 지표 (Gold, VIX, Bond 등)만
  News_Only    : 감성 피처 + FinBERT 임베딩만  (AR 제외)  ← 기존과 다름
  AR+News      : AR_Only + News_Only  (= 기존 News_Only)
  Full         : 전체 피처

이 설계로 뉴스의 '순수 기여'를 측정한다:
  뉴스 기여 = Full R² - AR_Only R²
  또는       = (AR+News) R² - AR_Only R²
"""

from __future__ import annotations
import json, math, warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
SEED = 42
np.random.seed(SEED)

EMB_PCA_DIM   = 50
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
OPTUNA_TRIALS = 60   # ablation용 빠른 실행
LOG_VOL_EPS   = 1e-8

NEWS_SENT_COLS = [
    "news_volume", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "sent_score_mean", "sent_score_std", "sent_max_neg",
    "sent_pos_ratio", "sent_neg_ratio",
    "sent_score_ma5", "sent_score_ma20", "sent_surprise",
    "sent_momentum_3d", "sent_vol_5d", "news_volume_chg",
    "news_available",
]
EXCLUDE_AS_FEATURE = {"Date", "Headline", "ETF"}

NEW_AR_COLS = [
    "rolling_absret_22", "rolling_std_22",
    "leverage_5d", "leverage_20d", "sign_run",
    "rsi_14", "dow", "vol_ratio_5_20",
    "absret_surprise", "log_absret_lag1",
]


# ─────────────────────────────────────────────────────────
# Import helpers from original v3 (reuse data + model code)
# ─────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(BASE))

from _run_enhanced_models_v4 import (
    load_market, add_targets_and_ar, build_features, time_split,
    xgb_reg_optuna, ridge_reg, rf_reg,
    xgb_cls_optuna, logit_cls, rf_cls,
    metrics_reg, metrics_cls,
    diebold_mariano, dm_classification,
)


# ─────────────────────────────────────────────────────────
# Feature group splitter - 핵심 함수
# ─────────────────────────────────────────────────────────

def split_feature_groups(feat: pd.DataFrame,
                          fin_cols: List[str],
                          news_cols: List[str],
                          ar_cols: List[str],
                          emb_dim: int = EMB_PCA_DIM):
    """
    피처를 4개 그룹으로 분리한다.

    반환:
        groups: {그룹명: [컬럼명 리스트]} 딕셔너리
    """
    emb_cols  = [f"emb_pc{i+1}" for i in range(emb_dim)
                 if f"emb_pc{i+1}" in feat.columns]

    # AR = 가격 기반 자기회귀 피처 (뉴스 무관)
    ar_group   = [c for c in ar_cols if c in feat.columns]

    # Financial = 외부 시장 지표
    fin_group  = [c for c in fin_cols if c in feat.columns]

    # News_Pure = 감성 점수 + FinBERT 임베딩 (AR 제외)
    sent_cols  = [c for c in news_cols if c in feat.columns]
    news_group = sent_cols + emb_cols

    # Full = 전부
    all_cols   = [c for c in feat.columns
                  if c not in {"target_vol", "target_log_vol",
                               "target_dir", "next_ret", "Date"}]

    groups = {
        "AR_Only"      : ar_group,
        "Financial_Only": fin_group,
        "News_Pure"    : news_group,        # ← 새 정의: AR 없이 순수 뉴스만
        "AR+News"      : ar_group + news_group,  # ← 기존 News_Only와 동일
        "Full"         : all_cols,
    }

    print("\n  피처 그룹 구성:")
    for name, cols in groups.items():
        print(f"    {name:15s}: {len(cols):3d}개")

    return groups


def run_group(name: str, cols: List[str],
              train, val, test,
              task: str = "reg") -> dict:
    """하나의 피처 그룹에 대해 XGBoost 실험 실행."""
    import xgboost as xgb

    if len(cols) == 0:
        print(f"    [{name}] 피처 없음 → skip")
        return {}

    sc = StandardScaler().fit(train[cols].fillna(0).values)
    Xtr = sc.transform(train[cols].fillna(0).values)
    Xva = sc.transform(val[cols].fillna(0).values)
    Xte = sc.transform(test[cols].fillna(0).values)

    if task == "reg":
        ytr = train["target_log_vol"].values.astype(np.float32)
        yva = val["target_log_vol"].values.astype(np.float32)
        yte_lv = test["target_log_vol"].values.astype(np.float32)
        yte_v  = test["target_vol"].values.astype(np.float32)

        m, pred_lv, _, _ = xgb_reg_optuna(Xtr, ytr, Xva, yva, Xte, yte_lv,
                                            OPTUNA_TRIALS)
        pred_v = np.clip(np.exp(pred_lv) - LOG_VOL_EPS, 0, None)
        m_orig = metrics_reg(yte_v, pred_v)
        print(f"    [{name}] log-R²={m['r2']:+.4f}  orig-R²={m_orig['r2']:+.4f}  "
              f"RMSE={m_orig['rmse']:.5f}  corr={m_orig['corr']:+.3f}")
        return {"group": name, "task": "vol",
                "r2_log": m["r2"], **m_orig,
                "pred": pred_v, "pred_lv": pred_lv}

    else:  # cls
        ytr = train["target_dir"].values.astype(np.int32)
        yva = val["target_dir"].values.astype(np.int32)
        yte = test["target_dir"].values.astype(np.int32)

        m, prob, _, _ = xgb_cls_optuna(Xtr, ytr, Xva, yva, Xte, yte,
                                        OPTUNA_TRIALS)
        print(f"    [{name}] ACC={m['acc']:.4f}  AUC={m['auc']:.4f}  "
              f"F1={m['f1']:.4f}")
        return {"group": name, "task": "dir", **m, "prob": prob}


# ─────────────────────────────────────────────────────────
# Main ablation runner
# ─────────────────────────────────────────────────────────

def run_ablation(market_code: str) -> None:
    print(f"\n{'='*65}")
    print(f"  AR_Only Ablation  -  {market_code} market")
    print(f"{'='*65}")

    sub     = "USD" if market_code == "US" else "UK"
    out_dir = BASE / sub / "results_ar_ablation_v4"
    out_dir.mkdir(exist_ok=True)

    # ── Load data (same as v3) ──
    df, emb = load_market(market_code)
    df = add_targets_and_ar(df)

    n_total = len(df)
    i_tr    = int(n_total * TRAIN_FRAC)
    pca     = PCA(n_components=EMB_PCA_DIM, random_state=SEED).fit(emb[:i_tr])
    emb_pca = pca.transform(emb).astype(np.float32)

    feat, fin_cols, news_cols, ar_cols = build_features(df, emb_pca)
    feat = feat.dropna(subset=["target_vol", "target_dir"]).dropna()
    feat = feat.reset_index(drop=True)
    train, val, test = time_split(feat)

    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"  test: {test['Date'].min().date()} → {test['Date'].max().date()}")

    groups = split_feature_groups(feat, fin_cols, news_cols, ar_cols)

    # ── Volatility ablation ──
    print(f"\n  > 변동성 (log-vol) 예측 - XGBoost")
    vol_rows = []
    vol_preds = {}
    for gname, gcols in groups.items():
        row = run_group(gname, gcols, train, val, test, task="reg")
        if row:
            vol_rows.append({k: v for k, v in row.items()
                             if k not in ("pred", "pred_lv")})
            vol_preds[gname] = (row["pred"], row.get("pred_lv"))

    # ── Direction ablation ──
    print(f"\n  > 방향성 분류 - XGBoost")
    dir_rows = []
    dir_probs = {}
    for gname, gcols in groups.items():
        row = run_group(gname, gcols, train, val, test, task="cls")
        if row:
            dir_rows.append({k: v for k, v in row.items() if k != "prob"})
            dir_probs[gname] = row["prob"]

    # ── Diebold-Mariano tests ──
    y_vol = test["target_vol"].values.astype(np.float32)
    y_dir = test["target_dir"].values.astype(np.int32)

    dm_rows = []
    reference = "Full"
    if reference in vol_preds:
        p_ref_v = vol_preds[reference][0]
        p_ref_d = dir_probs[reference]
        for gname in vol_preds:
            if gname == reference:
                continue
            # Vol DM
            dm_stat, p_val = diebold_mariano(y_vol,
                                              vol_preds[gname][0],
                                              p_ref_v)
            dm_rows.append({"market": market_code, "task": "vol",
                            "baseline": gname, "vs": reference,
                            "DM_stat": round(dm_stat, 3),
                            "p_value": round(p_val, 4),
                            "Full_wins (p<0.05)": p_val < 0.05 and dm_stat > 0})
            # Dir DM
            if gname in dir_probs:
                dm_d, p_d = dm_classification(y_dir.astype(float),
                                               dir_probs[gname],
                                               p_ref_d)
                dm_rows.append({"market": market_code, "task": "dir",
                                "baseline": gname, "vs": reference,
                                "DM_stat": round(dm_d, 3),
                                "p_value": round(p_d, 4),
                                "Full_wins (p<0.05)": p_d < 0.05 and dm_d > 0})

    # ── Save results ──
    vol_df = pd.DataFrame(vol_rows)
    dir_df = pd.DataFrame(dir_rows)
    dm_df  = pd.DataFrame(dm_rows)

    vol_df.to_csv(out_dir / "ablation_vol.csv",  index=False)
    dir_df.to_csv(out_dir / "ablation_dir.csv",  index=False)
    dm_df.to_csv( out_dir / "ablation_dm.csv",   index=False)

    # ── Summary printout ──
    print(f"\n  {'─'*60}")
    print(f"  결과 요약 - 변동성 R²(log scale)")
    print(f"  {'그룹':<18} {'log-R²':>8}  {'orig-R²':>8}  {'해석'}")
    for _, r in vol_df.iterrows():
        note = ""
        if r['group'] == 'AR_Only':
            note = "← 가격 패턴만"
        elif r['group'] == 'News_Pure':
            note = "← 뉴스(감성+임베딩)만, AR 없음"
        elif r['group'] == 'AR+News':
            note = "← 기존 News_Only와 동일"
        elif r['group'] == 'Full':
            note = "← 최종 모델"
        print(f"  {r['group']:<18} {r['r2_log']:>8.4f}  {r['r2']:>8.4f}  {note}")

    print(f"\n  결과 요약 - 방향성 AUC")
    for _, r in dir_df.iterrows():
        print(f"  {r['group']:<18} AUC={r['auc']:.4f}  ACC={r['acc']:.4f}")

    print(f"\n  Diebold-Mariano 검정 (vs Full)")
    print(dm_df.to_string(index=False))
    print(f"\n  [저장] {out_dir}")

    # ── 뉴스 순수 기여 계산 ──
    if "AR_Only" in {r['group'] for r in vol_rows} and \
       "Full"    in {r['group'] for r in vol_rows}:
        r_ar  = next(r for r in vol_rows if r['group'] == 'AR_Only')
        r_arn = next((r for r in vol_rows if r['group'] == 'AR+News'), None)
        r_ful = next(r for r in vol_rows if r['group'] == 'Full')

        delta_v = r_ful['r2'] - r_ar['r2']
        print(f"\n  ★ 뉴스 순수 기여 (변동성 R², orig scale)")
        print(f"     AR_Only  R² = {r_ar['r2']:+.4f}")
        if r_arn:
            print(f"     AR+News  R² = {r_arn['r2']:+.4f}  "
                  f"(AR 대비 +{r_arn['r2'] - r_ar['r2']:+.4f})")
        print(f"     Full     R² = {r_ful['r2']:+.4f}  "
              f"(AR 대비 +{delta_v:+.4f})")

        if "AR_Only" in {r['group'] for r in dir_rows} and \
           "Full"    in {r['group'] for r in dir_rows}:
            d_ar  = next(r for r in dir_rows if r['group'] == 'AR_Only')
            d_ful = next(r for r in dir_rows if r['group'] == 'Full')
            print(f"\n  ★ 뉴스 순수 기여 (방향성 AUC)")
            print(f"     AR_Only AUC = {d_ar['auc']:.4f}")
            print(f"     Full    AUC = {d_ful['auc']:.4f}  "
                  f"(AR 대비 +{d_ful['auc'] - d_ar['auc']:+.4f})")


if __name__ == "__main__":
    for market in ["US", "UK"]:
        run_ablation(market)
    print("\n[DONE]")
