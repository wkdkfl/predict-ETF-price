# -*- coding: utf-8 -*-
"""Phase 3 v4 — v3 + 거래일 필터 (비거래일 전방보간 행 제거).

Improvements over v2
--------------------
1. Enhanced features: leverage effect, VIX dynamics, DoW dummies,
   sentiment × regime interactions
2. Log-vol target for better distribution handling
3. LightGBM alongside XGBoost (Optuna-tuned)
4. Stacking ensemble (XGB + LGBM + RF → Ridge meta-learner)
5. Walk-forward validation (5-fold expanding window)
6. Block bootstrap for time-series aware CIs
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent

from _trading_days import filter_trading_days
from _news_variant import DATA_SFX

SEED = 42
np.random.seed(SEED)

EMB_PCA_DIM = 50
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
OPTUNA_TRIALS_FULL = 100
OPTUNA_TRIALS_ABL = 30
BOOTSTRAP_N = 5000
WF_N_FOLDS = 5
LOG_VOL_EPS = 1e-8

NEWS_SENT_COLS = [
    "news_volume", "sent_pos_mean", "sent_neg_mean", "sent_neu_mean",
    "sent_score_mean", "sent_score_std", "sent_max_neg",
    "sent_pos_ratio", "sent_neg_ratio",
    "sent_score_ma5", "sent_score_ma20", "sent_surprise",
    "sent_momentum_3d", "sent_vol_5d", "news_volume_chg",
    "news_available",
]
EXCLUDE_AS_FEATURE = {"Date", "Headline", "ETF"}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def load_market(market_code: str):
    sub = "USD" if market_code == "US" else "UK"
    df = pd.read_csv(BASE / sub / f"{market_code}_research_enhanced{DATA_SFX}.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    emb = np.load(BASE / sub / f"news_embeddings_daily_aligned{DATA_SFX}.npy")
    # === v4: 비거래일(주말/공휴일 전방보간 행) 제거 ===
    df, emb = filter_trading_days(df, market_code, emb)
    return df, emb


def add_targets_and_ar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p = df["ETF"].astype(float)
    ret = np.log(p).diff()

    next_ret = ret.shift(-1)
    df["target_vol"] = next_ret.abs()
    df["target_log_vol"] = np.log(next_ret.abs() + LOG_VOL_EPS)
    df["target_dir"] = (next_ret > 0).astype(int)
    df["next_ret"] = next_ret

    # AR features (observable at t, after close)
    for k in range(1, 6):
        df[f"ret_lag{k}"] = ret.shift(k - 1)
    for k in range(1, 4):
        df[f"absret_lag{k}"] = ret.abs().shift(k - 1)
    df["rolling_mean_5"]  = ret.rolling(5,  min_periods=1).mean()
    df["rolling_mean_20"] = ret.rolling(20, min_periods=1).mean()
    df["rolling_std_5"]   = ret.rolling(5,  min_periods=2).std()
    df["rolling_std_20"]  = ret.rolling(20, min_periods=2).std()
    df["rolling_absret_5"]  = ret.abs().rolling(5,  min_periods=1).mean()
    df["rolling_absret_20"] = ret.abs().rolling(20, min_periods=1).mean()
    df["rolling_skew_20"] = ret.rolling(20, min_periods=5).skew()
    df["rolling_kurt_20"] = ret.rolling(20, min_periods=5).kurt()

    # === NEW v3 features ===
    # HAR monthly component (22-day avg abs return)
    df["rolling_absret_22"] = ret.abs().rolling(22, min_periods=5).mean()
    df["rolling_std_22"]    = ret.rolling(22, min_periods=5).std()

    # Leverage effect: negative returns increase future vol
    df["leverage_5d"] = (ret * (ret < 0).astype(float)).rolling(5, min_periods=1).mean()
    df["leverage_20d"] = (ret * (ret < 0).astype(float)).rolling(20, min_periods=1).mean()

    # Return sign persistence
    df["sign_run"] = _sign_run_length(ret)

    # RSI(14)
    df["rsi_14"] = _rsi(ret, 14)

    # Day of week (0=Mon, 4=Fri)
    df["dow"] = df["Date"].dt.dayofweek.astype(float)

    # VIX dynamics (will be computed for VIX column in build_features)
    # Realized vol ratio (short vs long term vol regime indicator)
    if df["rolling_std_5"].std() > 0 and df["rolling_std_20"].std() > 0:
        df["vol_ratio_5_20"] = df["rolling_std_5"] / df["rolling_std_20"].clip(lower=1e-8)

    # Absolute return surprise (today vs recent avg)
    df["absret_surprise"] = ret.abs() / df["rolling_absret_20"].clip(lower=1e-8)

    # Log-volume of abs return (to better capture tails)
    df["log_absret_lag1"] = np.log(ret.abs().shift(0) + LOG_VOL_EPS)

    return df


def _sign_run_length(ret: pd.Series) -> pd.Series:
    """Count consecutive same-sign returns ending at each day."""
    signs = np.sign(ret.values)
    runs = np.ones(len(signs))
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1] and signs[i] != 0:
            runs[i] = runs[i-1] + 1
    return pd.Series(runs, index=ret.index)


def _rsi(ret: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index from log returns."""
    gain = ret.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-ret).clip(lower=0).rolling(period, min_periods=1).mean()
    rs = gain / loss.clip(lower=1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def build_features(df: pd.DataFrame, emb_pca: np.ndarray):
    # v3 new AR/technical columns
    new_ar_cols = [
        "rolling_absret_22", "rolling_std_22",
        "leverage_5d", "leverage_20d", "sign_run",
        "rsi_14", "dow", "vol_ratio_5_20",
        "absret_surprise", "log_absret_lag1",
    ]

    fin_cols = [c for c in df.columns
                if c not in EXCLUDE_AS_FEATURE
                and c not in NEWS_SENT_COLS
                and not c.startswith("target_")
                and c != "next_ret"
                and not c.startswith("ret_lag")
                and not c.startswith("absret_lag")
                and not c.startswith("rolling_")
                and c not in new_ar_cols]
    news_cols = [c for c in NEWS_SENT_COLS if c in df.columns]
    ar_cols = [c for c in df.columns
               if c.startswith("ret_lag")
               or c.startswith("absret_lag")
               or c.startswith("rolling_")]
    ar_cols += [c for c in new_ar_cols if c in df.columns]
    # set()은 PYTHONHASHSEED에 따라 순회 순서가 달라져 실행마다 컬럼 순서가 바뀐다.
    # colsample_bytree가 컬럼 순서에 의존하므로 순서를 보존하며 중복만 제거한다.
    ar_cols = list(dict.fromkeys(ar_cols))

    feat = pd.DataFrame(index=df.index)
    for c in fin_cols:
        feat[c] = df[c].shift(1)  # financial features at t-1
    for c in news_cols + ar_cols:
        feat[c] = df[c]

    # Sentiment × VIX interaction (captures regime-dependent sentiment)
    if "sent_score_mean" in df.columns:
        vix_col = df.get("vix")
        if vix_col is not None:
            feat["sent_x_vix"] = df["sent_score_mean"] * vix_col.shift(1)
        feat["sent_x_vol"] = df["sent_score_mean"] * df.get("rolling_std_5", 0)

    # VIX dynamics
    if "vix" in df.columns:
        vix = df["vix"].shift(1)
        feat["vix_chg_1d"] = vix.diff()
        feat["vix_chg_5d"] = vix.diff(5)
        feat["vix_ma5_ratio"] = vix / vix.rolling(5, min_periods=1).mean().clip(lower=0.01)

    for i in range(emb_pca.shape[1]):
        feat[f"emb_pc{i+1}"] = emb_pca[:, i]
    feat["target_vol"]     = df["target_vol"]
    feat["target_log_vol"] = df["target_log_vol"]
    feat["target_dir"]     = df["target_dir"]
    feat["next_ret"]       = df["next_ret"]
    feat["Date"]           = df["Date"]
    return feat, fin_cols, news_cols, ar_cols


def time_split(feat: pd.DataFrame):
    n = len(feat)
    i_tr = int(n * TRAIN_FRAC)
    i_va = int(n * (TRAIN_FRAC + VAL_FRAC))
    return feat.iloc[:i_tr].copy(), feat.iloc[i_tr:i_va].copy(), feat.iloc[i_va:].copy()


# ---------------------------------------------------------------------------
# Regression models: realized volatility
# ---------------------------------------------------------------------------

def metrics_reg(y, p):
    return {"r2": float(r2_score(y, p)),
            "rmse": float(math.sqrt(mean_squared_error(y, p))),
            "mae":  float(mean_absolute_error(y, p)),
            "corr": float(np.corrcoef(y, p)[0, 1]) if y.std() > 0 and p.std() > 0 else 0.0}


def xgb_reg_optuna(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials):
    import optuna, xgboost as xgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        params = dict(
            n_estimators       = t.suggest_int("n_estimators", 200, 1500, step=100),
            max_depth          = t.suggest_int("max_depth", 3, 10),
            learning_rate      = t.suggest_float("learning_rate", 0.003, 0.15, log=True),
            subsample          = t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree   = t.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight   = t.suggest_int("min_child_weight", 1, 15),
            reg_alpha          = t.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
            reg_lambda         = t.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            gamma              = t.suggest_float("gamma", 0, 1.0),
            random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0,
            early_stopping_rounds=40, eval_metric="rmse")
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return mean_squared_error(y_va, m.predict(X_va))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params,
                random_state=SEED, n_jobs=-1, tree_method="hist",
                verbosity=0, eval_metric="rmse", early_stopping_rounds=40)
    m = xgb.XGBRegressor(**best)
    X_full = np.vstack([X_tr, X_va]); y_full = np.concatenate([y_tr, y_va])
    m.fit(X_full, y_full, eval_set=[(X_va, y_va)], verbose=False)
    pred = m.predict(X_te)
    return metrics_reg(y_te, pred), pred, m, study.best_params


def lgbm_reg_optuna(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials):
    import optuna, lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        params = dict(
            n_estimators       = t.suggest_int("n_estimators", 200, 1500, step=100),
            max_depth          = t.suggest_int("max_depth", 3, 12),
            learning_rate      = t.suggest_float("learning_rate", 0.003, 0.15, log=True),
            subsample          = t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree   = t.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_samples  = t.suggest_int("min_child_samples", 5, 50),
            reg_alpha          = t.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
            reg_lambda         = t.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            num_leaves         = t.suggest_int("num_leaves", 15, 127),
            random_state=SEED, n_jobs=-1, verbose=-1)
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(-1)])
        return mean_squared_error(y_va, m.predict(X_va))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params,
                random_state=SEED, n_jobs=-1, verbose=-1)
    m = lgb.LGBMRegressor(**best)
    X_full = np.vstack([X_tr, X_va]); y_full = np.concatenate([y_tr, y_va])
    m.fit(X_full, y_full, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    pred = m.predict(X_te)
    return metrics_reg(y_te, pred), pred, m, study.best_params


def ridge_reg(X_tr, y_tr, X_te, y_te):
    m = Ridge(alpha=1.0, random_state=SEED).fit(X_tr, y_tr)
    p = m.predict(X_te)
    return metrics_reg(y_te, p), p


def rf_reg(X_tr, y_tr, X_te, y_te):
    m = RandomForestRegressor(n_estimators=500, max_depth=10,
                              min_samples_leaf=5, n_jobs=-1,
                              random_state=SEED).fit(X_tr, y_tr)
    p = m.predict(X_te)
    return metrics_reg(y_te, p), p, m


def stacking_reg(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials):
    """Stacking ensemble: XGB + LGBM + RF → Ridge meta-learner."""
    import xgboost as xgb; import lightgbm as lgb
    print("      [STACK] Training base models...")

    # Base models on train, OOF predictions on val
    _, p_xgb_va, _, _ = xgb_reg_optuna(X_tr, y_tr, X_va, y_va, X_va, y_va, max(n_trials//3, 20))
    _, p_lgbm_va, _, _ = lgbm_reg_optuna(X_tr, y_tr, X_va, y_va, X_va, y_va, max(n_trials//3, 20))
    _, p_rf_va, _ = rf_reg(X_tr, y_tr, X_va, y_va)

    meta_va = np.column_stack([p_xgb_va, p_lgbm_va, p_rf_va])
    meta_model = Ridge(alpha=1.0).fit(meta_va, y_va)

    # Retrain base on full train+val with fixed params
    X_full = np.vstack([X_tr, X_va]); y_full = np.concatenate([y_tr, y_va])

    xgb_m2 = xgb.XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.7,
                                random_state=SEED, n_jobs=-1, verbosity=0)
    xgb_m2.fit(X_full, y_full)

    lgbm_m2 = lgb.LGBMRegressor(n_estimators=600, max_depth=8, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.7,
                                  random_state=SEED, n_jobs=-1, verbose=-1)
    lgbm_m2.fit(X_full, y_full)

    rf_m2 = RandomForestRegressor(n_estimators=500, max_depth=10,
                                   min_samples_leaf=5, n_jobs=-1,
                                   random_state=SEED).fit(X_full, y_full)

    p_xgb_te = xgb_m2.predict(X_te)
    p_lgbm_te = lgbm_m2.predict(X_te)
    p_rf_te = rf_m2.predict(X_te)
    meta_te = np.column_stack([p_xgb_te, p_lgbm_te, p_rf_te])
    pred = meta_model.predict(meta_te)

    return metrics_reg(y_te, pred), pred


# ---------------------------------------------------------------------------
# Classification models: direction
# ---------------------------------------------------------------------------

def metrics_cls(y, p_prob, p_lab=None):
    if p_lab is None:
        p_lab = (p_prob >= 0.5).astype(int)
    return {"acc": float(accuracy_score(y, p_lab)),
            "auc": float(roc_auc_score(y, p_prob)),
            "f1":  float(f1_score(y, p_lab)),
            "brier": float(brier_score_loss(y, p_prob)),
            "logloss": float(log_loss(y, np.clip(p_prob, 1e-6, 1-1e-6)))}


def xgb_cls_optuna(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials):
    import optuna, xgboost as xgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        params = dict(
            n_estimators     = t.suggest_int("n_estimators", 200, 1500, step=100),
            max_depth        = t.suggest_int("max_depth", 3, 10),
            learning_rate    = t.suggest_float("learning_rate", 0.003, 0.15, log=True),
            subsample        = t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = t.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_weight = t.suggest_int("min_child_weight", 1, 15),
            reg_alpha        = t.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
            reg_lambda       = t.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            gamma            = t.suggest_float("gamma", 0, 1.0),
            random_state=SEED, n_jobs=-1, tree_method="hist", verbosity=0,
            early_stopping_rounds=40, eval_metric="logloss",
            objective="binary:logistic")
        m = xgb.XGBClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        return log_loss(y_va, np.clip(m.predict_proba(X_va)[:, 1], 1e-6, 1-1e-6))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params,
                random_state=SEED, n_jobs=-1, tree_method="hist",
                verbosity=0, eval_metric="logloss",
                early_stopping_rounds=40, objective="binary:logistic")
    m = xgb.XGBClassifier(**best)
    X_full = np.vstack([X_tr, X_va]); y_full = np.concatenate([y_tr, y_va])
    m.fit(X_full, y_full, eval_set=[(X_va, y_va)], verbose=False)
    prob = m.predict_proba(X_te)[:, 1]
    return metrics_cls(y_te, prob), prob, m, study.best_params


def lgbm_cls_optuna(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials):
    import optuna, lightgbm as lgb
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        params = dict(
            n_estimators     = t.suggest_int("n_estimators", 200, 1500, step=100),
            max_depth        = t.suggest_int("max_depth", 3, 12),
            learning_rate    = t.suggest_float("learning_rate", 0.003, 0.15, log=True),
            subsample        = t.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = t.suggest_float("colsample_bytree", 0.4, 1.0),
            min_child_samples= t.suggest_int("min_child_samples", 5, 50),
            reg_alpha        = t.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
            reg_lambda       = t.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            num_leaves       = t.suggest_int("num_leaves", 15, 127),
            random_state=SEED, n_jobs=-1, verbose=-1,
            objective="binary")
        m = lgb.LGBMClassifier(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(-1)])
        return log_loss(y_va, np.clip(m.predict_proba(X_va)[:, 1], 1e-6, 1-1e-6))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params,
                random_state=SEED, n_jobs=-1, verbose=-1, objective="binary")
    m = lgb.LGBMClassifier(**best)
    X_full = np.vstack([X_tr, X_va]); y_full = np.concatenate([y_tr, y_va])
    m.fit(X_full, y_full, eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    prob = m.predict_proba(X_te)[:, 1]
    return metrics_cls(y_te, prob), prob, m, study.best_params


def logit_cls(X_tr, y_tr, X_te, y_te):
    m = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED,
                           solver="liblinear").fit(X_tr, y_tr)
    prob = m.predict_proba(X_te)[:, 1]
    return metrics_cls(y_te, prob), prob


def rf_cls(X_tr, y_tr, X_te, y_te):
    m = RandomForestClassifier(n_estimators=500, max_depth=10,
                               min_samples_leaf=5, n_jobs=-1,
                               random_state=SEED, class_weight="balanced"
                               ).fit(X_tr, y_tr)
    prob = m.predict_proba(X_te)[:, 1]
    return metrics_cls(y_te, prob), prob, m


def stacking_cls(X_tr, y_tr, X_va, y_va, X_te, y_te, n_trials):
    """Stacking ensemble: XGB + LGBM + RF → Logistic meta-learner."""
    import xgboost as xgb; import lightgbm as lgb
    print("      [STACK] Training base classifiers...")

    _, p_xgb_va, _, _ = xgb_cls_optuna(X_tr, y_tr, X_va, y_va, X_va, y_va, max(n_trials//3, 20))
    _, p_lgbm_va, _, _ = lgbm_cls_optuna(X_tr, y_tr, X_va, y_va, X_va, y_va, max(n_trials//3, 20))
    _, p_rf_va, _ = rf_cls(X_tr, y_tr, X_va, y_va)

    meta_va = np.column_stack([p_xgb_va, p_lgbm_va, p_rf_va])
    meta_model = LogisticRegression(C=1.0, max_iter=2000, random_state=SEED).fit(
        meta_va, y_va)

    # Retrain base on train+val with fixed params
    X_full = np.vstack([X_tr, X_va]); y_full = np.concatenate([y_tr, y_va])

    xgb_m2 = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.7,
                                 random_state=SEED, n_jobs=-1, verbosity=0,
                                 objective="binary:logistic")
    xgb_m2.fit(X_full, y_full)

    lgbm_m2 = lgb.LGBMClassifier(n_estimators=600, max_depth=8, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.7,
                                   random_state=SEED, n_jobs=-1, verbose=-1,
                                   objective="binary")
    lgbm_m2.fit(X_full, y_full)

    rf_m2 = RandomForestClassifier(n_estimators=500, max_depth=10,
                                    min_samples_leaf=5, n_jobs=-1,
                                    random_state=SEED, class_weight="balanced").fit(X_full, y_full)

    p_xgb_te = xgb_m2.predict_proba(X_te)[:, 1]
    p_lgbm_te = lgbm_m2.predict_proba(X_te)[:, 1]
    p_rf_te = rf_m2.predict_proba(X_te)[:, 1]
    meta_te = np.column_stack([p_xgb_te, p_lgbm_te, p_rf_te])
    prob = meta_model.predict_proba(meta_te)[:, 1]

    return metrics_cls(y_te, prob), prob


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def diebold_mariano(y_true, p1, p2, h=1):
    e1 = (y_true - p1) ** 2
    e2 = (y_true - p2) ** 2
    d = e1 - e2
    n = len(d); mean_d = d.mean()
    gamma = [(d - mean_d) @ (d - mean_d) / n]
    for k in range(1, h):
        gamma.append(((d[k:] - mean_d) @ (d[:-k] - mean_d)) / n)
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return 0.0, 1.0
    dm = mean_d / math.sqrt(var_d)
    k_corr = math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= k_corr
    from scipy.stats import t as t_dist
    p = 2 * (1 - t_dist.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def dm_classification(y_true, prob1, prob2, h=1):
    e1 = (y_true - prob1) ** 2
    e2 = (y_true - prob2) ** 2
    d = e1 - e2
    n = len(d); mean_d = d.mean()
    var_d = ((d - mean_d) @ (d - mean_d)) / n / n
    if var_d <= 0:
        return 0.0, 1.0
    dm = mean_d / math.sqrt(var_d)
    from scipy.stats import t as t_dist
    p = 2 * (1 - t_dist.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)


def block_bootstrap_metric(y, p, metric_fn, n_boot=BOOTSTRAP_N,
                            block_size=10, alpha=0.05):
    """Block bootstrap for time-series data."""
    rng = np.random.RandomState(SEED)
    n = len(y)
    samples = np.empty(n_boot)
    n_blocks = max(1, n // block_size)
    for b in range(n_boot):
        block_starts = rng.randint(0, n - block_size + 1, n_blocks)
        idx = np.concatenate([np.arange(s, min(s + block_size, n))
                              for s in block_starts])[:n]
        if len(idx) < 10:
            samples[b] = np.nan; continue
        try:
            samples[b] = metric_fn(y[idx], p[idx])
        except Exception:
            samples[b] = np.nan
    samples = samples[~np.isnan(samples)]
    if len(samples) < 100:
        return float(metric_fn(y, p)), np.nan, np.nan
    return (float(metric_fn(y, p)),
            float(np.percentile(samples, 100 * alpha / 2)),
            float(np.percentile(samples, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_validate(feat, all_feat_cols, target_col, task="reg",
                          n_folds=WF_N_FOLDS):
    """Expanding-window walk-forward validation."""
    import lightgbm as lgb
    n = len(feat)
    min_train = int(n * 0.40)
    fold_size = (n - min_train) // n_folds
    results = []

    for fold in range(n_folds):
        tr_end = min_train + fold * fold_size
        te_end = min(tr_end + fold_size, n)
        if tr_end >= n or te_end <= tr_end:
            continue
        tr = feat.iloc[:tr_end]
        te = feat.iloc[tr_end:te_end]

        sc = StandardScaler().fit(tr[all_feat_cols].values)
        X_tr = sc.transform(tr[all_feat_cols].values)
        X_te = sc.transform(te[all_feat_cols].values)
        y_tr = tr[target_col].values.astype(np.float32)
        y_te = te[target_col].values.astype(np.float32)

        if task == "reg":
            m = lgb.LGBMRegressor(n_estimators=500, max_depth=8,
                                   learning_rate=0.05, random_state=SEED,
                                   n_jobs=-1, verbose=-1)
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
            r2 = r2_score(y_te, pred)
            corr = float(np.corrcoef(y_te, pred)[0, 1]) if y_te.std() > 0 else 0
            results.append({"fold": fold, "train_size": len(tr), "test_size": len(te),
                           "r2": r2, "corr": corr,
                           "test_start": str(te["Date"].iloc[0].date()),
                           "test_end": str(te["Date"].iloc[-1].date())})
        else:
            y_tr_c = y_tr.astype(int); y_te_c = y_te.astype(int)
            m = lgb.LGBMClassifier(n_estimators=500, max_depth=8,
                                    learning_rate=0.05, random_state=SEED,
                                    n_jobs=-1, verbose=-1, objective="binary")
            m.fit(X_tr, y_tr_c)
            prob = m.predict_proba(X_te)[:, 1]
            acc = accuracy_score(y_te_c, (prob >= 0.5).astype(int))
            auc = roc_auc_score(y_te_c, prob)
            results.append({"fold": fold, "train_size": len(tr), "test_size": len(te),
                           "acc": acc, "auc": auc,
                           "test_start": str(te["Date"].iloc[0].date()),
                           "test_end": str(te["Date"].iloc[-1].date())})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_market(market_code: str) -> Dict:
    print(f"\n{'='*60}\n  {market_code} -- V4: TRADING-DAY FILTERED\n{'='*60}")
    sub = "USD" if market_code == "US" else "UK"
    out_dir = BASE / sub / "results_v4"
    out_dir.mkdir(exist_ok=True)

    df, emb = load_market(market_code)
    df = add_targets_and_ar(df)

    n_total = len(df)
    i_tr = int(n_total * TRAIN_FRAC)
    pca = PCA(n_components=EMB_PCA_DIM, random_state=SEED).fit(emb[:i_tr])
    emb_pca = pca.transform(emb).astype(np.float32)
    print(f"  PCA explained: {pca.explained_variance_ratio_.sum():.3f}")

    feat, fin_cols, news_cols, ar_cols = build_features(df, emb_pca)
    feat = feat.dropna(subset=["target_vol", "target_dir"]).dropna()
    feat = feat.reset_index(drop=True)
    train, val, test = time_split(feat)
    print(f"  shapes: train={len(train)} val={len(val)} test={len(test)}")
    print(f"  test dates: {test['Date'].min().date()} -> {test['Date'].max().date()}")
    br = test["target_dir"].mean()
    print(f"  base rate (up days in test): {br:.3f}  (majority-class acc = {max(br, 1-br):.3f})")
    print(f"  mean |r| in test: {test['target_vol'].mean():.5f}")

    all_feat_cols = [c for c in feat.columns
                     if c not in {"target_vol", "target_log_vol", "target_dir",
                                  "next_ret", "Date"}]
    news_feat_cols = [c for c in all_feat_cols
                      if c in NEWS_SENT_COLS or c.startswith("emb_pc")]
    nonnews_feat_cols = [c for c in all_feat_cols if c not in news_feat_cols]

    print(f"  total features: {len(all_feat_cols)}")

    # ==================== VOLATILITY ====================
    print(f"\n  --- VOLATILITY PREDICTION (log-vol target) ---")

    sc = StandardScaler().fit(train[all_feat_cols].values)
    X_tr = sc.transform(train[all_feat_cols].values)
    X_va = sc.transform(val[all_feat_cols].values)
    X_te = sc.transform(test[all_feat_cols].values)

    # Train on log-vol, evaluate on both scales
    y_tr_lv = train["target_log_vol"].values.astype(np.float32)
    y_va_lv = val["target_log_vol"].values.astype(np.float32)
    y_te_lv = test["target_log_vol"].values.astype(np.float32)
    y_te_vol = test["target_vol"].values.astype(np.float32)

    vol_results = []
    vol_boot = []
    vol_preds = {}

    # XGBoost
    m_xgb, p_xgb_lv, xgb_v, bp_xgb = xgb_reg_optuna(
        X_tr, y_tr_lv, X_va, y_va_lv, X_te, y_te_lv, OPTUNA_TRIALS_FULL)
    p_xgb_vol = np.clip(np.exp(p_xgb_lv) - LOG_VOL_EPS, 0, None)
    m_xgb_orig = metrics_reg(y_te_vol, p_xgb_vol)
    print(f"    XGBoost   log-R2={m_xgb['r2']:+.4f}  orig-R2={m_xgb_orig['r2']:+.4f}  corr={m_xgb_orig['corr']:+.3f}")
    vol_results.append({"model": "XGBoost", "r2_log": m_xgb["r2"], **m_xgb_orig})
    vol_preds["XGBoost"] = p_xgb_vol

    # LightGBM
    m_lgbm, p_lgbm_lv, lgbm_v, bp_lgbm = lgbm_reg_optuna(
        X_tr, y_tr_lv, X_va, y_va_lv, X_te, y_te_lv, OPTUNA_TRIALS_FULL)
    p_lgbm_vol = np.clip(np.exp(p_lgbm_lv) - LOG_VOL_EPS, 0, None)
    m_lgbm_orig = metrics_reg(y_te_vol, p_lgbm_vol)
    print(f"    LightGBM  log-R2={m_lgbm['r2']:+.4f}  orig-R2={m_lgbm_orig['r2']:+.4f}  corr={m_lgbm_orig['corr']:+.3f}")
    vol_results.append({"model": "LightGBM", "r2_log": m_lgbm["r2"], **m_lgbm_orig})
    vol_preds["LightGBM"] = p_lgbm_vol

    # Random Forest
    m_rf_v, p_rf_lv, rf_v = rf_reg(X_tr, y_tr_lv, X_te, y_te_lv)
    p_rf_vol = np.clip(np.exp(p_rf_lv) - LOG_VOL_EPS, 0, None)
    m_rf_orig = metrics_reg(y_te_vol, p_rf_vol)
    print(f"    RF        log-R2={m_rf_v['r2']:+.4f}  orig-R2={m_rf_orig['r2']:+.4f}  corr={m_rf_orig['corr']:+.3f}")
    vol_results.append({"model": "RandomForest", "r2_log": m_rf_v["r2"], **m_rf_orig})
    vol_preds["RandomForest"] = p_rf_vol

    # Ridge (clip predictions to prevent exp() explosion)
    m_ridge_v, p_ridge_lv = ridge_reg(X_tr, y_tr_lv, X_te, y_te_lv)
    p_ridge_lv_clipped = np.clip(p_ridge_lv, np.percentile(y_tr_lv, 1), np.percentile(y_tr_lv, 99))
    p_ridge_vol = np.exp(p_ridge_lv_clipped) - LOG_VOL_EPS
    m_ridge_orig = metrics_reg(y_te_vol, np.clip(p_ridge_vol, 0, None))
    print(f"    Ridge     log-R2={m_ridge_v['r2']:+.4f}  orig-R2={m_ridge_orig['r2']:+.4f}")
    vol_results.append({"model": "Ridge", "r2_log": m_ridge_v["r2"], **m_ridge_orig})
    vol_preds["Ridge"] = np.clip(p_ridge_vol, 0, None)

    # Stacking
    m_stack, p_stack_lv = stacking_reg(X_tr, y_tr_lv, X_va, y_va_lv, X_te, y_te_lv,
                                        OPTUNA_TRIALS_FULL)
    p_stack_vol = np.clip(np.exp(p_stack_lv) - LOG_VOL_EPS, 0, None)
    m_stack_orig = metrics_reg(y_te_vol, p_stack_vol)
    print(f"    Stacking  log-R2={m_stack['r2']:+.4f}  orig-R2={m_stack_orig['r2']:+.4f}  corr={m_stack_orig['corr']:+.3f}")
    vol_results.append({"model": "Stacking", "r2_log": m_stack["r2"], **m_stack_orig})
    vol_preds["Stacking"] = p_stack_vol

    # Bootstrap CI for all models
    for nm, pp in vol_preds.items():
        r2_pt, lo, hi = block_bootstrap_metric(y_te_vol, pp, r2_score)
        vol_boot.append({"market": market_code, "model": nm,
                         "r2_point": r2_pt, "ci_low": lo, "ci_high": hi})

    # Feature importance (XGBoost)
    feat_imp_vol = []
    booster = xgb_v.get_booster()
    sc_imp = booster.get_score(importance_type="gain")
    for fname, gain in sc_imp.items():
        idx = int(fname[1:])
        if idx < len(all_feat_cols):
            feat_imp_vol.append({"market": market_code,
                                 "feature": all_feat_cols[idx],
                                 "gain": float(gain)})

    # DM tests (best vs each other)
    vol_dm = []
    best_vol_model = max(vol_results, key=lambda x: x["r2"])["model"]
    best_vol_pred = vol_preds[best_vol_model]
    for nm, pp in vol_preds.items():
        if nm == best_vol_model:
            continue
        dm, pv = diebold_mariano(y_te_vol, pp, best_vol_pred)
        vol_dm.append({"market": market_code, "baseline": nm,
                       "model": best_vol_model,
                       "DM_stat": dm, "p_value": pv,
                       "best_wins": pv < 0.05 and dm > 0})

    # ==================== DIRECTION ====================
    print(f"\n  --- DIRECTION PREDICTION ---")

    y_tr_d = train["target_dir"].values.astype(np.int32)
    y_va_d = val["target_dir"].values.astype(np.int32)
    y_te_d = test["target_dir"].values.astype(np.int32)

    dir_results = []
    dir_boot = []
    dir_preds = {}

    # XGBoost
    m_xgb_d, prob_xgb_d, xgb_d, bp_xgb_d = xgb_cls_optuna(
        X_tr, y_tr_d, X_va, y_va_d, X_te, y_te_d, OPTUNA_TRIALS_FULL)
    print(f"    XGBoost   ACC={m_xgb_d['acc']:.3f}  AUC={m_xgb_d['auc']:.3f}")
    dir_results.append({"model": "XGBoost", **m_xgb_d})
    dir_preds["XGBoost"] = prob_xgb_d

    # LightGBM
    m_lgbm_d, prob_lgbm_d, lgbm_d, bp_lgbm_d = lgbm_cls_optuna(
        X_tr, y_tr_d, X_va, y_va_d, X_te, y_te_d, OPTUNA_TRIALS_FULL)
    print(f"    LightGBM  ACC={m_lgbm_d['acc']:.3f}  AUC={m_lgbm_d['auc']:.3f}")
    dir_results.append({"model": "LightGBM", **m_lgbm_d})
    dir_preds["LightGBM"] = prob_lgbm_d

    # RF
    m_rf_d, prob_rf_d, _ = rf_cls(X_tr, y_tr_d, X_te, y_te_d)
    print(f"    RF        ACC={m_rf_d['acc']:.3f}  AUC={m_rf_d['auc']:.3f}")
    dir_results.append({"model": "RandomForest", **m_rf_d})
    dir_preds["RandomForest"] = prob_rf_d

    # Logistic
    m_lr, prob_lr = logit_cls(X_tr, y_tr_d, X_te, y_te_d)
    print(f"    Logit     ACC={m_lr['acc']:.3f}  AUC={m_lr['auc']:.3f}")
    dir_results.append({"model": "Logit", **m_lr})
    dir_preds["Logit"] = prob_lr

    # Stacking
    m_stack_d, prob_stack_d = stacking_cls(X_tr, y_tr_d, X_va, y_va_d,
                                           X_te, y_te_d, OPTUNA_TRIALS_FULL)
    print(f"    Stacking  ACC={m_stack_d['acc']:.3f}  AUC={m_stack_d['auc']:.3f}")
    dir_results.append({"model": "Stacking", **m_stack_d})
    dir_preds["Stacking"] = prob_stack_d

    # Bootstrap CI
    for nm, pr in dir_preds.items():
        a_pt, lo, hi = block_bootstrap_metric(y_te_d.astype(float), pr, roc_auc_score)
        dir_boot.append({"market": market_code, "model": nm,
                         "auc_point": a_pt, "ci_low": lo, "ci_high": hi})

    # Feature importance (XGBoost direction)
    feat_imp_dir = []
    booster_d = xgb_d.get_booster()
    sc_imp_d = booster_d.get_score(importance_type="gain")
    for fname, gain in sc_imp_d.items():
        idx = int(fname[1:])
        if idx < len(all_feat_cols):
            feat_imp_dir.append({"market": market_code,
                                 "feature": all_feat_cols[idx],
                                 "gain": float(gain)})

    # DM tests
    dir_dm = []
    best_dir_model = max(dir_results, key=lambda x: x["auc"])["model"]
    best_dir_prob = dir_preds[best_dir_model]
    for nm, pr in dir_preds.items():
        if nm == best_dir_model:
            continue
        dm, pv = dm_classification(y_te_d.astype(float), pr, best_dir_prob)
        dir_dm.append({"market": market_code, "baseline": nm,
                       "model": best_dir_model,
                       "DM_stat": dm, "p_value": pv,
                       "best_wins": pv < 0.05 and dm > 0})

    # Direction predictions
    dir_predictions = {
        "Date": test["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual_dir": test["target_dir"].tolist(),
        "next_ret": test["next_ret"].tolist(),
    }
    for nm, pr in dir_preds.items():
        dir_predictions[f"prob_{nm}"] = pr.tolist()

    # ==================== WALK-FORWARD ====================
    print(f"\n  --- WALK-FORWARD VALIDATION ---")
    wf_vol = walk_forward_validate(feat, all_feat_cols, "target_log_vol", "reg")
    wf_dir = walk_forward_validate(feat, all_feat_cols, "target_dir", "cls")
    print(f"    Vol WF avg R2: {wf_vol['r2'].mean():.4f}  (range: {wf_vol['r2'].min():.4f} ~ {wf_vol['r2'].max():.4f})")
    print(f"    Dir WF avg AUC: {wf_dir['auc'].mean():.4f}  (range: {wf_dir['auc'].min():.4f} ~ {wf_dir['auc'].max():.4f})")

    # ==================== SAVE ====================
    pd.DataFrame(vol_results).to_csv(out_dir / "volatility_model_comparison.csv", index=False)
    pd.DataFrame(vol_boot).to_csv(out_dir / "volatility_bootstrap_ci.csv", index=False)
    pd.DataFrame(vol_dm).to_csv(out_dir / "volatility_dm.csv", index=False)
    pd.DataFrame(dir_results).to_csv(out_dir / "direction_model_comparison.csv", index=False)
    pd.DataFrame(dir_boot).to_csv(out_dir / "direction_bootstrap_ci.csv", index=False)
    pd.DataFrame(dir_dm).to_csv(out_dir / "direction_dm.csv", index=False)
    pd.DataFrame(dir_predictions).to_csv(out_dir / "direction_predictions.csv", index=False)
    pd.DataFrame(feat_imp_vol).sort_values("gain", ascending=False
        ).to_csv(out_dir / "feature_importance_vol.csv", index=False)
    pd.DataFrame(feat_imp_dir).sort_values("gain", ascending=False
        ).to_csv(out_dir / "feature_importance_dir.csv", index=False)
    wf_vol.to_csv(out_dir / "walk_forward_vol.csv", index=False)
    wf_dir.to_csv(out_dir / "walk_forward_dir.csv", index=False)
    print(f"\n  [SAVED] {out_dir}")

    # Summary
    best_vol = max(vol_results, key=lambda x: x["r2_log"])
    best_dir = max(dir_results, key=lambda x: x["auc"])
    return {"market": market_code,
            "best_vol_model": best_vol["model"],
            "vol_r2_log": best_vol["r2_log"], "vol_r2_orig": best_vol["r2"],
            "vol_corr": best_vol["corr"],
            "best_dir_model": best_dir["model"],
            "dir_acc": best_dir["acc"], "dir_auc": best_dir["auc"],
            "wf_vol_mean_r2": float(wf_vol["r2"].mean()),
            "wf_dir_mean_auc": float(wf_dir["auc"].mean())}


if __name__ == "__main__":
    out = [run_market(m) for m in ["US", "UK"]]
    print("\n" + "="*60)
    print("  FINAL SUMMARY (v4, trading days only)")
    print("="*60)
    for s in out:
        print(f"  {s['market']}: Best Vol={s['best_vol_model']} logR2={s['vol_r2_log']:.4f} "
              f"origR2={s['vol_r2_orig']:.4f} corr={s['vol_corr']:.3f} | "
              f"Best Dir={s['best_dir_model']} ACC={s['dir_acc']:.3f} AUC={s['dir_auc']:.3f}")
        print(f"         WF Vol avg R2={s['wf_vol_mean_r2']:.4f} | "
              f"WF Dir avg AUC={s['wf_dir_mean_auc']:.4f}")
    print("="*60)
