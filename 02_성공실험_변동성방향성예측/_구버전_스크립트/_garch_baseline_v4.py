# -*- coding: utf-8 -*-
"""
GARCH / HAR-RV Baseline Comparison
====================================
Fits traditional econometric volatility models on the same train/test split
as _run_enhanced_models_v3.py and reports R² / RMSE / Corr.

Models
------
1. GARCH(1,1)  - standard conditional-variance model (Bollerslev 1986)
2. GJR-GARCH(1,1,1) - asymmetric leverage effect (Glosten et al. 1993)
3. HAR-RV      - Heterogeneous Autoregressive (Corsi 2009)
               vol_t = β₀ + β_d·|r_{t-1}| + β_w·avg|r|_{t-1:5} + β_m·avg|r|_{t-1:22}

Output
------
  USD/results_garch/garch_baselines_US.json
  UK/results_garch/garch_baselines_UK.json
"""

from __future__ import annotations
import json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")

BASE       = Path(__file__).resolve().parent

import sys as _sys
_sys.path.insert(0, str(BASE))
from _trading_days import filter_trading_days
SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
LOG_VOL_EPS = 1e-8
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_returns(market_code: str) -> pd.DataFrame:
    sub = "USD" if market_code == "US" else "UK"
    df = pd.read_csv(BASE / sub / f"{market_code}_research_enhanced.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    # === v4: 비거래일 제거 ===
    df = filter_trading_days(df, market_code)
    p = df["ETF"].astype(float)
    df["ret"] = np.log(p).diff()
    df["abs_ret"] = df["ret"].abs()
    # next-day target (realized vol proxy)
    df["target_vol"]     = df["abs_ret"].shift(-1)
    df["target_log_vol"] = np.log(df["abs_ret"].shift(-1) + LOG_VOL_EPS)
    # HAR components (all observable at close of day t)
    df["har_d"]  = df["abs_ret"].shift(1)                          # lag-1
    df["har_w"]  = df["abs_ret"].shift(1).rolling(5,  min_periods=1).mean()  # 5-day avg
    df["har_m"]  = df["abs_ret"].shift(1).rolling(22, min_periods=5).mean()  # 22-day avg
    df = df.dropna(subset=["ret", "target_vol", "har_d", "har_w", "har_m"])
    return df


def metrics(y_true, y_pred, tag: str) -> dict:
    r2   = float(r2_score(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 else 0.0
    print(f"  [{tag}]  R²={r2:.4f}  RMSE={rmse:.6f}  Corr={corr:.4f}")
    return {"r2": r2, "rmse": rmse, "corr": corr}


# ---------------------------------------------------------------------------
# HAR-RV (OLS / Ridge)
# ---------------------------------------------------------------------------

def run_har(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> dict:
    har_cols = ["har_d", "har_w", "har_m"]
    X_tr = df_tr[har_cols].values
    y_tr = df_tr["target_vol"].values
    X_te = df_te[har_cols].values
    y_te = df_te["target_vol"].values

    m = Ridge(alpha=1e-4, fit_intercept=True, random_state=SEED).fit(X_tr, y_tr)
    pred = np.maximum(m.predict(X_te), 0)   # vol ≥ 0
    return metrics(y_te, pred, "HAR-RV"), pred, y_te


def run_har_log(df_tr: pd.DataFrame, df_te: pd.DataFrame) -> dict:
    """HAR on log-vol scale (same as ML models)."""
    har_cols = ["har_d", "har_w", "har_m"]
    X_tr = df_tr[har_cols].values
    y_tr = df_tr["target_log_vol"].values
    X_te = df_te[har_cols].values
    y_te = df_te["target_log_vol"].values

    m = Ridge(alpha=1e-4, fit_intercept=True, random_state=SEED).fit(X_tr, y_tr)
    pred = m.predict(X_te)
    return metrics(y_te, pred, "HAR-RV (log-vol)"), pred, y_te


# ---------------------------------------------------------------------------
# GARCH(1,1)
# ---------------------------------------------------------------------------

def run_garch(df_tr: pd.DataFrame, df_te: pd.DataFrame, gjr: bool = False) -> dict:
    try:
        from arch import arch_model
    except ImportError:
        print("  [GARCH] 'arch' package not installed - skipping.")
        return {"r2": None, "rmse": None, "corr": None, "error": "arch not installed"}

    returns_tr = df_tr["ret"].values * 100   # scale to % for numerical stability
    returns_te = df_te["ret"].values * 100

    tag_label = "GJR-GARCH(1,1,1)" if gjr else "GARCH(1,1)"

    try:
        if gjr:
            am = arch_model(returns_tr, vol="Garch", p=1, o=1, q=1, dist="Normal")
        else:
            am = arch_model(returns_tr, vol="Garch", p=1, q=1, dist="Normal")

        res = am.fit(disp="off", show_warning=False)

        # 1-step-ahead forecast for each test point using rolling estimation
        # (fit on train, then forecast test period one step at a time)
        forecasts = []
        full_returns = np.concatenate([returns_tr, returns_te])
        n_tr = len(returns_tr)

        for i in range(len(returns_te)):
            window = full_returns[:n_tr + i]
            try:
                if gjr:
                    am_i = arch_model(window, vol="Garch", p=1, o=1, q=1, dist="Normal")
                else:
                    am_i = arch_model(window, vol="Garch", p=1, q=1, dist="Normal")
                res_i = am_i.fit(disp="off", show_warning=False,
                                 starting_values=res.params.values)
                fc = res_i.forecast(horizon=1, reindex=False)
                # conditional std → vol proxy (scaled back to decimal)
                forecasts.append(float(np.sqrt(fc.variance.values[-1, 0])) / 100.0)
            except Exception:
                # fallback: use last known conditional std
                forecasts.append(forecasts[-1] if forecasts else np.abs(window[-1]) / 100.0)

        pred  = np.array(forecasts)
        y_te  = df_te["target_vol"].values
        return metrics(y_te, pred, tag_label), pred, y_te

    except Exception as e:
        print(f"  [{tag_label}] fit failed: {e}")
        return {"r2": None, "rmse": None, "corr": None, "error": str(e)}


# ---------------------------------------------------------------------------
# GARCH (fast variant - use static params, forecast whole test window)
# ---------------------------------------------------------------------------

def run_garch_fast(df_tr: pd.DataFrame, df_te: pd.DataFrame, gjr: bool = False):
    """GARCH baseline: fit on train only, then propagate params through test period.
    Always returns 3-tuple: (metrics_dict, pred_or_None, y_te_or_None)
    """
    _err = ({"r2": None, "rmse": None, "corr": None}, None, None)

    try:
        from arch import arch_model
    except ImportError:
        print("  [GARCH] 'arch' package not installed - skipping.")
        return _err

    tag_label = "GJR-GARCH(1,1,1)" if gjr else "GARCH(1,1)"

    # Clean training returns
    ret_tr = df_tr["ret"].fillna(0).values * 100
    ret_te = df_te["ret"].fillna(0).values * 100

    if len(ret_tr) < 50:
        print(f"  [{tag_label}] not enough data ({len(ret_tr)} rows)")
        return _err

    try:
        # Fit on training returns only
        if gjr:
            am = arch_model(ret_tr, vol="Garch", p=1, o=1, q=1, dist="Normal")
        else:
            am = arch_model(ret_tr, vol="Garch", p=1, q=1, dist="Normal")

        res = am.fit(disp="off", show_warning=False)
        params = res.params  # omega, alpha[1], (gamma[1] for GJR,) beta[1]

        # Extract fitted parameters
        omega = float(params.get("omega", params.iloc[1]))
        alpha = float(params.get("alpha[1]", params.iloc[2]))
        beta  = float(params.get("beta[1]",  params.iloc[-1]))
        gamma = float(params.get("gamma[1]", 0.0)) if gjr else 0.0

        # Seed sigma2 with last conditional variance from train fit
        sigma2 = float(res.conditional_volatility[-1] ** 2)

        # Propagate through test period
        forecasts = []
        for r in ret_te:
            sigma2_next = omega + alpha * r**2 + gamma * r**2 * (r < 0) + beta * sigma2
            sigma2_next = max(sigma2_next, 1e-10)
            forecasts.append(np.sqrt(sigma2_next) / 100.0)  # back to decimal
            sigma2 = sigma2_next

        pred  = np.array(forecasts)
        y_te  = df_te["target_vol"].fillna(df_te["target_vol"].mean()).values

        min_len = min(len(pred), len(y_te))
        pred, y_te = pred[:min_len], y_te[:min_len]

        return metrics(y_te, pred, tag_label), pred, y_te

    except Exception as e:
        print(f"  [{tag_label}] failed: {e}")
        return {"r2": None, "rmse": None, "corr": None, "error": str(e)}, None, None

    except Exception as e:
        print(f"  [{tag_label}] failed: {e}")
        return {"r2": None, "rmse": None, "corr": None, "error": str(e)}, None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_market(market_code: str) -> dict:
    print(f"\n{'='*55}\n  {market_code} - GARCH / HAR-RV Baselines\n{'='*55}")

    df = load_returns(market_code)
    n  = len(df)
    n_tr = int(n * TRAIN_FRAC)
    n_va = int(n * VAL_FRAC)
    # same split as main model: 70% train, 15% val, 15% test
    df_tr = df.iloc[:n_tr]
    df_va = df.iloc[n_tr: n_tr + n_va]
    df_te = df.iloc[n_tr + n_va:]

    # For baselines, train on train+val (same as ML models after Optuna)
    df_train = pd.concat([df_tr, df_va], ignore_index=True)

    print(f"  Train rows: {len(df_train)}  Test rows: {len(df_te)}")
    print(f"  Test period: {df_te['Date'].iloc[0].date()} → {df_te['Date'].iloc[-1].date()}")

    results = {}

    # 1. HAR-RV (level)
    har_metrics, har_pred, har_yte = run_har(df_train, df_te)
    results["HAR-RV"] = har_metrics

    # 2. HAR-RV (log-vol scale - comparable to ML R²)
    har_log_metrics, _, _ = run_har_log(df_train, df_te)
    results["HAR-RV_log"] = har_log_metrics

    # 3. GARCH(1,1) fast (fit on train, condition through test)
    g_metrics, _, _ = run_garch_fast(df_train, df_te, gjr=False)
    results["GARCH(1,1)"] = g_metrics

    # 4. GJR-GARCH fast
    gjr_metrics, _, _ = run_garch_fast(df_train, df_te, gjr=True)
    results["GJR-GARCH(1,1,1)"] = gjr_metrics

    # Save
    sub = "USD" if market_code == "US" else "UK"
    out_dir = BASE / sub / "results_garch_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"garch_baselines_{market_code}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → {out_path}")

    return results


if __name__ == "__main__":
    all_results = {}
    for mkt in ["US", "UK"]:
        all_results[mkt] = run_market(mkt)

    print("\n" + "="*55)
    print("  SUMMARY - Volatility Prediction R² Comparison")
    print("="*55)
    print(f"{'Model':<25}  {'US R²':>8}  {'UK R²':>8}")
    print("-"*45)
    for model in ["HAR-RV", "HAR-RV_log", "GARCH(1,1)", "GJR-GARCH(1,1,1)"]:
        us_r2 = all_results.get("US", {}).get(model, {}).get("r2")
        uk_r2 = all_results.get("UK", {}).get(model, {}).get("r2")
        us_str = f"{us_r2:.4f}" if us_r2 is not None else "N/A"
        uk_str = f"{uk_r2:.4f}" if uk_r2 is not None else "N/A"
        print(f"  {model:<23}  {us_str:>8}  {uk_str:>8}")
    print("-"*45)
    print("  [ML models from main experiment for comparison]")
    print(f"  {'RandomForest (Full)':<23}  {'0.8300':>8}  {'0.9320':>8}")
    print(f"  {'XGBoost (Full)':<23}  {'0.7400':>8}  {'0.8800':>8}")
