# -*- coding: utf-8 -*-
"""그림 8(방향성 진단) / 그림 10(변동성-감성 동행) 재생성 — v10 확정 프로토콜 기준."""
from __future__ import annotations

import io, contextlib, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402
from _news_variant import RES, DATA_SFX  # noqa: E402

OUT = Path(os.environ.get("FIG_OUT", BASE))
SEED = 42
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
US_C, UK_C = "#1F4E79", "#C0504D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.grid": True,
                     "grid.alpha": 0.35, "axes.axisbelow": True, "figure.facecolor": "white"})


def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


def real_news_dates(sub):
    h = pd.read_csv(BASE / sub / f"news_per_headline{DATA_SFX}.csv", usecols=["date"])
    return set(pd.to_datetime(h["date"], errors="coerce").dropna().dt.normalize())


# ---------------- 예측값 산출 (v10 direction 과 동일) ----------------
US, keep_us, ar_us = build("US")
UK, keep_uk, ar_uk = build("UK")
frames = {"US": US, "UK": UK}
common = [c for c in keep_us if c in keep_uk]
CUT_TEST = min(s["Date"].iloc[int(len(s) * 0.85)] for s in frames.values())
CUT_TRAIN = min(s["Date"].iloc[int(len(s) * 0.70)] for s in frames.values())

Z, Y, MK = {}, {}, {}
for m, s in frames.items():
    tr = (s["Date"] < CUT_TRAIN).values
    sc = StandardScaler().fit(s[common].values[tr])
    Z[m] = sc.transform(s[common].values)
    Y[m] = s["dir5"].values.astype(int)
    MK[m] = {"tr": tr,
             "va": ((s["Date"] >= CUT_TRAIN) & (s["Date"] < CUT_TEST)).values,
             "te": (s["Date"] >= CUT_TEST).values}

PR = {}
for m in ["US", "UK"]:
    o = "UK" if m == "US" else "US"
    tr, va, te = MK[m]["tr"], MK[m]["va"], MK[m]["te"]
    Xp = np.vstack([Z[m][tr], Z[o][MK[o]["tr"]]])
    yp = np.concatenate([Y[m][tr], Y[o][MK[o]["tr"]]])
    mdl = lgb.LGBMClassifier(**KW).fit(Xp, yp)
    p_va = mdl.predict_proba(Z[m][va])[:, 1]
    p_te = mdl.predict_proba(Z[m][te])[:, 1]
    plat = LogisticRegression().fit(p_va.reshape(-1, 1), Y[m][va])
    PR[m] = dict(prob=plat.predict_proba(p_te.reshape(-1, 1))[:, 1],
                 raw=p_te, y=Y[m][te],
                 date=frames[m]["Date"].values[te])
    _pd_dir = RES / ("USD" if m == "US" else "UK") / "results_v6"
    _pd_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Date": PR[m]["date"], "actual_dir5": PR[m]["y"],
                  "prob": PR[m]["prob"]}).to_csv(
        _pd_dir / "direction5_predictions_v10.csv",
        index=False)
    print("  %s 예측 저장: n=%d  AUC=%.3f" % (m, len(PR[m]["y"]),
                                          roc_auc_score(PR[m]["y"], PR[m]["raw"])))

# ================= 그림 8 =================
fig, ax = plt.subplots(2, 2, figsize=(14.5, 10.4))
for j, (m, c, lb) in enumerate([("US", US_C, "US (SPY)"), ("UK", UK_C, "UK (ISF / FTSE100)")]):
    y, p = PR[m]["y"], PR[m]["prob"]
    # (상단) 확률 보정 곡선 — 십분위
    a = ax[0, j]
    q = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(
        pm=("p", "mean"), ym=("y", "mean"), n=("y", "size"))
    a.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    a.plot(g.pm, g.ym, "o-", color=c, lw=2, ms=7, label=lb)
    a.axhline(y.mean(), color="0.5", ls=":", lw=1.3,
              label="Base rate = %.3f" % y.mean())
    a.set_xlabel("Predicted probability of 5-day rise")
    a.set_ylabel("Observed frequency")
    a.set_title("%s: probability calibration (deciles)" % m, fontsize=13.5, pad=9)
    # 예측 확률이 좁은 구간에 몰리므로 데이터 범위에 맞춰 확대
    lo = min(g.pm.min(), g.ym.min()); hi = max(g.pm.max(), g.ym.max())
    pad = max(0.06, 0.25 * (hi - lo))
    a.set_xlim(lo - pad, hi + pad); a.set_ylim(lo - pad, hi + pad)
    a.legend(fontsize=9.5, loc="upper left")

    # (하단) 누적 적중률 (lift)
    a = ax[1, j]
    o = np.argsort(-p)
    cum = np.cumsum(y[o]) / np.arange(1, len(y) + 1)
    frac = np.arange(1, len(y) + 1) / len(y)
    a.plot(frac, cum, color=c, lw=2, label="Cumulative hit-rate of top-K%")
    a.axhline(y.mean(), color="k", ls="--", lw=1.2, label="Base rate = %.3f" % y.mean())
    a.set_xlabel("Top fraction of test days (ranked by predicted probability)")
    a.set_ylabel("Realised rise rate")
    a.set_title("%s: cumulative lift" % m, fontsize=13.5, pad=9)
    a.set_xlim(0, 1); a.legend(fontsize=9.5, loc="upper right")

fig.suptitle("Direction classification diagnostics: 5-day cumulative sign (pooled model, test period)",
             fontsize=14.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.955])
fig.savefig(OUT / "fig8_v10.png", dpi=145)
plt.close(fig)
print("fig8_v10.png")

# ================= 그림 10 =================
fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.9))
for j, (m, sub, c, lb) in enumerate([("US", "USD", US_C, "US (SPY)"),
                                     ("UK", "UK", UK_C, "UK (ISF / FTSE100)")]):
    s = frames[m].copy()
    s["Date"] = pd.to_datetime(s["Date"])
    s = s[s["Date"] >= "2017-01-01"].reset_index(drop=True)
    rv = np.exp(s["rv5_past"].values)
    roll = pd.Series(rv).rolling(20, min_periods=5).mean()
    sent = s["sent_score_mean"] if "sent_score_mean" in s.columns else None

    a = ax[j]
    a.plot(s["Date"], roll, color=c, lw=1.5, label="Realised vol (20d mean of RV5)")
    a.set_ylabel("Realised volatility", color=c)
    a.tick_params(axis="y", labelcolor=c)
    a.set_title("%s: realised volatility vs news sentiment (2017-2025)" % m,
                fontsize=13.5, pad=9)

    if sent is not None:
        a2 = a.twinx(); a2.grid(False)
        a2.plot(s["Date"], sent.rolling(20, min_periods=5).mean(),
                color="0.35", lw=1.2, alpha=0.85, label="FinBERT sentiment (20d mean)")
        a2.set_ylabel("FinBERT sentiment", color="0.35")
        a2.tick_params(axis="y", labelcolor="0.35")

    # 실제 뉴스가 존재하는 날 표시
    nd = real_news_dates(sub)
    has = s["Date"].dt.normalize().isin(nd).values
    ymin, ymax = a.get_ylim()
    a.plot(s["Date"][has], np.full(has.sum(), ymin + 0.02 * (ymax - ymin)),
           "|", color="0.15", ms=6, alpha=0.55,
           label="Days with actual news (%d of %d)" % (has.sum(), len(s)))
    a.set_ylim(ymin, ymax)
    a.legend(fontsize=9, loc="upper left")

fig.suptitle("News coverage and the volatility-sentiment relationship "
             "(ticks mark days with genuinely collected news; other days are forward-filled)",
             fontsize=13.5, y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig10_v10.png", dpi=145)
plt.close(fig)
print("fig10_v10.png")
