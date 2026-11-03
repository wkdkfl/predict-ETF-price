# -*- coding: utf-8 -*-
"""그림 7·9 재생성 — v10 확정 결과 CSV 기준 (논문 표 11·12·15·17·18 과 동일 수치).
원본 이미지와 같은 픽셀 크기(145 dpi)로 저장하여 문서 내 표시 크기를 유지한다."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

BASE = Path(sys.argv[1])
OUT = Path(sys.argv[2])
US_C, UK_C = "#1F4E79", "#C0504D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.grid": True,
                     "grid.alpha": 0.35, "axes.axisbelow": True, "figure.facecolor": "white"})
DPI = 145
GROUPS = ["Financial_Only", "News_Pure", "AR_Only", "AR+News", "Full"]
GLAB = ["Financial\nOnly", "News\nPure", "AR\nOnly", "AR +\nNews", "Full"]

vol = pd.read_csv(BASE / "_final_v10_volatility.csv", encoding="utf-8-sig")
dirr = pd.read_csv(BASE / "_final_v10_direction.csv", encoding="utf-8-sig")
pool = pd.read_csv(BASE / "_pooled_calibrated_v9_results.csv", encoding="utf-8-sig")
folds = pd.read_csv(BASE / "_robustness_v13_folds.csv", encoding="utf-8-sig")

# 표 11 (예측 시계 비교, 시장별 단독 학습) — 논문 <표 11> 수치
H = ["1-day", "5-day", "10-day"]
T11 = {"US_ML": [0.116, 0.364, 0.282], "US_HAR": [0.026, 0.187, 0.181],
       "UK_ML": [-0.011, -0.001, -0.103], "UK_HAR": [-0.039, -0.101, -0.222]}

# ================= 그림 7 =================
fig, ax = plt.subplots(1, 3, figsize=(2827 / DPI, 841 / DPI))
a = ax[0]
x = np.arange(3); w = 0.2
a.bar(x - 1.5 * w, T11["US_ML"], w, color=US_C, label="US (SPY) ML")
a.bar(x - 0.5 * w, T11["US_HAR"], w, color=US_C, alpha=0.4, hatch="//", edgecolor=US_C, label="US (SPY) HAR-RV")
a.bar(x + 0.5 * w, T11["UK_ML"], w, color=UK_C, label="UK (ISF / FTSE100) ML")
a.bar(x + 1.5 * w, T11["UK_HAR"], w, color=UK_C, alpha=0.4, hatch="//", edgecolor=UK_C, label="UK (ISF / FTSE100) HAR-RV")
a.axhline(0, color="k", lw=0.8)
a.set_xticks(x); a.set_xticklabels(H); a.set_xlabel("Forecast horizon")
a.set_ylabel("Test $R^2$ (log scale)"); a.set_ylim(-0.3, 0.45)
a.set_title("(a) Horizon effect (market-specific)", fontsize=13.5, pad=9)
a.legend(fontsize=8.5, ncol=2, loc="lower left")

for k, (df, col, ttl, ylab) in enumerate([
        (vol, "r2", r"(b) 5-day volatility  $\log RV_5$", "Test $R^2$ (log scale)"),
        (dirr, "auc", r"(c) 5-day direction  sign($\sum r$)", "Test AUC")]):
    a = ax[k + 1]
    x = np.arange(len(GROUPS)); w = 0.38
    for j, (m, c, lb) in enumerate([("US", US_C, "US (SPY)"), ("UK", UK_C, "UK (ISF / FTSE100)")]):
        d = df[df.market == m].set_index("group").loc[GROUPS]
        v = d[col].values
        err = np.vstack([v - d.ci_low.values, d.ci_high.values - v])
        a.bar(x + (j - 0.5) * w, v, w, color=c, label=lb, yerr=err, capsize=4,
              error_kw=dict(lw=1.2))
    if col == "r2":
        a.axhline(0, color="k", lw=0.8)
    else:
        a.axhline(0.5, color="0.3", ls="--", lw=1.2, label="Random (AUC = 0.50)")
        a.set_ylim(0.30, 0.75)
    a.set_xticks(x); a.set_xticklabels(GLAB)
    a.set_ylabel(ylab); a.set_title(ttl, fontsize=13.5, pad=9)
    a.legend(fontsize=9, loc="lower left" if col == "r2" else "upper left")
fig.suptitle("Final results: pooled cross-market training with validation-set calibration", fontsize=14.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig7_final.png", dpi=DPI)
plt.close(fig)

# ================= 그림 9 =================
fig, ax = plt.subplots(1, 2, figsize=(2247 / DPI, 841 / DPI))
a = ax[0]
cases = [("US", "AR_Only"), ("UK", "AR_Only"), ("US", "Full"), ("UK", "Full")]
x = np.arange(4); w = 0.36
for i, (m, f) in enumerate(cases):
    d = pool[(pool.market == m) & (pool.features == f)].set_index("model")
    s, p = d.loc["단독", "r2_calibrated"], d.loc["pooled", "r2_calibrated"]
    a.bar(i - w / 2, s, w, color="0.62", label="Market-specific training" if i == 0 else None)
    a.bar(i + w / 2, p, w, color=US_C if m == "US" else UK_C, label="Pooled training" if i == 0 else None)
    for xx, vv in [(i - w / 2, s), (i + w / 2, p)]:
        a.text(xx, vv + (0.012 if vv >= 0 else -0.012), "%+.3f" % vv, ha="center",
               va="bottom" if vv >= 0 else "top", fontsize=10)
a.axhline(0, color="k", lw=0.8)
a.set_xticks(x); a.set_xticklabels(["%s\n%s" % (f, m) for m, f in cases], fontsize=10.5)
a.set_ylabel("Test $R^2$ (log scale, calibrated)"); a.set_ylim(-0.25, 0.45)
a.set_title("(a) Pooled cross-market training", fontsize=13.5, pad=9)
a.legend(fontsize=10, loc="lower left")

a = ax[1]
for m, c, lb in [("US", US_C, "US (SPY)"), ("UK", UK_C, "UK (ISF / FTSE100)")]:
    d = folds[folds.market == m]
    xv, yv = d.V_within.values, d.AR_Only_r2.values
    r, pv = pearsonr(xv, yv)
    a.scatter(xv, yv, s=60, color=c, label="%s   r=%+.3f (p=%.3f)" % (lb, r, pv), zorder=3)
    b1, b0 = np.polyfit(xv, yv, 1)
    xx = np.linspace(xv.min(), xv.max(), 50)
    a.plot(xx, b0 + b1 * xx, "--", color=c, lw=1.6, alpha=0.8)
    print("fig9 %s r=%.3f p=%.3f" % (m, r, pv))
r, pv = pearsonr(folds.V_within, folds.AR_Only_r2)
print("fig9 pooled 16 folds r=%.3f p=%.4f" % (r, pv))
a.axhline(0, color="k", lw=0.8)
a.set_xlabel(r"Dispersion of $\log RV_5$ within the fold   ($\sigma$)")
a.set_ylabel("Fold test $R^2$ (AR_Only)")
a.set_title("(b) Predictability vs volatility dispersion (8 folds each)", fontsize=13.5, pad=9)
a.legend(fontsize=10, loc="upper left")
fig.suptitle("Cross-market pooling and the dispersion dependence of volatility predictability", fontsize=14.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig9_final.png", dpi=DPI)
plt.close(fig)
print("saved fig7_final.png fig9_final.png")
