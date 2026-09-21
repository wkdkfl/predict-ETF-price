# -*- coding: utf-8 -*-
"""그림 10 재작성 — 기존/신규 뉴스 커버리지를 두 줄로 대비 (§5.1.1 재검정 반영).

기존 `_figs_8_10_v10.py` 의 그림 10 은 '실제 뉴스가 수집된 날' 눈금을 한 줄만 그렸다.
§5.1.1 에서 커버리지를 전 거래일로 채운 자료로 재검정했으므로, 두 자료의 커버리지를
나란히 보여 주어야 그림 하나로 재검정의 전제가 드러난다.

상단 곡선(실현변동성·FinBERT 감성)은 본 실험 자료 기준으로 기존 그림과 동일하다.
바뀐 것은 하단 눈금이 2줄이 된 점뿐이다.

출력: fig10_v15.png (기존 fig10_v10.png 는 그대로 둔다)
실행: PYTHONIOENCODING=utf-8 python _fig10_coverage_v15.py
"""
from __future__ import annotations

import contextlib, io, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

OUT = Path(os.environ.get("FIG_OUT", BASE))
US_C, UK_C = "#1F4E79", "#C0504D"
OLD_C, NEW_C = "0.15", "#2E7D32"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12.5, "axes.grid": True,
                     "grid.alpha": 0.35, "axes.axisbelow": True, "figure.facecolor": "white"})


def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


def news_dates(sub, sfx):
    """해당 자료에서 실제로 기사가 존재하는 날짜 집합."""
    h = pd.read_csv(BASE / sub / ("news_per_headline%s.csv" % sfx), usecols=["date"])
    return set(pd.to_datetime(h["date"], errors="coerce").dropna().dt.normalize())


def main():
    os.environ.setdefault("NEWS_DATA", "old")   # 상단 곡선은 본 실험(기존) 자료 기준
    os.environ.setdefault("PCA_SOLVER", "full")
    frames = {mc: build(mc)[0] for mc in ("US", "UK")}

    fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.4))
    for j, (m, sub, c, lb) in enumerate([("US", "USD", US_C, "US (SPY)"),
                                         ("UK", "UK", UK_C, "UK (ISF / FTSE100)")]):
        s = frames[m].copy()
        s["Date"] = pd.to_datetime(s["Date"])
        s = s[s["Date"] >= "2017-01-01"].reset_index(drop=True)
        rv = np.exp(s["rv5_past"].values)
        roll = pd.Series(rv).rolling(20, min_periods=5).mean()

        a = ax[j]
        a.plot(s["Date"], roll, color=c, lw=1.5, label="Realised vol (20d mean of RV5)")
        a.set_ylabel("Realised volatility", color=c)
        a.tick_params(axis="y", labelcolor=c)
        a.set_title("%s: realised volatility vs news sentiment (2017-2025)" % m,
                    fontsize=13.5, pad=9)

        if "sent_score_mean" in s.columns:
            a2 = a.twinx(); a2.grid(False)
            a2.plot(s["Date"], s["sent_score_mean"].rolling(20, min_periods=5).mean(),
                    color="0.35", lw=1.2, alpha=0.85)
            a2.set_ylabel("FinBERT sentiment", color="0.35")
            a2.tick_params(axis="y", labelcolor="0.35")

        # ---- 하단: 커버리지 눈금 2줄 (기존 / 신규) -------------------------
        d = s["Date"].dt.normalize()
        ymin, ymax = a.get_ylim()
        span = ymax - ymin
        a.set_ylim(ymin - 0.17 * span, ymax)          # 눈금 자리 확보
        ymin2, _ = a.get_ylim(); span2 = ymax - ymin2

        for k, (sfx, col, name) in enumerate([("", OLD_C, "original (forward-filled)"),
                                              ("_newsv2", NEW_C, "augmented (§5.1.1)")]):
            has = d.isin(news_dates(sub, sfx)).values
            yk = ymin2 + (0.085 - 0.048 * k) * span2
            a.plot(s["Date"][has], np.full(has.sum(), yk), "|", color=col, ms=7,
                   alpha=0.75 if k else 0.6,
                   label="News days, %s: %s of %s" % (name, f"{has.sum():,}", f"{len(s):,}"))
            print("  %s %-28s %4d / %d" % (m, name, has.sum(), len(s)))

        # 눈금 영역은 축을 늘려 만든 여백이므로 음(-)의 변동성 눈금 값은 지운다
        a.set_yticks([t for t in a.get_yticks() if t >= 0 and t <= ymax])
        a.set_ylim(ymin2, ymax)
        a.axhline(ymin2 + 0.125 * span2, color="0.8", lw=0.8, zorder=1)
        a.legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.suptitle("News coverage before and after augmentation "
                 "(ticks mark days with genuinely collected news)",
                 fontsize=13.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = OUT / "fig10_v15.png"
    fig.savefig(p, dpi=145)
    plt.close(fig)
    print("저장: %s" % p)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
