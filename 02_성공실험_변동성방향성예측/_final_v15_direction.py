# -*- coding: utf-8 -*-
"""v15 방향성 — v10 프로토콜에서 **분류기만** 교체 (LightGBM -> L2 로지스틱 C=0.01).

사전 고정 근거: plan/20260921_direction_v15_plan.md (시험 구간을 보기 전에 작성).
표본 분할·pooled 학습·표준화·Platt 보정·블록 부트스트랩·DM 검정은 _final_v10_direction.py 와 동일하다.
v10 기준선을 같은 실행 안에서 함께 적합해 ΔAUC 의 쌍체(paired) 블록 부트스트랩 CI 를 낸다.

  python _final_v15_direction.py                 # 기존 뉴스
  NEWS_DATA=new python _final_v15_direction.py   # 신규 일별 뉴스
"""
from __future__ import annotations

import io, contextlib, math, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402
from _news_variant import RES  # noqa: E402

SEED, N_BOOT = 42, 5000
# 타깃 시계: dir5(주 결과) | dir20(탐색적 보조, plan 5절). 블록 길이·DM h 를 시계에 맞춘다.
TARGET = os.environ.get("DIR_TARGET", "dir5")
HORIZON = int(TARGET.replace("dir", ""))
BLOCK = 2 * HORIZON
LGB_KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
              colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
              n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
LOGIT_C = 0.01
GROUPS = ["AR_Only", "Financial_Only", "Financial_Clean", "News_Pure", "AR+News", "Full"]
DM_PAIRS = [("AR_Only", "AR+News"), ("AR_Only", "Full"),
            ("Financial_Only", "Full"), ("News_Pure", "Full"),
            ("Financial_Clean", "Full")]


def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


def _boot_idx(n, rng):
    nb = max(1, n // BLOCK)
    st = rng.randint(0, max(1, n - BLOCK + 1), nb)
    return np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]


def block_boot_auc(y, p):
    rng = np.random.RandomState(SEED)
    out = []
    for _ in range(N_BOOT):
        idx = _boot_idx(len(y), rng)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def block_boot_dauc(y, p_new, p_old):
    """같은 부트스트랩 표본에서 두 모형의 AUC 차 -> 쌍체 CI."""
    rng = np.random.RandomState(SEED)
    out = []
    for _ in range(N_BOOT):
        idx = _boot_idx(len(y), rng)
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_old[idx]))
    out = np.array(out)
    return float(out.mean()), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def dm_cls(y, p1, p2, h=None):
    h = HORIZON if h is None else h
    d = (y - p1) ** 2 - (y - p2) ** 2
    n = len(d); md = d.mean()
    g = [((d - md) @ (d - md)) / n]
    for k in range(1, h):
        g.append(((d[k:] - md) @ (d[:-k] - md)) / n)
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 0:
        return 0.0, 1.0
    s = md / math.sqrt(var)
    s *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(s), float(2 * (1 - t_dist.cdf(abs(s), df=n - 1)))


def main():
    US, keep_us, ar_us = build("US")
    UK, keep_uk, ar_uk = build("UK")
    frames = {"US": US.dropna(subset=[TARGET]).reset_index(drop=True),
              "UK": UK.dropna(subset=[TARGET]).reset_index(drop=True)}
    common = [c for c in keep_us if c in keep_uk]
    from _run_enhanced_models_v4 import NEWS_SENT_COLS
    emb_c = [c for c in common if c.startswith("emb_pc")]
    sent_c = [c for c in common if c in NEWS_SENT_COLS]
    ar_c = [c for c in common if c in ar_us and c in ar_uk]
    fin_c = [c for c in common if c not in emb_c + sent_c + ar_c]
    G = {"AR_Only": ar_c, "Financial_Only": fin_c, "News_Pure": sent_c + emb_c,
         "AR+News": ar_c + sent_c + emb_c, "Full": common,
         "Financial_Clean": [c for c in fin_c if not c.startswith("sent_")]}

    CUT_TEST = min(s["Date"].iloc[int(len(s) * 0.85)] for s in frames.values())
    CUT_TRAIN = min(s["Date"].iloc[int(len(s) * 0.70)] for s in frames.values())
    print("타깃=%s | 뉴스=%s | 학습 < %s | 검증 < %s | 시험 >= %s\n"
          % (TARGET, os.environ.get("NEWS_DATA", "old"), CUT_TRAIN.date(), CUT_TEST.date(), CUT_TEST.date()))

    rows, preds = [], {}
    for gname in GROUPS:
        FEAT = G[gname]
        Z, Y, MK = {}, {}, {}
        for m, s in frames.items():
            tr = (s["Date"] < CUT_TRAIN).values
            sc = StandardScaler().fit(s[FEAT].values[tr])
            Z[m] = sc.transform(s[FEAT].values)
            Y[m] = s[TARGET].values.astype(int)
            MK[m] = {"tr": tr,
                     "va": ((s["Date"] >= CUT_TRAIN) & (s["Date"] < CUT_TEST)).values,
                     "te": (s["Date"] >= CUT_TEST).values}
        for m in ["US", "UK"]:
            o = "UK" if m == "US" else "US"
            tr, va, te = MK[m]["tr"], MK[m]["va"], MK[m]["te"]
            Xp = np.vstack([Z[m][tr], Z[o][MK[o]["tr"]]])
            yp = np.concatenate([Y[m][tr], Y[o][MK[o]["tr"]]])
            y_te = Y[m][te]
            for tag, mdl in (("v15", LogisticRegression(C=LOGIT_C, max_iter=2000, solver="lbfgs")),
                             ("v10", lgb.LGBMClassifier(**LGB_KW))):
                mdl.fit(Xp, yp)
                p_va = mdl.predict_proba(Z[m][va])[:, 1]
                p_te = mdl.predict_proba(Z[m][te])[:, 1]
                plat = LogisticRegression().fit(p_va.reshape(-1, 1), Y[m][va])
                p_cal = plat.predict_proba(p_te.reshape(-1, 1))[:, 1]
                preds[(tag, m, gname)] = (y_te, p_te, p_cal)
            _, p15, c15 = preds[("v15", m, gname)]
            _, p10, c10 = preds[("v10", m, gname)]
            lo, hi = block_boot_auc(y_te, p15)
            dm_, dlo, dhi = block_boot_dauc(y_te, p15, p10)
            base = max(y_te.mean(), 1 - y_te.mean())
            rows.append(dict(market=m, group=gname, n_feat=len(FEAT),
                             auc_v15=roc_auc_score(y_te, p15), ci_low=lo, ci_high=hi,
                             auc_v10=roc_auc_score(y_te, p10),
                             dauc=dm_, dauc_low=dlo, dauc_high=dhi,
                             acc=accuracy_score(y_te, (c15 > 0.5).astype(int)), baseline=base,
                             f1=f1_score(y_te, (c15 > 0.5).astype(int)),
                             brier=brier_score_loss(y_te, c15), n_test=int(te.sum())))

    res = pd.DataFrame(rows)
    print("=" * 104)
    print(("  %d일 누적 방향성 — v15(L2 로지스틱) vs v10(LightGBM), pooled·동일 분할") % HORIZON)
    print("  %-16s %-5s %8s %20s %9s %22s" % ("피처군", "시장", "AUC", "95% CI", "v10 AUC", "dAUC [95% CI]"))
    for _, r in res.iterrows():
        star = " *" if r["ci_low"] > 0.5 else "  "
        sig = " <" if r["dauc_low"] > 0 else ""
        print("  %-16s %-5s %8.3f  [%.3f, %.3f]%s %8.3f   %+.3f [%+.3f, %+.3f]%s"
              % (r["group"], r["market"], r["auc_v15"], r["ci_low"], r["ci_high"], star,
                 r["auc_v10"], r["dauc"], r["dauc_low"], r["dauc_high"], sig))
    print("  (* = AUC 의 95% CI 하한 > 0.5,  < = dAUC 의 95% CI 가 0 을 배제)")

    print("\n" + "=" * 104)
    print("  Diebold-Mariano 검정 (Brier 손실, v15 예측)")
    dm_rows = []
    for m in ["US", "UK"]:
        for b, c in DM_PAIRS:
            y, _, p1 = preds[("v15", m, b)]
            _, _, p2 = preds[("v15", m, c)]
            s, pv = dm_cls(y.astype(float), p1, p2)
            dm_rows.append(dict(market=m, baseline=b, model=c, dm=s, p=pv))
            print("  %-4s %-16s vs %-10s DM=%+7.3f  p=%.4f  %s"
                  % (m, b, c, s, pv, ("%s 우위" % c) if (pv < 0.05 and s > 0)
                     else ("%s 우위" % b) if pv < 0.05 else "유의차 없음"))

    sfx = "" if TARGET == "dir5" else "_" + TARGET
    res.to_csv(RES / ("_final_v15_direction%s.csv" % sfx), index=False, encoding="utf-8-sig")
    pd.DataFrame(dm_rows).to_csv(RES / ("_final_v15_dm_dir%s.csv" % sfx), index=False, encoding="utf-8-sig")
    print("\n저장: %s" % RES)


if __name__ == "__main__":
    main()
