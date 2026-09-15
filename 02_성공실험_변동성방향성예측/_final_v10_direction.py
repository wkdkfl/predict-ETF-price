# -*- coding: utf-8 -*-
"""v10 방향성 — 변동성과 완전히 동일한 pooled 프로토콜.

AUC 는 단조변환에 불변이므로 확률 보정의 영향을 받지 않는다.
정확도는 영향을 받으므로 검증 구간에서 Platt scaling 을 적합해 적용한다.
"""
from __future__ import annotations

import io, contextlib, math, sys, warnings
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

SEED, N_BOOT, BLOCK = 42, 5000, 10
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
GROUPS = ["AR_Only", "Financial_Only", "News_Pure", "AR+News", "Full"]


def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


def block_boot_auc(y, p, n_boot=N_BOOT, block=BLOCK):
    rng = np.random.RandomState(SEED)
    n = len(y); nb = max(1, n // block); out = []
    for _ in range(n_boot):
        st = rng.randint(0, max(1, n - block + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def dm_cls(y, p1, p2, h=5):
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
    frames = {"US": US, "UK": UK}
    common = [c for c in keep_us if c in keep_uk]
    from _run_enhanced_models_v4 import NEWS_SENT_COLS
    emb_c = [c for c in common if c.startswith("emb_pc")]
    sent_c = [c for c in common if c in NEWS_SENT_COLS]
    ar_c = [c for c in common if c in ar_us and c in ar_uk]
    fin_c = [c for c in common if c not in emb_c + sent_c + ar_c]
    G = {"AR_Only": ar_c, "Financial_Only": fin_c, "News_Pure": sent_c + emb_c,
         "AR+News": ar_c + sent_c + emb_c, "Full": common}

    CUT_TEST = min(s["Date"].iloc[int(len(s) * 0.85)] for s in frames.values())
    CUT_TRAIN = min(s["Date"].iloc[int(len(s) * 0.70)] for s in frames.values())
    print("학습 < %s | 검증 < %s | 시험 >= %s\n"
          % (CUT_TRAIN.date(), CUT_TEST.date(), CUT_TEST.date()))

    rows, preds = [], {}
    for gname in GROUPS:
        FEAT = G[gname]
        Z, Y, MK = {}, {}, {}
        for m, s in frames.items():
            tr = (s["Date"] < CUT_TRAIN).values
            sc = StandardScaler().fit(s[FEAT].values[tr])
            Z[m] = sc.transform(s[FEAT].values)
            Y[m] = s["dir5"].values.astype(int)
            MK[m] = {"tr": tr,
                     "va": ((s["Date"] >= CUT_TRAIN) & (s["Date"] < CUT_TEST)).values,
                     "te": (s["Date"] >= CUT_TEST).values}
        for m in ["US", "UK"]:
            o = "UK" if m == "US" else "US"
            tr, va, te = MK[m]["tr"], MK[m]["va"], MK[m]["te"]
            Xp = np.vstack([Z[m][tr], Z[o][MK[o]["tr"]]])
            yp = np.concatenate([Y[m][tr], Y[o][MK[o]["tr"]]])
            mdl = lgb.LGBMClassifier(**KW).fit(Xp, yp)
            p_va = mdl.predict_proba(Z[m][va])[:, 1]
            p_te = mdl.predict_proba(Z[m][te])[:, 1]
            plat = LogisticRegression().fit(p_va.reshape(-1, 1), Y[m][va])
            p_cal = plat.predict_proba(p_te.reshape(-1, 1))[:, 1]
            y_te = Y[m][te]
            auc = roc_auc_score(y_te, p_te)          # 보정 불변
            lo, hi = block_boot_auc(y_te, p_te)
            base = max(y_te.mean(), 1 - y_te.mean())
            acc = accuracy_score(y_te, (p_cal > 0.5).astype(int))
            rows.append(dict(market=m, group=gname, n_feat=len(FEAT), auc=auc,
                             ci_low=lo, ci_high=hi, acc=acc, baseline=base,
                             f1=f1_score(y_te, (p_cal > 0.5).astype(int)),
                             brier=brier_score_loss(y_te, p_cal), n_test=int(te.sum())))
            preds[(m, gname)] = (y_te, p_cal)

    res = pd.DataFrame(rows)
    print("=" * 96)
    print("  5일 누적 방향성 — pooled, 두 시장 동일 절차")
    print("  %-16s %-6s %8s %22s %9s %9s" % ("피처군", "시장", "AUC", "95% CI", "정확도", "다수클래스"))
    for _, r in res.iterrows():
        star = " *" if r["ci_low"] > 0.5 else ""
        print("  %-16s %-6s %8.3f   [%.3f, %.3f]%s %8.3f %9.3f"
              % (r["group"], r["market"], r["auc"], r["ci_low"], r["ci_high"],
                 star, r["acc"], r["baseline"]))
    print("  (* = 95% 신뢰구간 하한이 0.5 초과)")

    print("\n" + "=" * 96)
    print("  Diebold-Mariano 검정 (Brier 손실 기반)")
    dm_rows = []
    for m in ["US", "UK"]:
        for b, c in [("AR_Only", "AR+News"), ("AR_Only", "Full"),
                     ("Financial_Only", "Full"), ("News_Pure", "Full")]:
            y, p1 = preds[(m, b)]
            _, p2 = preds[(m, c)]
            s, pv = dm_cls(y.astype(float), p1, p2)
            dm_rows.append(dict(market=m, baseline=b, model=c, dm=s, p=pv))
            print("  %-4s %-16s vs %-10s DM=%+7.3f  p=%.4f  %s"
                  % (m, b, c, s, pv,
                     ("%s 우위" % c) if (pv < 0.05 and s > 0)
                     else ("%s 우위" % b) if pv < 0.05 else "유의차 없음"))

    res.to_csv(BASE / "_final_v10_direction.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dm_rows).to_csv(BASE / "_final_v10_dm_dir.csv", index=False, encoding="utf-8-sig")
    print("\n저장 완료")


if __name__ == "__main__":
    main()
