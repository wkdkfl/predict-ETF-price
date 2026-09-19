# -*- coding: utf-8 -*-
"""v10 최종 — pooled 학습 + 검증구간 보정, 두 시장에 완전히 동일한 절차.

확정 설정
---------
  * 거래일 필터 (비거래일 전방보간 행 제거)
  * 결측 과다 열 제거로 표본 회복
  * 타깃: log RV5(t+1..t+5) - log RV5(t-4..t)  (레짐 이동 불변) -> 되돌려 평가
  * 학습: 두 시장 공통 컷오프 이전 데이터를 합쳐서(pooled) 학습
  * 보정: 검증 구간에서 Mincer-Zarnowitz 선형 보정계수 추정 후 시험 구간에 적용
  * 피처: 두 시장 공통 열만 사용, 시장별 표준화

시장별로 설정을 달리하지 않으며, 모든 피처군에 같은 절차를 적용한다.
블록 부트스트랩 CI 와 Diebold-Mariano 검정으로 유의성을 평가한다.
"""
from __future__ import annotations

import io, contextlib, math, os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402
from _news_variant import RES  # noqa: E402

SEED = 42
N_BOOT = 5000
BLOCK = 10
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
GROUPS = ["AR_Only", "Financial_Only", "News_Pure", "AR+News", "Full"]
DM_PAIRS = [("AR_Only", "AR+News"), ("AR_Only", "Full"),
            ("Financial_Only", "Full"), ("News_Pure", "Full")]
# Financial_Only 에는 뉴스 파생 열(sent_x_vix, sent_x_vol)이 섞여 있다(NEWS_SENT_COLS 에 미등록).
# 기존 그룹·값은 그대로 두고, 이를 제외한 Financial_Clean 을 추가 행으로만 계산한다.
if os.environ.get("EXTRA_CLEAN_FIN") == "1":
    GROUPS = GROUPS + ["Financial_Clean"]
    DM_PAIRS = DM_PAIRS + [("Financial_Clean", "Full")]


def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


def block_boot(y, p, fn, n_boot=N_BOOT, block=BLOCK):
    rng = np.random.RandomState(SEED)
    n = len(y); nb = max(1, n // block); out = []
    for _ in range(n_boot):
        st = rng.randint(0, max(1, n - block + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        try:
            out.append(fn(y[idx], p[idx]))
        except Exception:
            pass
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def dm_test(y, p1, p2, h=5):
    """p1 기준 vs p2. 양수면 p2 우위. Newey-West HAC + HLN 수정."""
    d = (y - p1) ** 2 - (y - p2) ** 2
    n = len(d); md = d.mean()
    g = [((d - md) @ (d - md)) / n]
    for k in range(1, h):
        g.append(((d[k:] - md) @ (d[:-k] - md)) / n)
    var = (g[0] + 2 * sum(g[1:])) / n
    if var <= 0:
        return 0.0, 1.0
    stat = md / math.sqrt(var)
    stat *= math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat), float(2 * (1 - t_dist.cdf(abs(stat), df=n - 1)))


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
    if "Financial_Clean" in GROUPS:
        G["Financial_Clean"] = [c for c in fin_c if not c.startswith("sent_")]

    CUT_TEST = min(s["Date"].iloc[int(len(s) * 0.85)] for s in frames.values())
    CUT_TRAIN = min(s["Date"].iloc[int(len(s) * 0.70)] for s in frames.values())
    print("학습 < %s | 검증 < %s | 시험 >= %s" % (CUT_TRAIN.date(), CUT_TEST.date(), CUT_TEST.date()))
    print("피처: AR %d / 금융 %d / 감성 %d / 임베딩 %d  (공통 %d)\n"
          % (len(ar_c), len(fin_c), len(sent_c), len(emb_c), len(common)))

    rows, preds = [], {}
    for gname in GROUPS:
        FEAT = G[gname]
        Z, D, Y, PAST, MK = {}, {}, {}, {}, {}
        for m, s in frames.items():
            tr = (s["Date"] < CUT_TRAIN).values
            sc = StandardScaler().fit(s[FEAT].values[tr])
            Z[m] = sc.transform(s[FEAT].values)
            Y[m] = s["vol5"].values
            PAST[m] = s["rv5_past"].values
            D[m] = Y[m] - PAST[m]
            MK[m] = {"tr": tr,
                     "va": ((s["Date"] >= CUT_TRAIN) & (s["Date"] < CUT_TEST)).values,
                     "te": (s["Date"] >= CUT_TEST).values}
        for m in ["US", "UK"]:
            o = "UK" if m == "US" else "US"
            tr, va, te = MK[m]["tr"], MK[m]["va"], MK[m]["te"]
            Xp = np.vstack([Z[m][tr], Z[o][MK[o]["tr"]]])
            yp = np.concatenate([D[m][tr], D[o][MK[o]["tr"]]])
            mdl = lgb.LGBMRegressor(**KW).fit(Xp, yp)
            p_va = mdl.predict(Z[m][va]) + PAST[m][va]
            p_te = mdl.predict(Z[m][te]) + PAST[m][te]
            cal = LinearRegression().fit(p_va.reshape(-1, 1), Y[m][va])
            p_cal = cal.predict(p_te.reshape(-1, 1))
            y_te = Y[m][te]
            r2 = r2_score(y_te, p_cal)
            lo, hi = block_boot(y_te, p_cal, r2_score)
            rows.append(dict(market=m, group=gname, n_feat=len(FEAT), r2=r2,
                             ci_low=lo, ci_high=hi, corr=np.corrcoef(y_te, p_cal)[0, 1],
                             n_test=int(te.sum())))
            preds[(m, gname)] = (y_te, p_cal)

    res = pd.DataFrame(rows)
    print("=" * 92)
    print("  변동성(5일 RV) 예측 — pooled + 보정, 두 시장 동일 절차")
    print("  %-16s %-8s %9s %22s %9s" % ("피처군", "시장", "R²", "95% CI", "상관"))
    for _, r in res.iterrows():
        star = " *" if r.ci_low > 0 else ""
        print("  %-16s %-8s %+9.4f   [%+.4f, %+.4f]%s %8.3f"
              % (r["group"], r["market"], r["r2"], r["ci_low"], r["ci_high"], star, r["corr"]))
    print("  (* = 95% 신뢰구간이 0을 포함하지 않음)")

    print("\n" + "=" * 92)
    print("  Diebold-Mariano 검정 (기준 vs AR+News / Full) — 뉴스 증분 기여")
    dm_rows = []
    for m in ["US", "UK"]:
        for base, cmp_ in DM_PAIRS:
            y, p1 = preds[(m, base)]
            _, p2 = preds[(m, cmp_)]
            s, pv = dm_test(y, p1, p2)
            dm_rows.append(dict(market=m, baseline=base, model=cmp_, dm=s, p=pv))
            print("  %-4s %-16s vs %-10s DM=%+7.3f  p=%.4f  %s"
                  % (m, base, cmp_, s, pv,
                     ("%s 우위" % cmp_) if (pv < 0.05 and s > 0)
                     else ("%s 우위" % base) if pv < 0.05 else "유의차 없음"))

    res.to_csv(RES / "_final_v10_volatility.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dm_rows).to_csv(RES / "_final_v10_dm.csv", index=False, encoding="utf-8-sig")
    print("\n저장: _final_v10_volatility.csv, _final_v10_dm.csv")


if __name__ == "__main__":
    main()
