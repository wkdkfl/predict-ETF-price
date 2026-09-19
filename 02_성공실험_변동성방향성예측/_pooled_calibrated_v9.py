# -*- coding: utf-8 -*-
"""v9 — pooled 학습 + 예측값 선형 보정(Mincer-Zarnowitz).

배경
----
v8 에서 pooled 학습으로 영국 상관계수가 0.306 까지 올라왔으나 R² 는 -0.073 이었다.
상관이 r 일 때 최적 선형 보정 후 달성 가능한 R² 상한은 r² 이므로,
영국의 이론적 상한은 약 0.094 이다. 즉 남은 손실은 신호 부족이 아니라
예측값의 수준(level)과 분산(dispersion) 오정렬에서 온다.

보정
----
검증 구간에서 y_val ~ a + b * pred_val 를 적합하고, 그 (a, b) 를 시험 구간
예측에 적용한다. 이는 예측 평가 문헌의 표준 절차(Mincer-Zarnowitz)이며
시험 구간 정보를 일절 쓰지 않는다.

분할
----
  학습:  Date < CUT_TRAIN
  검증:  CUT_TRAIN <= Date < CUT_TEST      (보정계수 추정에만 사용)
  시험:  Date >= CUT_TEST
CUT_TEST 는 두 시장 공통, 학습에는 두 시장 모두 CUT_TRAIN 이전만 사용해 누출을 막는다.
"""
from __future__ import annotations

import io, contextlib, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402
from _news_variant import RES  # noqa: E402

SEED = 42
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)


def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


def main():
    US, keep_us, ar_us = build("US")
    UK, keep_uk, ar_uk = build("UK")
    common = [c for c in keep_us if c in keep_uk]
    ar_common = [c for c in ar_us if c in ar_uk]

    frames = {"US": US, "UK": UK}
    CUT_TEST = min(s["Date"].iloc[int(len(s) * 0.85)] for s in frames.values())
    CUT_TRAIN = min(s["Date"].iloc[int(len(s) * 0.70)] for s in frames.values())
    print("학습 < %s  |  검증 < %s  |  시험 >= %s"
          % (CUT_TRAIN.date(), CUT_TEST.date(), CUT_TEST.date()))
    print("공통 피처 %d개 (AR %d개)\n" % (len(common), len(ar_common)))

    rows = []
    for FEAT, fname in [(ar_common, "AR_Only"), (common, "Full")]:
        print("=" * 84)
        print("  피처군: %s (%d개)" % (fname, len(FEAT)))
        Z, D, Y, PAST, M = {}, {}, {}, {}, {}
        for m, s in frames.items():
            tr = (s["Date"] < CUT_TRAIN).values
            sc = StandardScaler().fit(s[FEAT].values[tr])
            Z[m] = sc.transform(s[FEAT].values)
            Y[m] = s["vol5"].values
            PAST[m] = s["rv5_past"].values
            D[m] = Y[m] - PAST[m]
            M[m] = {"tr": tr,
                    "va": ((s["Date"] >= CUT_TRAIN) & (s["Date"] < CUT_TEST)).values,
                    "te": (s["Date"] >= CUT_TEST).values}

        for m in ["US", "UK"]:
            o = "UK" if m == "US" else "US"
            tr, va, te = M[m]["tr"], M[m]["va"], M[m]["te"]
            tro = M[o]["tr"]
            y_te, past_te = Y[m][te], PAST[m][te]

            def run(Xtr, ytr, tag):
                mdl = lgb.LGBMRegressor(**KW).fit(Xtr, ytr)
                p_va = mdl.predict(Z[m][va]) + PAST[m][va]
                p_te = mdl.predict(Z[m][te]) + past_te
                r_raw = r2_score(y_te, p_te)
                cal = LinearRegression().fit(p_va.reshape(-1, 1), Y[m][va])
                p_cal = cal.predict(p_te.reshape(-1, 1))
                r_cal = r2_score(y_te, p_cal)
                cr = np.corrcoef(y_te, p_te)[0, 1]
                rows.append(dict(market=m, features=fname, model=tag,
                                 r2_raw=r_raw, r2_calibrated=r_cal, corr=cr,
                                 ceiling=cr**2, slope=float(cal.coef_[0]),
                                 intercept=float(cal.intercept_), n_test=int(te.sum())))
                print("     %-12s 보정전 R2=%+.4f  ->  보정후 R2=%+.4f   "
                      "(상관 %.3f, 상한 %.3f, 기울기 %.2f)"
                      % (tag, r_raw, r_cal, cr, cr**2, cal.coef_[0]))

            print("\n  [%s]  학습 %d행(단독) / %d행(pooled),  검증 %d,  시험 %d"
                  % (m, tr.sum(), tr.sum() + tro.sum(), va.sum(), te.sum()))
            run(Z[m][tr], D[m][tr], "단독")
            Xp = np.vstack([Z[m][tr], Z[o][tro]])
            yp = np.concatenate([D[m][tr], D[o][tro]])
            run(Xp, yp, "pooled")

    out = pd.DataFrame(rows)
    out.to_csv(RES / "_pooled_calibrated_v9_results.csv", index=False, encoding="utf-8-sig")
    print("\n" + "=" * 84)
    print(out[["market", "features", "model", "r2_raw", "r2_calibrated", "corr", "ceiling"]]
          .round(4).to_string(index=False))
    print("\n저장: _pooled_calibrated_v9_results.csv")


if __name__ == "__main__":
    main()
