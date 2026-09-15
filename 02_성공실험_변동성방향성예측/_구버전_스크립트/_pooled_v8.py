# -*- coding: utf-8 -*-
"""v8 — 두 시장 데이터를 합쳐 학습(pooled / panel)하면 영국 예측이 개선되는가?

동기
----
영국 표본은 2,886행뿐이고 2023년 이후 저변동성 국면이 이어져 학습 신호가 부족하다.
변동성 동학(군집성, 레버리지 효과, 평균회귀)은 시장 간 공통 구조이므로,
미국 데이터를 함께 학습하면 영국의 표본 부족을 보완할 수 있는지 검정한다.

누출 방지 설계
--------------
  * 공통 컷오프 날짜를 정하고, 학습에는 두 시장 모두 컷오프 '이전' 관측치만 사용
    (미국의 미래 정보가 영국 시험 구간 예측에 들어가지 않도록)
  * 피처는 두 시장에 공통으로 존재하는 열만 사용
  * 표준화는 시장별로 학습 구간에서만 적합 (수준 차이를 제거하고 동학만 공유)
  * 타깃은 차분 형태 log RV5(t+1..t+5) - log RV5(t-4..t) 로 레짐 이동에 불변

비교군
------
  (1) 시장별 단독 학습 (기준)
  (2) pooled 학습
  (3) pooled 학습 + 시장 더미
  (4) pooled 사전학습 후 해당 시장 데이터로 미세조정(2단계)
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import lightgbm as lgb  # noqa: E402

SEED = 42
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)


def build(mc):
    """_regime_fixes_v7.prep 와 동일한 프레임을 조용히 재현."""
    import io, contextlib
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        s, keep, ar_c = prep(mc)
    return s, keep, ar_c


def evaluate(name, pred, actual):
    return r2_score(actual, pred), np.corrcoef(actual, pred)[0, 1]


def main():
    US, keep_us, ar_us = build("US")
    UK, keep_uk, ar_uk = build("UK")

    common = [c for c in keep_us if c in keep_uk]
    ar_common = [c for c in ar_us if c in ar_uk]
    print("공통 피처 %d개 (그중 AR %d개)" % (len(common), len(ar_common)))
    print("  미국 단독 %d / 영국 단독 %d" % (len(keep_us), len(keep_uk)))

    # 공통 컷오프: 두 시장의 시험 시작일 중 이른 쪽
    def test_start(s):
        return s["Date"].iloc[int(len(s) * 0.85)]
    CUT = min(test_start(US), test_start(UK))
    print("공통 컷오프: %s  (이 날짜 이전만 학습에 사용)" % CUT.date())

    frames = {"US": US, "UK": UK}
    results = []

    for FEAT, fname in [(ar_common, "AR_Only"), (common, "Full")]:
        print("\n" + "=" * 82)
        print("  피처군: %s (%d개)" % (fname, len(FEAT)))

        # 시장별 표준화 (학습 구간에서만 적합)
        Z, Y, D, PAST = {}, {}, {}, {}
        for m, s in frames.items():
            tr_mask = (s["Date"] < CUT).values
            sc = StandardScaler().fit(s[FEAT].values[tr_mask])
            Z[m] = sc.transform(s[FEAT].values)
            Y[m] = s["vol5"].values
            PAST[m] = s["rv5_past"].values
            D[m] = Y[m] - PAST[m]

        for m in ["US", "UK"]:
            s = frames[m]
            tr = (s["Date"] < CUT).values
            te = ~tr
            other = "UK" if m == "US" else "US"
            so = frames[other]
            tro = (so["Date"] < CUT).values

            y_te = Y[m][te]
            past_te = PAST[m][te]

            # (1) 단독 학습
            m1 = lgb.LGBMRegressor(**KW).fit(Z[m][tr], D[m][tr])
            p1 = m1.predict(Z[m][te]) + past_te

            # (2) pooled
            Xp = np.vstack([Z[m][tr], Z[other][tro]])
            yp = np.concatenate([D[m][tr], D[other][tro]])
            m2 = lgb.LGBMRegressor(**KW).fit(Xp, yp)
            p2 = m2.predict(Z[m][te]) + past_te

            # (3) pooled + 시장 더미
            dum_tr = np.concatenate([np.zeros(tr.sum()), np.ones(tro.sum())])
            m3 = lgb.LGBMRegressor(**KW).fit(np.column_stack([Xp, dum_tr]), yp)
            p3 = m3.predict(np.column_stack([Z[m][te], np.zeros(te.sum())])) + past_te

            # (4) pooled 사전학습 -> 해당 시장 미세조정
            m4 = lgb.LGBMRegressor(**KW)
            m4.fit(Z[m][tr], D[m][tr], init_model=m2.booster_)
            p4 = m4.predict(Z[m][te]) + past_te

            row = {"market": m, "features": fname, "n_train_own": int(tr.sum()),
                   "n_train_pooled": int(tr.sum() + tro.sum()), "n_test": int(te.sum())}
            for tag, p in [("단독", p1), ("pooled", p2), ("pooled+더미", p3), ("pooled+미세조정", p4)]:
                r2, cr = evaluate(tag, p, y_te)
                row[tag + "_R2"] = r2
                row[tag + "_corr"] = cr
            results.append(row)

            print("\n  [%s]  학습 단독 %d행 -> pooled %d행,  시험 %d행"
                  % (m, row["n_train_own"], row["n_train_pooled"], row["n_test"]))
            for tag in ["단독", "pooled", "pooled+더미", "pooled+미세조정"]:
                print("     %-16s R2 = %+.4f   상관 = %.3f"
                      % (tag, row[tag + "_R2"], row[tag + "_corr"]))

    out = pd.DataFrame(results)
    out.to_csv(BASE / "_pooled_v8_results.csv", index=False, encoding="utf-8-sig")
    print("\n저장: _pooled_v8_results.csv")


if __name__ == "__main__":
    main()
