# -*- coding: utf-8 -*-
"""v17 중첩 워크포워드 예측의 F1·혼동행렬.

임계값 두 가지(라벨을 보지 않고 정한다):
  t050   : 확률 0.5
  prior  : fold 마다 '예측 상승 비율 = 그 fold 학습 구간의 실제 상승 비율' 이 되도록 정한 분위 임계값
비교 기준:
  always_up : 항상 상승으로 예측
  random    : 학습 구간 상승 비율로 무작위 예측 (기대 macro-F1 ≈ 0.5). 5,000회 모의.
macro-F1 의 95% CI 는 블록 길이 10 블록 부트스트랩.

  PCA_SOLVER=full RESULT_DIR=방향성개선_v17 python _direction_v17_f1.py
"""
from __future__ import annotations

import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from _news_variant import RES  # noqa: E402
from _direction_v16_walkforward import build, train_rows, boot_idx, SEED, N_BOOT, N_FOLDS, START_Q  # noqa: E402


# 두 클래스 평균 F1
def macro_f1(y, yh):
    return f1_score(y, yh, average="macro", zero_division=0)


# 임계값별 혼동행렬·F1·macro-F1 신뢰구간 계산 및 저장
def main():
    frames = {m: build(m)[0] for m in ["US", "UK"]}
    us_dates = frames["US"]["Date"].values
    n = len(us_dates); a0 = int(n * START_Q); fs = (n - a0) // N_FOLDS
    edges = [pd.Timestamp(us_dates[a0 + k * fs]) for k in range(N_FOLDS)]
    pr = pd.read_csv(RES / "v17_nested_predictions.csv", parse_dates=["Date"])

    rows = []
    for m, s in frames.items():
        t = pr[pr.market == m].sort_values("Date").reset_index(drop=True)
        y = t["y"].values.astype(int)
        up_rate = {k + 1: s["dir5"].values[train_rows(s, edges[k])].mean() for k in range(N_FOLDS)}
        q = t["fold"].map(up_rate).values
        preds = {}
        for rule in ["best", "top3"]:
            p = t[rule].values
            preds[(rule, "t050")] = (p > 0.5).astype(int)
            thr = t.groupby("fold")[rule].transform(lambda v: v.quantile(1 - up_rate[v.name])).values
            preds[(rule, "prior")] = (p > thr).astype(int)
        preds[("always_up", "-")] = np.ones_like(y)
        rng = np.random.RandomState(SEED)
        rand_f1 = np.array([macro_f1(y, (rng.rand(len(y)) < q).astype(int)) for _ in range(N_BOOT)])

        for (rule, thr), yh in preds.items():
            rng = np.random.RandomState(SEED)
            bs = [macro_f1(y[i], yh[i]) for i in (boot_idx(len(y), rng) for _ in range(N_BOOT))]
            tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
            mf = macro_f1(y, yh)
            rows.append(dict(market=m, rule=rule, threshold=thr, n=len(y),
                             TP=tp, FP=fp, FN=fn, TN=tn,
                             accuracy=accuracy_score(y, yh), up_rate=y.mean(),
                             f1_up=f1_score(y, yh, pos_label=1, zero_division=0),
                             f1_down=f1_score(y, yh, pos_label=0, zero_division=0),
                             macro_f1=mf, macro_f1_ci_low=np.percentile(bs, 2.5),
                             macro_f1_ci_high=np.percentile(bs, 97.5),
                             random_macro_f1_mean=rand_f1.mean(),
                             p_vs_random=float((rand_f1 >= mf).mean())))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    res.to_csv(RES / "v17_nested_f1.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
