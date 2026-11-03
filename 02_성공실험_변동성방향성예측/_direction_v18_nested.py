# -*- coding: utf-8 -*-
"""v18 방향성 — fold 별 확률 표준화(CAL) + 확장 금융 피처(FIN_ext) + dir20 전체 기간 평가.

사전 고정 프로토콜: ../plan/20261028_direction_v18_plan.md (실행 전 작성, 결과를 본 뒤 바꾸지 않는다).
중첩 워크포워드 구조는 v17 과 같고, 후보는 6 피처군 × 7 분류기 = 42 개.

  PCA_SOLVER=full TARGET=dir5  RESULT_DIR=방향성개선_v18 python _direction_v18_nested.py
  PCA_SOLVER=full TARGET=dir20 RESULT_DIR=방향성개선_v18 python _direction_v18_nested.py
"""
from __future__ import annotations

import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from _news_variant import RES  # noqa: E402
from _direction_v16_walkforward import build, holm, SEED, N_FOLDS, START_Q, PAPER_TEST_START  # noqa: E402
from _direction_v17_nested import CLASSIFIERS, N_INNER, INNER_SPAN, TOP_K  # noqa: E402

TARGET = os.environ.get("TARGET", "dir5")
H = {"dir5": 5, "dir20": 20}[TARGET]
EMBARGO = BLOCK = 2 * H
N_BOOT = 5000
EXT_COLS = ["vix_term", "vvix_dev", "credit_5d", "slope", "slope_5d", "tlt_5d", "uup_5d"]
V17_PRED = BASE / "방향성개선_v17" / "v17_nested_predictions.csv"


# 확장 금융 피처 7열 (모두 t-1 시차, 직전 값으로만 짧은 결측 보간)
def add_fin_ext(s):
    s = s.sort_values("Date").reset_index(drop=True)
    raw = s[["vix9d", "vvix", "hyg", "lqd", "us5y", "us30y", "tlt", "uup"]].ffill(limit=5)
    lag = raw.shift(1)
    s["vix_term"] = np.log(lag["vix9d"] / s["vix"])  # s["vix"] 는 이미 t-1 수준
    lv = np.log(lag["vvix"])
    s["vvix_dev"] = lv - lv.rolling(20, min_periods=20).mean()
    s["credit_5d"] = np.log(lag["hyg"] / lag["lqd"]).diff(5)
    s["slope"] = lag["us30y"] - lag["us5y"]
    s["slope_5d"] = s["slope"].diff(5)
    s["tlt_5d"] = np.log(lag["tlt"]).diff(5)
    s["uup_5d"] = np.log(lag["uup"]).diff(5)
    return s


# 엠바고를 적용한 학습 행 인덱스
def train_rows(s, before):
    idx = np.where((s["Date"] < before).values)[0]
    return idx[:-EMBARGO] if len(idx) > EMBARGO else idx[:0]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# pooled 학습 후 시장별 평가 확률과 학습 행 예측 logit 의 평균·표준편차
def fit_predict(frames, feats, clf, tr, te):
    ok = {m: np.isfinite(frames[m][feats].values[tr[m]]).all(1) for m in frames}
    tr = {m: tr[m][ok[m]] for m in frames}
    Xp = np.vstack([frames[m][feats].values[tr[m]] for m in frames])
    yp = np.concatenate([frames[m][TARGET].values[tr[m]] for m in frames]).astype(int)
    sc = StandardScaler().fit(Xp)
    mdl = CLASSIFIERS[clf]().fit(sc.transform(Xp), yp)
    out = {}
    for m in frames:
        p_te = mdl.predict_proba(sc.transform(frames[m][feats].values[te[m]]))[:, 1]
        l_tr = _logit(mdl.predict_proba(sc.transform(frames[m][feats].values[tr[m]]))[:, 1])
        sd = l_tr.std()
        out[m] = dict(raw=p_te, cal=(_logit(p_te) - l_tr.mean()) / (sd if sd > 1e-9 else 1.0))
    return out


# 외부 학습 구간 안 내부 워크포워드 AUC (v17 과 같은 구조)
def inner_scores(frames, cands, featsets, tr):
    dates = np.sort(np.concatenate([frames[m]["Date"].values[tr[m]] for m in frames]))
    lo = int(len(dates) * (1 - INNER_SPAN)); step = (len(dates) - lo) // N_INNER
    edges = [pd.Timestamp(dates[lo + j * step]) for j in range(N_INNER)] + [pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)]
    sc = {k: {m: [] for m in frames} for k in cands}
    for j in range(N_INNER):
        itr = {m: np.intersect1d(tr[m], train_rows(s, edges[j])) for m, s in frames.items()}
        ite = {m: tr[m][(s["Date"].values[tr[m]] >= np.datetime64(edges[j]))
                        & (s["Date"].values[tr[m]] < np.datetime64(edges[j + 1]))]
               for m, s in frames.items()}
        for f, c in cands:
            p = fit_predict(frames, featsets[f], c, itr, ite)
            for m, s in frames.items():
                sc[(f, c)][m].append(roc_auc_score(s[TARGET].values[ite[m]].astype(int), p[m]["raw"]))
    return {k: {m: float(np.mean(v[m])) for m in frames} for k, v in sc.items()}


# 블록 부트스트랩: AUC CI, 단측 p, (선택) 쌍체 ΔAUC CI
def boot_auc(y, p, p_ref=None):
    rng = np.random.RandomState(SEED)
    n = len(y); nb = max(1, n // BLOCK)
    a, d = [], []
    for _ in range(N_BOOT):
        st = rng.randint(0, max(1, n - BLOCK + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], p[idx]))
        if p_ref is not None:
            d.append(a[-1] - roc_auc_score(y[idx], p_ref[idx]))
    a = np.array(a)
    res = dict(ci_low=np.percentile(a, 2.5), ci_high=np.percentile(a, 97.5), p_gt_half=float((a <= 0.5).mean()))
    if p_ref is not None:
        res.update(d_low=np.percentile(d, 2.5), d_high=np.percentile(d, 97.5))
    return res


def main():
    frames, keeps, ars = {}, {}, {}
    for m in ["US", "UK"]:
        s, keeps[m], ars[m] = build(m)
        frames[m] = add_fin_ext(s).dropna(subset=[TARGET]).reset_index(drop=True)
    from _run_enhanced_models_v4 import NEWS_SENT_COLS
    full_c = [c for c in keeps["US"] if c in keeps["UK"]]
    emb_c = [c for c in full_c if c.startswith("emb_pc")]
    sent_c = [c for c in full_c if c in NEWS_SENT_COLS]
    ar_c = [c for c in full_c if c in ars["US"] and c in ars["UK"]]
    fin_clean = [c for c in full_c if c not in emb_c + sent_c + ar_c and not c.startswith("sent_")]
    featsets = {"AR": ar_c, "FIN_clean": fin_clean, "AR+FIN_clean": ar_c + fin_clean, "Full": full_c,
                "AR+FIN_ext": ar_c + EXT_COLS, "AR+FIN_clean+FIN_ext": ar_c + fin_clean + EXT_COLS}
    all_cands = [(f, c) for f in featsets for c in CLASSIFIERS]

    us_dates = frames["US"]["Date"].values
    n = len(us_dates); a0 = int(n * START_Q); fs = (n - a0) // N_FOLDS
    edges = [pd.Timestamp(us_dates[a0 + k * fs]) for k in range(N_FOLDS)] + [pd.Timestamp.max]

    rules = ["best_raw", "best_cal", "top3_raw", "top3_cal"]
    oos = {m: {"Date": [], "y": [], "fold": [], **{r: [] for r in rules}} for m in frames}
    sel_rows = []
    for k in range(N_FOLDS):
        tr = {m: train_rows(s, edges[k]) for m, s in frames.items()}
        te = {m: np.where(((s["Date"] >= edges[k]) & (s["Date"] < edges[k + 1])).values)[0]
              for m, s in frames.items()}
        # 평가 행에 결측이 있는 피처군(FIN_ext, fold 8)은 후보에서 제외 — 사전 규칙
        cands = [(f, c) for f, c in all_cands
                 if all(np.isfinite(frames[m][featsets[f]].values[te[m]]).all() for m in frames)]
        scores = inner_scores(frames, cands, featsets, tr)
        cache = {}
        for m, s in frames.items():
            ranked = sorted(scores, key=lambda key: -scores[key][m])
            for key in ranked[:TOP_K]:
                if key not in cache:
                    cache[key] = fit_predict(frames, featsets[key[0]], key[1], tr, te)
            top = ranked[:TOP_K]
            oos[m]["best_raw"].append(cache[top[0]][m]["raw"])
            oos[m]["best_cal"].append(cache[top[0]][m]["cal"])
            oos[m]["top3_raw"].append(np.mean([cache[key][m]["raw"] for key in top], axis=0))
            oos[m]["top3_cal"].append(np.mean([cache[key][m]["cal"] for key in top], axis=0))
            y_f = s[TARGET].values[te[m]].astype(int)
            oos[m]["Date"].append(s["Date"].values[te[m]])
            oos[m]["y"].append(y_f)
            oos[m]["fold"].append(np.full(len(te[m]), k + 1))
            for r, key in enumerate(top):
                sel_rows.append(dict(fold=k + 1, market=m, rank=r + 1, featset=key[0], clf=key[1],
                                     n_cands=len(cands), inner_auc=scores[key][m],
                                     outer_auc=roc_auc_score(y_f, cache[key][m]["raw"])))
            print("[%s] fold %d %s 후보 %d  선택: %-22s %-13s 내부 %.3f -> 외부 %.3f"
                  % (TARGET, k + 1, m, len(cands), top[0][0], top[0][1], scores[top[0]][m],
                     roc_auc_score(y_f, cache[top[0]][m]["raw"])), flush=True)

    v17 = pd.read_csv(V17_PRED, parse_dates=["Date"]) if TARGET == "dir5" else None
    rows, pred_frames = [], []
    for m in frames:
        o = {key: np.concatenate(v) for key, v in oos[m].items()}
        pf = pd.DataFrame({"market": m, **o})
        pred_frames.append(pf)
        ref = None
        if v17 is not None:
            mg = pf[["Date"]].merge(v17[v17.market == m][["Date", "top3"]], on="Date", how="left")
            ref = mg["top3"].values if mg["top3"].notna().all() else None
        paper = pd.to_datetime(o["Date"]) >= PAPER_TEST_START
        for rule in rules:
            b = boot_auc(o["y"], o[rule], ref)
            folds = [roc_auc_score(o["y"][o["fold"] == f], o[rule][o["fold"] == f]) for f in range(1, N_FOLDS + 1)]
            rows.append(dict(target=TARGET, market=m, rule=rule, n_oos=len(o["y"]), up_rate=float(o["y"].mean()),
                             auc=roc_auc_score(o["y"], o[rule]), ci_low=b["ci_low"], ci_high=b["ci_high"],
                             p_gt_half=b["p_gt_half"], d_vs_v17top3_low=b.get("d_low"),
                             d_vs_v17top3_high=b.get("d_high"), mean_fold_auc=float(np.mean(folds)),
                             folds_above_half=int(sum(a > 0.5 for a in folds)),
                             auc_paper_test_ref=roc_auc_score(o["y"][paper], o[rule][paper])))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p_gt_half"].values)

    print("\n" + "=" * 110)
    print("  [%s] 중첩 워크포워드 OOS (42 후보, 엠바고·블록 %d)" % (TARGET, BLOCK))
    for _, r in res.iterrows():
        d = "" if pd.isna(r.d_vs_v17top3_low) else "  ΔvsV17top3 [%+.3f, %+.3f]" % (r.d_vs_v17top3_low, r.d_vs_v17top3_high)
        print("  %-3s %-9s n=%d 상승%.3f  AUC %.3f [%.3f, %.3f]  Holm p=%.3f  fold>0.5 %d/8  fold평균 %.3f%s"
              % (r.market, r.rule, r.n_oos, r.up_rate, r.auc, r.ci_low, r.ci_high, r.p_holm,
                 r.folds_above_half, r.mean_fold_auc, d))

    res.to_csv(RES / ("v18_%s_summary.csv" % TARGET), index=False, encoding="utf-8-sig")
    pd.DataFrame(sel_rows).to_csv(RES / ("v18_%s_selections.csv" % TARGET), index=False, encoding="utf-8-sig")
    pd.concat(pred_frames).to_csv(RES / ("v18_%s_predictions.csv" % TARGET), index=False, encoding="utf-8-sig")
    print("\n저장: %s" % RES)


if __name__ == "__main__":
    main()
