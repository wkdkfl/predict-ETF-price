# -*- coding: utf-8 -*-
"""v16 방향성 — 변동성 예측값 피처(방안 2) + 전체 기간 확장 워크포워드 평가(방안 3).

사전 고정 프로토콜: ../plan/20261007_direction_v16_plan.md (실행 전 작성, 결과를 본 뒤 바꾸지 않는다).

  PCA_SOLVER=full RESULT_DIR=방향성개선_v16 python _direction_v16_walkforward.py
"""
from __future__ import annotations

import io, contextlib, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402
from _news_variant import RES  # noqa: E402

SEED, N_BOOT, BLOCK = 42, 5000, 10
N_FOLDS, START_Q, EMBARGO, N_INNER = 8, 0.35, 10, 5
PAPER_TEST_START = pd.Timestamp("2024-03-07")
LGB_KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
              colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
              n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)
SHALLOW_KW = dict(n_estimators=300, max_depth=2, learning_rate=0.02, min_child_samples=200,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
                  random_state=SEED, n_jobs=4, verbose=-1, deterministic=True,
                  force_row_wise=True)
LOGIT_C = 0.01
MODELS = ["A_v10", "B_v15", "C_vol", "D_ens"]
VOL_COLS = ["vhat", "vhat_chg", "vrp", "vix_chg_5d", "vix_ma5_ratio"]


# 논문 §4.5 와 같은 최종 피처 프레임 생성
def build(mc):
    from _regime_fixes_v7 import prep
    with contextlib.redirect_stdout(io.StringIO()):
        return prep(mc)


# L2 로지스틱 분류기(C=0.01) 생성
def logit():
    return LogisticRegression(C=LOGIT_C, max_iter=2000, solver="lbfgs")


# 변동성 예측 모형 학습
def fit_vol(s, idx, ar_c):
    """AR_Only LightGBM 변동성 모형 (v10 설정, 차분 표적)."""
    d = s["vol5"].values[idx] - s["rv5_past"].values[idx]
    return lgb.LGBMRegressor(**LGB_KW).fit(s[ar_c].values[idx], d)


# 변동성 예측값(로그 RV5) 복원
def pred_vol(mdl, s, idx, ar_c):
    return mdl.predict(s[ar_c].values[idx]) + s["rv5_past"].values[idx]


# 엠바고를 적용한 학습 행 인덱스
def train_rows(s, before):
    """before 이전 행에서 마지막 EMBARGO 행을 뺀 인덱스."""
    idx = np.where((s["Date"] < before).values)[0]
    return idx[:-EMBARGO] if len(idx) > EMBARGO else idx[:0]


# fold 별 변동성 예측 피처(VOL 5열) 생성
def vol_features(frames, tr, te, ar_c):
    """fold 별 VOL 피처. 학습 행 vhat = 내부 확장 OOF, 평가 행 vhat = 학습 전체 적합."""
    out = {}
    tr_dates = np.sort(np.concatenate([frames[m]["Date"].values[tr[m]] for m in frames]))
    edges = [pd.Timestamp(tr_dates[int(len(tr_dates) * j / (N_INNER + 1))])
             for j in range(1, N_INNER + 1)] + [pd.Timestamp.max]
    for m, s in frames.items():
        out[m] = {"tr": np.full(len(tr[m]), np.nan), "te": None}
    for j in range(N_INNER):
        fit_idx = {m: np.intersect1d(tr[m], train_rows(s, edges[j])) for m, s in frames.items()}
        mdl = fit_vol(pd.concat([frames[m].iloc[fit_idx[m]] for m in frames]),
                      slice(None), ar_c)
        for m, s in frames.items():
            dts = s["Date"].values[tr[m]]
            sel = (dts >= np.datetime64(edges[j])) & (dts < np.datetime64(edges[j + 1]))
            if sel.any():
                out[m]["tr"][sel] = pred_vol(mdl, s, tr[m][sel], ar_c)
    mdl = fit_vol(pd.concat([frames[m].iloc[tr[m]] for m in frames]), slice(None), ar_c)
    for m, s in frames.items():
        out[m]["te"] = pred_vol(mdl, s, te[m], ar_c)

    def assemble(s, idx, vhat):
        rv = s["rv5_past"].values[idx]
        implied = np.log(s["vix"].values[idx] / 100.0 / np.sqrt(252.0))
        return np.column_stack([vhat, vhat - rv, implied - rv,
                                s["vix_chg_5d"].values[idx], s["vix_ma5_ratio"].values[idx]])

    return {m: {"tr": assemble(s, tr[m], out[m]["tr"]), "te": assemble(s, te[m], out[m]["te"])}
            for m, s in frames.items()}


# 블록 부트스트랩 표본 인덱스
def boot_idx(n, rng):
    nb = max(1, n // BLOCK)
    st = rng.randint(0, max(1, n - BLOCK + 1), nb)
    return np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]


# AUC 신뢰구간·단측 p·쌍체 ΔAUC 계산
def boot_auc(y, p, p_ref=None):
    """AUC CI, 단측 p(AUC<=0.5), 그리고 p_ref 가 있으면 쌍체 ΔAUC CI."""
    rng = np.random.RandomState(SEED)
    a, d = [], []
    for _ in range(N_BOOT):
        idx = boot_idx(len(y), rng)
        if len(np.unique(y[idx])) < 2:
            continue
        a.append(roc_auc_score(y[idx], p[idx]))
        if p_ref is not None:
            d.append(a[-1] - roc_auc_score(y[idx], p_ref[idx]))
    a = np.array(a)
    res = dict(ci_low=np.percentile(a, 2.5), ci_high=np.percentile(a, 97.5),
               p_gt_half=float((a <= 0.5).mean()))
    if p_ref is not None:
        d = np.array(d)
        res.update(d_low=np.percentile(d, 2.5), d_high=np.percentile(d, 97.5))
    return res


# Holm 다중 비교 보정
def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * pvals[i])); adj[i] = run
    return adj


# 8-fold 워크포워드로 4개 모형을 평가하고 결과 저장
def main():
    US, keep_us, ar_us = build("US")
    UK, keep_uk, ar_uk = build("UK")
    frames = {"US": US, "UK": UK}
    full_c = [c for c in keep_us if c in keep_uk]
    ar_c = [c for c in full_c if c in ar_us and c in ar_uk]

    us_dates = US["Date"].values
    n = len(us_dates); a0 = int(n * START_Q); fs = (n - a0) // N_FOLDS
    edges = [pd.Timestamp(us_dates[a0 + k * fs]) for k in range(N_FOLDS)] + [pd.Timestamp.max]

    oos = {m: {"Date": [], "y": [], "fold": [], **{k: [] for k in MODELS}} for m in frames}
    # fold 별 학습·예측
    for k in range(N_FOLDS):
        tr = {m: train_rows(s, edges[k]) for m, s in frames.items()}
        te = {m: np.where(((s["Date"] >= edges[k]) & (s["Date"] < edges[k + 1])).values)[0]
              for m, s in frames.items()}
        Y = {m: s["dir5"].values.astype(int) for m, s in frames.items()}
        V = vol_features(frames, tr, te, ar_c)

        def stack(cols, extra=None):
            Xtr = {m: s[cols].values[tr[m]] for m, s in frames.items()}
            Xte = {m: s[cols].values[te[m]] for m, s in frames.items()}
            if extra is not None:
                Xtr = {m: np.hstack([Xtr[m], extra[m]["tr"]]) for m in frames}
                Xte = {m: np.hstack([Xte[m], extra[m]["te"]]) for m in frames}
            ok = {m: np.isfinite(Xtr[m]).all(1) for m in frames}
            Xp = np.vstack([Xtr[m][ok[m]] for m in frames])
            yp = np.concatenate([Y[m][tr[m]][ok[m]] for m in frames])
            sc = StandardScaler().fit(Xp)
            return sc.transform(Xp), yp, {m: sc.transform(Xte[m]) for m in frames}

        P = {m: {} for m in frames}
        Xp, yp, Xte = stack(full_c)
        mdl = lgb.LGBMClassifier(**LGB_KW).fit(Xp, yp)
        for m in frames:
            P[m]["A_v10"] = mdl.predict_proba(Xte[m])[:, 1]
        Xp, yp, Xte = stack(ar_c)
        mdl = logit().fit(Xp, yp)
        for m in frames:
            P[m]["B_v15"] = mdl.predict_proba(Xte[m])[:, 1]
        Xp, yp, Xte = stack(ar_c, V)
        mdl_c = logit().fit(Xp, yp)
        mdl_t = lgb.LGBMClassifier(**SHALLOW_KW).fit(Xp, yp)
        for m in frames:
            pc = mdl_c.predict_proba(Xte[m])[:, 1]
            P[m]["C_vol"] = pc
            P[m]["D_ens"] = 0.5 * (pc + mdl_t.predict_proba(Xte[m])[:, 1])

        for m, s in frames.items():
            oos[m]["Date"].append(s["Date"].values[te[m]])
            oos[m]["y"].append(Y[m][te[m]])
            oos[m]["fold"].append(np.full(len(te[m]), k + 1))
            for key in MODELS:
                oos[m][key].append(P[m][key])
        print("fold %d  %s ~  학습 US %d / UK %d  평가 US %d / UK %d"
              % (k + 1, edges[k].date(), len(tr["US"]), len(tr["UK"]),
                 len(te["US"]), len(te["UK"])))

    # fold 를 합친 OOS 지표 계산
    rows, fold_rows, pred_frames = [], [], []
    for m in frames:
        o = {key: np.concatenate(v) for key, v in oos[m].items()}
        pf = pd.DataFrame({"market": m, "Date": o["Date"], "fold": o["fold"], "y": o["y"],
                           **{key: o[key] for key in MODELS}})
        pred_frames.append(pf)
        y = o["y"]
        paper = pd.to_datetime(o["Date"]) >= PAPER_TEST_START
        for key in MODELS:
            b = boot_auc(y, o[key], None if key == "A_v10" else o["A_v10"])
            rb = boot_auc(y, o[key], o["B_v15"]) if key in ("C_vol", "D_ens") else {}
            rows.append(dict(market=m, model=key, n_oos=len(y), auc=roc_auc_score(y, o[key]),
                             ci_low=b["ci_low"], ci_high=b["ci_high"], p_gt_half=b["p_gt_half"],
                             d_vs_A_low=b.get("d_low"), d_vs_A_high=b.get("d_high"),
                             d_vs_B_low=rb.get("d_low"), d_vs_B_high=rb.get("d_high"),
                             mean_fold_auc=np.mean([roc_auc_score(y[o["fold"] == f], o[key][o["fold"] == f])
                                                    for f in range(1, N_FOLDS + 1)]),
                             folds_above_half=int(sum(roc_auc_score(y[o["fold"] == f], o[key][o["fold"] == f]) > 0.5
                                                      for f in range(1, N_FOLDS + 1))),
                             auc_paper_test_ref=roc_auc_score(y[paper], o[key][paper])))
            for f in range(1, N_FOLDS + 1):
                sel = o["fold"] == f
                fold_rows.append(dict(market=m, model=key, fold=f,
                                      start=str(pd.Timestamp(o["Date"][sel][0]).date()),
                                      n=int(sel.sum()), up_rate=float(y[sel].mean()),
                                      auc=roc_auc_score(y[sel], o[key][sel])))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p_gt_half"].values)

    print("\n" + "=" * 100)
    print("  5일 방향성 — 전체 기간 8-fold 확장 워크포워드 OOS (pooled 학습, 엠바고 %d행)" % EMBARGO)
    print("  %-6s %-4s %6s %7s %17s %7s %8s %10s %12s"
          % ("모형", "시장", "n", "AUC", "95% CI", "Holm p", "fold>0.5", "fold평균", "논문시험(참고)"))
    for _, r in res.iterrows():
        print("  %-6s %-4s %6d %7.3f   [%.3f, %.3f] %7.3f %6d/%d %10.3f %12.3f"
              % (r.model, r.market, r.n_oos, r.auc, r.ci_low, r.ci_high, r.p_holm,
                 r.folds_above_half, N_FOLDS, r.mean_fold_auc, r.auc_paper_test_ref))
    print("\n  쌍체 ΔAUC 95% CI")
    for _, r in res[res.model != "A_v10"].iterrows():
        s = "  %-6s %-4s vs A_v10 [%+.3f, %+.3f]" % (r.model, r.market, r.d_vs_A_low, r.d_vs_A_high)
        if pd.notna(r.d_vs_B_low):
            s += "   vs B_v15 [%+.3f, %+.3f]" % (r.d_vs_B_low, r.d_vs_B_high)
        print(s)

    res.to_csv(RES / "v16_walkforward_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(RES / "v16_walkforward_folds.csv", index=False, encoding="utf-8-sig")
    pd.concat(pred_frames).to_csv(RES / "v16_walkforward_predictions.csv", index=False,
                                  encoding="utf-8-sig")
    print("\n저장: %s" % RES)


if __name__ == "__main__":
    main()
