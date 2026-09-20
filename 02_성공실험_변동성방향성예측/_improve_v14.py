# -*- coding: utf-8 -*-
"""뉴스 활용 개선 실험 — plan/20260920_improvement_plan.md 의 사전 고정 프로토콜.

후보(피처군 F0~F5 x 모형 lgbm/linear x 타깃 vol/dir)는 이 스크립트 작성 시점에 고정돼 있다.
  1) 선택: 시험 구간(>= CUT_TEST)을 쓰지 않는 확장 윈도우 워크포워드(사전 표본의 뒤쪽 50%, 5블록, 7일 엠바고)
  2) 시험: 선택 결과와 무관하게 모든 후보를 v10 절차(풀링·검증구간 보정)로 1회 평가해 공개
  3) 판정: 선택된 후보가 F0·F1 모두 이기는지(변동성 DM p<0.05 & R² 우위 / 방향성 AUC 우위 & ΔAUC CI 가 0 배제)

기준선 F0(AR_Only), F1(AR+기존 뉴스)의 LightGBM 결과는 재실험_결과/newsv2_full 의 v10 값과 일치해야 하며 불일치하면 중단한다.

  python _improve_v14.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["NEWS_DATA"] = "new"       # 신규 뉴스 데이터
os.environ["PCA_SOLVER"] = "full"     # 결정적 PCA (주 결과와 동일)
warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, RidgeCV  # noqa: E402
from sklearn.metrics import r2_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import _final_v10 as fv  # noqa: E402  (build, block_boot, dm_test, KW)
import _final_v10_direction as fd  # noqa: E402  (block_boot_auc, dm_cls, KW)
from _run_enhanced_models_v4 import NEWS_SENT_COLS  # noqa: E402

OUT = BASE / "개선실험"
OUT.mkdir(exist_ok=True)
SEED, N_BOOT, BLOCK = 42, 5000, 10
EMBARGO = pd.Timedelta(days=7)
N_BLOCKS = 5
ALPHAS = np.logspace(-1, 4, 12)
CS = np.logspace(-3, 1, 9)
MODELS = {"vol": ["lgbm", "linear"], "dir": ["lgbm", "linear"]}
INT_COLS = ["n_cand_rel", "n_hi_rel", "score_mean_rel", "topic_policy_rel", "topic_macro_rel",
            "topic_trade_geo_rel", "topic_market_rel", "topic_corp_rel", "topic_commod_fx_rel"]
SS_SRC = [("sent_score_mean", "score"), ("sent_neg_mean", "neg"), ("sent_max_neg", "maxneg")]
SS_COLS = [f"ss_{nm}_e{sp}" for _, nm in SS_SRC for sp in (5, 20)]
CANDS = ["F2", "F3", "F4", "F5"]           # 신규 후보
BASES = ["F0", "F1"]                        # 기준선


# ------------------------------------------------------------------ 데이터
def load():
    US, keep_us, ar_us = fv.build("US")
    UK, keep_uk, ar_uk = fv.build("UK")
    frames = {"US": US, "UK": UK}
    common = [c for c in keep_us if c in keep_uk]
    ar_c = [c for c in common if c in ar_us and c in ar_uk]
    sent_c = [c for c in common if c in NEWS_SENT_COLS]
    emb_c = [c for c in common if c.startswith("emb_pc")]
    for m, s in list(frames.items()):
        it = pd.read_csv(OUT / "data" / f"{m}_news_intensity.csv", parse_dates=["Date"])
        s2 = s.merge(it[["Date"] + INT_COLS], on="Date", how="left")
        assert not s2[INT_COLS].isna().any().any(), f"{m}: 강도 피처 결측"
        assert len(s2) == len(s)
        for span in (5, 20):                                   # 인과적 지수가중평균(과거+당일)
            for src, nm in SS_SRC:
                s2[f"ss_{nm}_e{span}"] = s2[src].ewm(span=span, adjust=False).mean()
        frames[m] = s2
    FS = {"F0": ar_c, "F1": ar_c + sent_c + emb_c, "F2": ar_c + INT_COLS, "F3": ar_c + INT_COLS + SS_COLS,
          "F4": ar_c + INT_COLS + sent_c, "F5": ar_c + SS_COLS}
    return frames, FS


frames, FS = load()
CUT_TEST = min(s["Date"].iloc[int(len(s) * 0.85)] for s in frames.values())
CUT_TRAIN = min(s["Date"].iloc[int(len(s) * 0.70)] for s in frames.values())
ARR = {m: {"date": s["Date"].values, "y": s["vol5"].values, "past": s["rv5_past"].values,
           "dy": s["vol5"].values - s["rv5_past"].values, "dir": s["dir5"].values.astype(int)}
       for m, s in frames.items()}
print("학습 < %s | 검증 < %s | 시험 >= %s" % (CUT_TRAIN.date(), CUT_TEST.date(), CUT_TEST.date()))
print("피처 수: " + ", ".join(f"{k}={len(v)}" for k, v in FS.items()), flush=True)


# ------------------------------------------------------------------ 모형
def model_fit(task, model, X, y):
    if task == "vol":
        if model == "lgbm":
            return lgb.LGBMRegressor(**fv.KW).fit(X, y)
        return RidgeCV(alphas=ALPHAS, cv=TimeSeriesSplit(5)).fit(X, y)
    if model == "lgbm":
        return lgb.LGBMClassifier(**fd.KW).fit(X, y)
    return LogisticRegressionCV(Cs=CS, cv=TimeSeriesSplit(5), scoring="neg_log_loss",
                                max_iter=3000).fit(X, y)


def fit_predict(fs, task, model, m, cut_train, embargo, masks):
    """시장 m 에 대한 풀링 학습(v10 과 동일한 적층 순서) 후 masks 각각의 예측을 반환."""
    o = "UK" if m == "US" else "US"
    cut_end = cut_train - embargo
    Z, TR = {}, {}
    for k, s in frames.items():
        tr = (s["Date"] < cut_end).values
        sc = StandardScaler().fit(s[FS[fs]].values[tr])
        Z[k], TR[k] = sc.transform(s[FS[fs]].values), tr
    tgt = "dy" if task == "vol" else "dir"
    Xp = np.vstack([Z[m][TR[m]], Z[o][TR[o]]])
    yp = np.concatenate([ARR[m][tgt][TR[m]], ARR[o][tgt][TR[o]]])
    if model == "linear":                                   # 시계열 CV 를 위해 날짜순 정렬
        dp = np.concatenate([ARR[m]["date"][TR[m]], ARR[o]["date"][TR[o]]])
        od = np.argsort(dp, kind="stable")
        Xp, yp = Xp[od], yp[od]
    mdl = model_fit(task, model, Xp, yp)
    outs = []
    for mask in masks:
        if task == "vol":
            outs.append(mdl.predict(Z[m][mask]) + ARR[m]["past"][mask])
        else:
            outs.append(mdl.predict_proba(Z[m][mask])[:, 1])
    return outs


# ------------------------------------------------------------------ 1) 사전 워크포워드 선택 (시험 미사용)
def wf_blocks():
    pre = np.sort(np.unique(np.concatenate([s["Date"].values[s["Date"] < CUT_TEST] for s in frames.values()])))
    idx = np.linspace(len(pre) // 2, len(pre), N_BLOCKS + 1).astype(int)
    edges = [pre[i] if i < len(pre) else CUT_TEST.to_datetime64() for i in idx[:-1]] + [CUT_TEST.to_datetime64()]
    return list(zip(edges[:-1], edges[1:]))


def walk_forward(task):
    rows = []
    blocks = wf_blocks()
    print(f"\n[{task}] 워크포워드 블록 {len(blocks)}개: " + ", ".join(f"{str(a)[:10]}~{str(b)[:10]}" for a, b in blocks), flush=True)
    for fs in BASES + CANDS:
        for model in MODELS[task]:
            ys = {m: [] for m in frames}
            ps = {m: [] for m in frames}
            for bs, be in blocks:
                for m in frames:
                    d = ARR[m]["date"]
                    mask = (d >= bs) & (d < be)
                    if not mask.any():
                        continue
                    (p,) = fit_predict(fs, task, model, m, pd.Timestamp(bs), EMBARGO, [mask])
                    ps[m].append(p)
                    ys[m].append(ARR[m]["y"][mask] if task == "vol" else ARR[m]["dir"][mask])
            yy = {m: np.concatenate(ys[m]) for m in frames}
            pp = {m: np.concatenate(ps[m]) for m in frames}
            if task == "vol":
                mse = float(np.mean(np.concatenate([(yy[m] - pp[m]) ** 2 for m in frames])))
                rows.append({"task": task, "featset": fs, "model": model, "score": mse,
                             "n_oos": int(sum(len(yy[m]) for m in frames))})
            else:
                auc = {m: roc_auc_score(yy[m], pp[m]) for m in frames}
                rows.append({"task": task, "featset": fs, "model": model, "score": float(np.mean(list(auc.values()))),
                             "auc_US": auc["US"], "auc_UK": auc["UK"], "n_oos": int(sum(len(yy[m]) for m in frames))})
            print(f"  {fs} {model:6s} score={rows[-1]['score']:.5f}", flush=True)
    df = pd.DataFrame(rows)
    # 같은 모형의 F0·F1 대비 개선 여부와 선택
    df["vs_F0"], df["vs_F1"], df["qualified"] = np.nan, np.nan, False
    for i, r in df.iterrows():
        b0 = df[(df.featset == "F0") & (df.model == r.model)].score.iloc[0]
        b1 = df[(df.featset == "F1") & (df.model == r.model)].score.iloc[0]
        if task == "vol":            # MSE 개선율
            df.loc[i, ["vs_F0", "vs_F1"]] = [1 - r.score / b0, 1 - r.score / b1]
        else:                        # AUC 차이
            df.loc[i, ["vs_F0", "vs_F1"]] = [r.score - b0, r.score - b1]
        df.loc[i, "qualified"] = bool(r.featset in CANDS and df.loc[i, "vs_F0"] > 0 and df.loc[i, "vs_F1"] > 0)
    q = df[df.qualified]
    df["selected"] = False
    if len(q):
        best = q.sort_values("score", ascending=(task == "vol")).index[0]
        df.loc[best, "selected"] = True
    return df


# ------------------------------------------------------------------ 2) 시험 평가 (v10 절차, 후보 전체 1회)
def paired_boot_dauc(y, p1, p2):
    rng = np.random.RandomState(SEED)
    n = len(y)
    nb = max(1, n // BLOCK)
    out = []
    for _ in range(N_BOOT):
        st = rng.randint(0, max(1, n - BLOCK + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p2[idx]) - roc_auc_score(y[idx], p1[idx]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def test_eval(task):
    res, store = [], {}
    for fs in BASES + CANDS:
        for model in MODELS[task]:
            for m in frames:
                d = ARR[m]["date"]
                va = (d >= CUT_TRAIN.to_datetime64()) & (d < CUT_TEST.to_datetime64())
                te = d >= CUT_TEST.to_datetime64()
                p_va, p_te = fit_predict(fs, task, model, m, CUT_TRAIN, pd.Timedelta(0), [va, te])
                if task == "vol":
                    y_va, y_te = ARR[m]["y"][va], ARR[m]["y"][te]
                    cal = LinearRegression().fit(p_va.reshape(-1, 1), y_va)
                    p_use = cal.predict(p_te.reshape(-1, 1))
                    lo, hi = fv.block_boot(y_te, p_use, r2_score)
                    res.append({"task": task, "featset": fs, "model": model, "market": m,
                                "r2": r2_score(y_te, p_use), "ci_low": lo, "ci_high": hi, "n_test": int(te.sum())})
                else:
                    y_va, y_te = ARR[m]["dir"][va], ARR[m]["dir"][te]
                    plat = LogisticRegression().fit(p_va.reshape(-1, 1), y_va)
                    p_use = plat.predict_proba(p_te.reshape(-1, 1))[:, 1]
                    lo, hi = fd.block_boot_auc(y_te, p_te)
                    res.append({"task": task, "featset": fs, "model": model, "market": m,
                                "auc": roc_auc_score(y_te, p_te), "ci_low": lo, "ci_high": hi, "n_test": int(te.sum())})
                store[(fs, model, m)] = (y_te, p_use, p_te)
        print(f"  [{task}] {fs} 시험 평가 완료", flush=True)
    df = pd.DataFrame(res)
    for b in BASES:                                           # 새 열은 미리 만들어 둔다
        for c in (["dm_vs_", "p_vs_"] if task == "vol" else ["dm_vs_", "p_vs_", "dauc_vs_", "dauc_lo_", "dauc_hi_"]):
            df[c + b] = np.nan
    # 기준선 = v10 프로토콜의 F0·F1 (lgbm) 대비 검정
    for i, r in df.iterrows():
        y, p_use, p_raw = store[(r.featset, r.model, r.market)]
        for b in BASES:
            yb, pb_use, pb_raw = store[(b, "lgbm", r.market)]
            if task == "vol":
                s_, p_ = fv.dm_test(yb, pb_use, p_use)
                df.loc[i, [f"dm_vs_{b}", f"p_vs_{b}"]] = [s_, p_]
            else:
                s_, p_ = fd.dm_cls(yb.astype(float), pb_use, p_use)
                lo, hi = paired_boot_dauc(yb, pb_raw, p_raw)
                df.loc[i, [f"dm_vs_{b}", f"p_vs_{b}", f"dauc_vs_{b}", f"dauc_lo_{b}", f"dauc_hi_{b}"]] = [
                    s_, p_, roc_auc_score(y, p_raw) - roc_auc_score(yb, pb_raw), lo, hi]
    return df


# ------------------------------------------------------------------ 기준선 재현 검증
def check_baselines(vol_test, dir_test):
    ref_v = pd.read_csv(BASE / "재실험_결과" / "newsv2_full" / "_final_v10_volatility.csv").set_index(["market", "group"])
    ref_d = pd.read_csv(BASE / "재실험_결과" / "newsv2_full" / "_final_v10_direction.csv").set_index(["market", "group"])
    bad = []
    for m in frames:
        for fs, g in (("F0", "AR_Only"), ("F1", "AR+News")):
            a = vol_test[(vol_test.featset == fs) & (vol_test.model == "lgbm") & (vol_test.market == m)].r2.iloc[0]
            b = dir_test[(dir_test.featset == fs) & (dir_test.model == "lgbm") & (dir_test.market == m)].auc.iloc[0]
            if abs(a - ref_v.loc[(m, g), "r2"]) > 1e-9:
                bad.append(("vol", m, fs, a, ref_v.loc[(m, g), "r2"]))
            if abs(b - ref_d.loc[(m, g), "auc"]) > 1e-9:
                bad.append(("dir", m, fs, b, ref_d.loc[(m, g), "auc"]))
    if bad:
        print("기준선 불일치:", bad)
        sys.exit("기준선(F0/F1)이 v10 재실험 값과 다름 — 중단")
    print("기준선 F0·F1(lgbm) 이 재실험_결과/newsv2_full 의 v10 값과 정확히 일치")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    wf_v = walk_forward("vol")
    wf_d = walk_forward("dir")
    wf = pd.concat([wf_v, wf_d], ignore_index=True)
    wf.to_csv(OUT / "selection_walkforward.csv", index=False, encoding="utf-8-sig")
    print("\n선택 결과:\n", wf[wf.selected][["task", "featset", "model", "score", "vs_F0", "vs_F1"]].to_string(index=False)
          if wf.selected.any() else "없음 (어떤 후보도 F0·F1 을 넘지 못함)", flush=True)

    print("\n시험 구간 평가 (모든 후보, 1회)")
    tv = test_eval("vol")
    td = test_eval("dir")
    check_baselines(tv, td)
    tv.to_csv(OUT / "test_volatility.csv", index=False, encoding="utf-8-sig")
    td.to_csv(OUT / "test_direction.csv", index=False, encoding="utf-8-sig")
    print("저장 완료")
