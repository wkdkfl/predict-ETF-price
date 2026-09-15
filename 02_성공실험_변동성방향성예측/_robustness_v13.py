# -*- coding: utf-8 -*-
"""v13 — 사후 강건성 검정 두 가지.

검정 A. FTSE 250 (MIDD) 대체 표적
-------------------------------
영국의 낮은 성능이 FTSE 100 추종 ETF(ISF) 고유의 성질인지, 영국 시장 전반의
성질인지를 구분한다. 피처 파이프라인과 v10 프로토콜을 그대로 두고 예측 표적만
MIDD(FTSE 250 추종)로 바꾼다. FTSE 100은 매출의 약 4분의 3이 해외에서 발생하는
다국적 기업 중심이라 영국 국내 뉴스와의 연결이 약한 반면, FTSE 250은 국내
경기 민감도가 높다. 뉴스 피처가 영국에서 기여하지 못한 이유를 가르는 검정이다.

검정 B. fold 성능의 설명 변수
---------------------------
논문은 당초 '변동성 수준이 높은 fold에서 성능이 좋다'고 서술했으나 미국에서
반대 부호가 나왔다. 수준(level) 대신 분포 이동(covariate shift)이 성능을
설명하는지 검정한다. 학습 분포와 시험 분포가 멀어질수록 성능이 떨어진다는
것은 지도학습의 일반 원리이므로, 두 시장에 동시에 적용되어야 한다.

  L 수준       fold 시험 구간의 평균 log RV5
  S 분포 이동  |평균(시험) - 평균(학습)| / 표준편차(학습)     ← 사전 지정 주 지표
  V 국면 내 변동  fold 시험 구간의 log RV5 표준편차

세 지표를 모두 계산하여 전부 보고한다. 시장별 8 fold와 두 시장 합산 16 fold에
대해 각각 상관을 구한다. 결과의 방향과 무관하게 그대로 기록한다.
"""
from __future__ import annotations

import contextlib, io, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402

from _regime_fixes_v7 import prep  # noqa: E402
from _run_enhanced_models_v4 import (  # noqa: E402
    load_market, add_targets_and_ar, build_features, TRAIN_FRAC, SEED,
)
from _explore_targets_v5 import make_targets  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

EPS, H, PCA_DIM = 1e-8, 5, 15
MAX_MISS = 0.05
N_BOOT, BLOCK, N_FOLDS = 5000, 10, 8
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)


def boot_ci(y, p):
    rng = np.random.RandomState(SEED)
    n = len(y); nb = max(1, n // BLOCK); out = []
    for _ in range(N_BOOT):
        st = rng.randint(0, max(1, n - BLOCK + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]
        out.append(r2_score(y[idx], p[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def prep_alt(mc, price_csv, price_col):
    """v10 파이프라인에서 예측 표적 자산만 교체한다 (피처는 동일)."""
    df0, emb = load_market(mc)
    alt = pd.read_csv(BASE / ("USD" if mc == "US" else "UK") / price_csv)
    alt["Date"] = pd.to_datetime(alt["Date"])
    m = df0[["Date"]].merge(alt[["Date", price_col]], on="Date", how="left")
    px = pd.to_numeric(m[price_col], errors="coerce").ffill()
    cover = px.notna().mean()
    df0 = df0.copy()
    df0["ETF"] = px.values

    df = add_targets_and_ar(df0)
    tg = make_targets(df0)
    i0 = int(len(df) * TRAIN_FRAC)
    pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(emb[:i0])
    feat, fin, news, ar = build_features(df, pca.transform(emb).astype(np.float32))
    for c in tg.columns:
        feat[c] = tg[c].values
    meta = {"target_vol", "target_log_vol", "target_dir", "next_ret", "Date"}
    tc = [c for c in feat.columns if c.startswith(("vol", "dir"))]
    cand = [c for c in feat.columns if c not in meta and c not in tc]
    miss = feat.dropna(subset=["target_vol"])[cand].isna().mean()
    keep = [c for c in cand if miss[c] <= MAX_MISS]
    ar_c = [c for c in keep if c in ar]
    r = np.log(df0["ETF"].astype(float)).diff()
    feat["rv5_past"] = np.log(np.sqrt((r ** 2).rolling(H, min_periods=H).mean()) + EPS).values
    s = feat.dropna(subset=["vol5", "rv5_past"]).dropna(subset=keep).reset_index(drop=True)
    s = s[np.isfinite(s["vol5"]) & np.isfinite(s["rv5_past"])].reset_index(drop=True)
    return s, keep, ar_c, cover


def final_eval(frames):
    """v10 확정 프로토콜: pooled 학습 + 검증구간 MZ 보정 + 블록 부트스트랩."""
    CUT_TEST = min(frames[m][0]["Date"].iloc[int(len(frames[m][0]) * 0.85)] for m in frames)
    CUT_TRAIN = min(frames[m][0]["Date"].iloc[int(len(frames[m][0]) * 0.70)] for m in frames)
    common = [c for c in frames["US"][1] if c in frames["UK"][1]]
    ar_common = [c for c in frames["US"][2] if c in frames["UK"][2]]
    out = {}
    for gname, cols in [("AR_Only", ar_common), ("Full", common)]:
        Z, Y, PAST, D, MK = {}, {}, {}, {}, {}
        for mc in frames:
            s = frames[mc][0]
            tr = (s["Date"] < CUT_TRAIN).values
            sc = StandardScaler().fit(s[cols].values[tr])
            Z[mc] = sc.transform(s[cols].values)
            Y[mc] = s["vol5"].values
            PAST[mc] = s["rv5_past"].values
            D[mc] = Y[mc] - PAST[mc]
            MK[mc] = {"tr": tr,
                      "va": ((s["Date"] >= CUT_TRAIN) & (s["Date"] < CUT_TEST)).values,
                      "te": (s["Date"] >= CUT_TEST).values}
        for mc in frames:
            o = [k for k in frames if k != mc][0]
            tr, va, te = MK[mc]["tr"], MK[mc]["va"], MK[mc]["te"]
            Xp = np.vstack([Z[mc][tr], Z[o][MK[o]["tr"]]])
            yp = np.concatenate([D[mc][tr], D[o][MK[o]["tr"]]])
            m = lgb.LGBMRegressor(**KW).fit(Xp, yp)
            p_va = m.predict(Z[mc][va]) + PAST[mc][va]
            p_te = m.predict(Z[mc][te]) + PAST[mc][te]
            cal = LinearRegression().fit(p_va.reshape(-1, 1), Y[mc][va])
            p = cal.predict(p_te.reshape(-1, 1))
            y = Y[mc][te]
            lo, hi = boot_ci(y, p)
            out[(gname, mc)] = (r2_score(y, p), lo, hi,
                                float(np.corrcoef(y, p)[0, 1]), int(te.sum()))
    return out


def walk_forward(s, keep, ar_c, mc, label):
    """8-fold 확장 윈도우. fold별 성능과 세 가지 국면 지표를 함께 기록한다."""
    y, past = s["vol5"].values, s["rv5_past"].values
    d = y - past
    n = len(s); mt = int(n * 0.35); fs = (n - mt) // N_FOLDS
    rows = []
    for k in range(N_FOLDS):
        a = mt + k * fs; b = min(a + fs, n)
        if b <= a:
            continue
        mu_tr, sd_tr = y[:a].mean(), y[:a].std()
        rec = {"market": mc, "label": label, "fold": k + 1,
               "test_start": str(s["Date"].iloc[a].date()),
               "test_end": str(s["Date"].iloc[b - 1].date()),
               "n_test": b - a,
               "L_level": float(y[a:b].mean()),
               "S_shift": float(abs(y[a:b].mean() - mu_tr) / sd_tr),
               "V_within": float(y[a:b].std())}
        for g, cols in [("AR_Only", ar_c), ("Full", keep)]:
            X = s[cols].values
            sc = StandardScaler().fit(X[:a])
            p = lgb.LGBMRegressor(**KW).fit(sc.transform(X[:a]), d[:a]) \
                   .predict(sc.transform(X[a:b])) + past[a:b]
            rec[g + "_r2"] = float(r2_score(y[a:b], p))
            rec[g + "_corr"] = float(np.corrcoef(y[a:b], p)[0, 1])
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 88)
    print("  검정 A. FTSE 250 (MIDD) 대체 표적 — v10 프로토콜 그대로")
    print("=" * 88)

    base = {}
    for mc in ["US", "UK"]:
        with contextlib.redirect_stdout(io.StringIO()):
            base[mc] = prep(mc)
    r_base = final_eval(base)

    alt = {"US": base["US"]}
    with contextlib.redirect_stdout(io.StringIO()):
        s_a, k_a, ar_a, cov = prep_alt("UK", "UK_MIDD_price.csv", "Price")
    alt["UK"] = (s_a, k_a, ar_a)
    print("  MIDD 가격 커버리지 %.1f%%,  모델링 표본 %d행 (ISF %d행)"
          % (100 * cov, len(s_a), len(base["UK"][0])))
    r_alt = final_eval(alt)

    print()
    print("  %-9s %-34s %s" % ("피처군", "영국 ISF / FTSE 100", "영국 MIDD / FTSE 250"))
    for g in ["AR_Only", "Full"]:
        def fmt(t):
            return "%+.3f [%+.3f,%+.3f] r=%.3f%s" % (t[0], t[1], t[2], t[3],
                                                     " *" if t[1] > 0 else "")
        print("  %-9s %-34s %s" % (g, fmt(r_base[(g, "UK")]), fmt(r_alt[(g, "UK")])))
    print("  (* = 95% 블록 부트스트랩 신뢰구간이 0을 제외)")
    print("  미국 대조군(두 실행 동일): AR_Only %+.3f / Full %+.3f"
          % (r_base[("AR_Only", "US")][0], r_base[("Full", "US")][0]))

    print()
    print("=" * 88)
    print("  검정 B. fold 성능을 설명하는 지표  (16 fold, 사전 지정 주 지표 = S)")
    print("=" * 88)

    fs = []
    for mc, lab in [("US", "미국 SPY"), ("UK", "영국 ISF")]:
        fs.append(walk_forward(*base[mc], mc, lab))
    fold = pd.concat(fs, ignore_index=True)
    fold.to_csv(BASE / "_robustness_v13_folds.csv", index=False, encoding="utf-8-sig")

    IDX = [("L_level", "수준  평균 log RV5"),
           ("S_shift", "분포 이동  |Δμ|/σ_train"),
           ("V_within", "국면 내 변동  σ(log RV5)")]
    print("  %-26s %-22s %-22s %s" % ("지표", "미국 8 fold", "영국 8 fold", "합산 16 fold"))
    for col, name in IDX:
        cells = []
        for d in [fold[fold.market == "US"], fold[fold.market == "UK"], fold]:
            r, p = stats.pearsonr(d[col], d.AR_Only_r2)
            cells.append("%+.3f (p=%.3f)%s" % (r, p, " *" if p < 0.05 else "  "))
        print("  %-26s %-22s %-22s %s" % (name, cells[0], cells[1], cells[2]))
    print("  (* = p<0.05.  부호는 R²와의 상관 방향)")

    print()
    print("  fold별 원자료")
    print("  %-9s %-5s %-20s %7s %7s %7s %8s"
          % ("시장", "fold", "시험 구간", "L", "S", "V", "AR R²"))
    for _, r in fold.iterrows():
        print("  %-9s %-5d %-20s %7.2f %7.2f %7.2f %+8.3f"
              % (r["label"], r["fold"], r["test_start"] + "~" + r["test_end"][:7],
                 r["L_level"], r["S_shift"], r["V_within"], r["AR_Only_r2"]))
    print("\n저장: _robustness_v13_folds.csv")
