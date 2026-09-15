# -*- coding: utf-8 -*-
"""v11 — 원칙적 개선 3종 (두 시장 동일 적용).

동기
----
영국 모델에서 예측 대상인 ISF 자신의 일중 고가·저가·거래량이 결측률 8.3%로
임계값(5%)을 넘어 전부 제외되고 있었다. 최초 유효일이 2014-01-02이므로 앞부분
절단이 아니라 산발적 결측이며, 논문 §3.2가 명시한 forward-fill 방침을 적용하면
복구할 수 있다. 일중 변동폭(High-Low)은 변동성 예측의 표준 피처이므로
(Parkinson 1980 추정량의 기초) 이 누락은 실질적 손실이다.

개선 항목 (모두 두 시장에 동일 적용, 미래 정보 사용 없음)
------------------------------------------------------
  I1. 산발적 결측 열의 전방보간 복구 (대상 ETF의 OHLCV 포함)
  I2. HAR 표준 3성분(일/주/월)을 타깃과 같은 로그 스케일로 명시 추가
  I3. Parkinson 고저가 변동성 추정량 및 그 이동평균 추가

평가는 v10 확정 프로토콜(pooled 학습 + 검증구간 보정 + 블록 부트스트랩)과 동일하다.
"""
from __future__ import annotations

import contextlib, io, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import lightgbm as lgb  # noqa: E402

from _run_enhanced_models_v4 import (  # noqa: E402
    load_market, add_targets_and_ar, build_features,
    TRAIN_FRAC, VAL_FRAC, SEED, NEWS_SENT_COLS,
)
from _explore_targets_v5 import make_targets  # noqa: E402

PCA_DIM, EPS, H = 15, 1e-8, 5
MAX_MISS_STRICT = 0.05      # 기존 기준
MAX_MISS_LOOSE = 0.15       # 전방보간 복구 후 기준
N_BOOT, BLOCK = 5000, 10
KW = dict(n_estimators=600, max_depth=5, learning_rate=0.04, subsample=0.8,
          colsample_bytree=0.6, reg_lambda=5.0, random_state=SEED,
          n_jobs=4, verbose=-1, deterministic=True, force_row_wise=True)

OHLC = {"US": "qqq", "UK": "isf"}   # 각 시장 파일에 존재하는 OHLCV 접두어


def prep(mc, improve: bool):
    df0, emb = load_market(mc)

    if improve:
        # ── I1. 산발적 결측 열 전방보간 (앞부분 절단이 아닌 열만) ──
        for c in df0.columns:
            if c in ("Date", "Headline"):
                continue
            v = pd.to_numeric(df0[c], errors="coerce")
            if v.isna().mean() == 0:
                continue
            fvi = v.first_valid_index()
            # 시계열 앞쪽이 통째로 비어 있는 열(eth 등)은 제외
            if fvi is not None and fvi < len(v) * 0.05:
                df0[c] = v.ffill()

    df = add_targets_and_ar(df0)
    tg = make_targets(df0)

    if improve:
        r = np.log(df0["ETF"].astype(float)).diff()
        a = r.abs()
        # ── I2. HAR 표준 3성분 (로그 스케일, t 시점 관측 가능) ──
        df["har_d_log"] = np.log(a.shift(1) + EPS)
        df["har_w_log"] = np.log(a.shift(1).rolling(5, min_periods=1).mean() + EPS)
        df["har_m_log"] = np.log(a.shift(1).rolling(22, min_periods=5).mean() + EPS)
        # ── I3. Parkinson 고저가 변동성 ──
        pre = OHLC[mc]
        hi, lo = pre + "_high", pre + "_low"
        if hi in df0.columns and lo in df0.columns:
            h_ = pd.to_numeric(df0[hi], errors="coerce")
            l_ = pd.to_numeric(df0[lo], errors="coerce")
            park = np.sqrt((np.log(h_ / l_) ** 2) / (4 * np.log(2)))
            df["park_log"] = np.log(park.shift(1) + EPS)
            df["park_ma5"] = np.log(park.shift(1).rolling(5, min_periods=1).mean() + EPS)
            df["park_ma22"] = np.log(park.shift(1).rolling(22, min_periods=5).mean() + EPS)

    i0 = int(len(df) * TRAIN_FRAC)
    pca = PCA(n_components=PCA_DIM, random_state=SEED).fit(emb[:i0])
    feat, fin, news, ar = build_features(df, pca.transform(emb).astype(np.float32))
    for c in tg.columns:
        feat[c] = tg[c].values
    if improve:
        for c in ("har_d_log", "har_w_log", "har_m_log",
                  "park_log", "park_ma5", "park_ma22"):
            if c in df.columns:
                feat[c] = df[c].values

    meta = {"target_vol", "target_log_vol", "target_dir", "next_ret", "Date"}
    tc = [c for c in feat.columns if c.startswith(("vol", "dir"))]
    cand = [c for c in feat.columns if c not in meta and c not in tc]
    f2 = feat.dropna(subset=["target_vol"])
    miss = f2[cand].isna().mean()
    thr = MAX_MISS_LOOSE if improve else MAX_MISS_STRICT
    keep = [c for c in cand if miss[c] <= thr]

    r = np.log(df0["ETF"].astype(float)).diff()
    feat["rv5_past"] = np.log(np.sqrt((r ** 2).rolling(H, min_periods=H).mean()) + EPS).values
    s = feat.dropna(subset=["vol5", "rv5_past"]).dropna(subset=keep).reset_index(drop=True)
    s = s[np.isfinite(s["vol5"]) & np.isfinite(s["rv5_past"])].reset_index(drop=True)

    ar_c = [c for c in keep if c in ar or c.startswith(("har_", "park_"))]
    return s, keep, ar_c


def block_boot(y, p, n_boot=N_BOOT, block=BLOCK):
    rng = np.random.RandomState(SEED)
    n = len(y); nb = max(1, n // block); out = []
    for _ in range(n_boot):
        st = rng.randint(0, max(1, n - block + 1), nb)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in st])[:n]
        out.append(r2_score(y[idx], p[idx]))
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run(improve: bool, tag: str):
    frames, F = {}, {}
    for mc in ["US", "UK"]:
        with contextlib.redirect_stdout(io.StringIO()):
            frames[mc] = prep(mc, improve)
    common = [c for c in frames["US"][1] if c in frames["UK"][1]]
    ar_common = [c for c in frames["US"][2] if c in frames["UK"][2]]
    CUT_TEST = min(frames[m][0]["Date"].iloc[int(len(frames[m][0]) * 0.85)] for m in frames)
    CUT_TRAIN = min(frames[m][0]["Date"].iloc[int(len(frames[m][0]) * 0.70)] for m in frames)

    print("\n" + "=" * 80)
    print("  [%s]  공통 피처 %d개 (AR %d) | 표본 US %d / UK %d"
          % (tag, len(common), len(ar_common),
             len(frames["US"][0]), len(frames["UK"][0])))

    out = {}
    for gname, cols in [("AR_Only", ar_common), ("Full", common)]:
        Z, Y, PAST, D, MK = {}, {}, {}, {}, {}
        for mc in ["US", "UK"]:
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
        for mc in ["US", "UK"]:
            o = "UK" if mc == "US" else "US"
            tr, va, te = MK[mc]["tr"], MK[mc]["va"], MK[mc]["te"]
            Xp = np.vstack([Z[mc][tr], Z[o][MK[o]["tr"]]])
            yp = np.concatenate([D[mc][tr], D[o][MK[o]["tr"]]])
            m = lgb.LGBMRegressor(**KW).fit(Xp, yp)
            p_va = m.predict(Z[mc][va]) + PAST[mc][va]
            p_te = m.predict(Z[mc][te]) + PAST[mc][te]
            cal = LinearRegression().fit(p_va.reshape(-1, 1), Y[mc][va])
            p = cal.predict(p_te.reshape(-1, 1))
            y = Y[mc][te]
            r2 = r2_score(y, p)
            lo, hi = block_boot(y, p)
            out[(gname, mc)] = (r2, lo, hi, np.corrcoef(y, p)[0, 1])
            star = " *" if lo > 0 else ""
            print("     %-9s %-3s  R2=%+.4f%s  95%%CI [%+.4f, %+.4f]  상관 %.3f"
                  % (gname, mc, r2, star, lo, hi, np.corrcoef(y, p)[0, 1]))
    return out


if __name__ == "__main__":
    base = run(False, "기준 (v10)")
    imp = run(True, "개선 (I1+I2+I3)")
    print("\n" + "=" * 80)
    print("  변화 요약  (* = 신뢰구간이 0을 제외)")
    for k in base:
        b, i = base[k], imp[k]
        print("   %-9s %-3s  %+.4f -> %+.4f  (%+.4f)   CI [%+.3f,%+.3f] -> [%+.3f,%+.3f]"
              % (k[0], k[1], b[0], i[0], i[0] - b[0], b[1], b[2], i[1], i[2]))
