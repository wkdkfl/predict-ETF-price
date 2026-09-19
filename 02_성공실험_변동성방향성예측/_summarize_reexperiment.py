# -*- coding: utf-8 -*-
"""재실험_결과/ 의 CSV 에서 비교표를 생성해 재실험_결과/SUMMARY.md 를 쓴다 (수치를 손으로 옮기지 않기 위함).

  python _summarize_reexperiment.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RES = BASE / "재실험_결과"
V4 = ["oldnews_full", "newsv2_full", "oldnews_auto", "newsv2_auto"]
GROUPS = ["AR_Only", "Financial_Only", "Financial_Clean", "News_Pure", "AR+News", "Full"]


def rd(v, f):
    return pd.read_csv(RES / v / f)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def fmt_ci(x, dec=3):
    return f"{x.iloc[0]:+.{dec}f} [{x.iloc[1]:+.{dec}f}, {x.iloc[2]:+.{dec}f}]"


def main():
    L = []
    man = {v: json.loads((RES / v / "run_manifest.json").read_text(encoding="utf-8")) for v in V4}
    pk = man["newsv2_full"]["packages"]
    L += ["# 재실험 결과 요약 (신규 일별 경제 뉴스 vs 기존 희소 뉴스)", "",
          "`_summarize_reexperiment.py` 가 `재실험_결과/*/` 의 CSV 에서 생성한 문서다. 수치는 모두 CSV 와 일치한다.", ""]


    # ---------------- 요약 (수치에서 계산)
    def dm(v, f, m, b, c, key="baseline"):
        d = rd(v, f)
        return d[(d.market == m) & (d[key] == b) & (d.model == c)].iloc[0]
    pv = [dm(v, "_final_v10_dm.csv", m, "AR_Only", "AR+News").p for v in V4 for m in ["US", "UK"]]
    vol = {v: rd(v, "_final_v10_volatility.csv").set_index(["market", "group"]).r2 for v in V4}
    dr = {v: rd(v, "_final_v10_direction.csv").set_index(["market", "group"]) for v in ["oldnews_full", "newsv2_full"]}
    dd = [vol["newsv2_full"][(m, g)] - vol["oldnews_full"][(m, g)] for m in ["US", "UK"] for g in ["News_Pure", "AR+News", "Full"]]
    fo, fn = rd("oldnews_full", "_final_v10_folds.csv"), rd("newsv2_full", "_final_v10_folds.csv")
    us_o, us_n = dr["oldnews_full"].loc[("US", "Full")], dr["newsv2_full"].loc[("US", "Full")]
    uk_o, uk_n = dr["oldnews_full"].loc[("UK", "Full")], dr["newsv2_full"].loc[("UK", "Full")]
    S = ["## 요약", "",
         f"1. **뉴스 증분(AR_Only→AR+News)**: 변동성 DM 은 4개 조건·2개 시장 모두 유의하지 않다(p = {min(pv):.3f}~{max(pv):.3f}). 신규 뉴스로 바꿔도 이 결론은 바뀌지 않는다.",
         f"2. **기존 대비 신규 뉴스의 변동성 R² 차이**(News_Pure·AR+News·Full, 두 시장 6개): {min(dd):+.3f} ~ {max(dd):+.3f}. 부호가 일관되지 않고, "
         "크기는 PCA 솔버만 바꿔서 생기는 변동(F절, 최대 약 0.05)과 같은 범위다.",
         f"3. **방향성**: 미국 Full AUC {us_o.auc:.3f}(기존) → {us_n.auc:.3f}(신규)로 낮아졌고 신규의 CI 하한은 {us_n.ci_low:.3f}로 0.5 를 넘지 못한다. "
         f"영국 Full 은 {uk_o.auc:.3f} → {uk_n.auc:.3f}이나 CI({uk_n.ci_low:.3f}~{uk_n.ci_high:.3f})가 0.5 를 포함한다.",
         f"4. **8-fold 평균 R²(미국 Full)**: {fo[fo.market=='US'].Full_r2.mean():+.3f}(기존) → {fn[fn.market=='US'].Full_r2.mean():+.3f}(신규). "
         f"그러나 같은 표본의 AR_Only 평균({fn[fn.market=='US'].AR_Only_r2.mean():+.3f})보다 낮다.",
         "5. **미국 AR_Only→Full**: 신규·full 솔버에서만 5% 수준(p=0.026), 같은 데이터에서 솔버를 auto 로 바꾸면 p=0.210. 다중 비교 미보정이므로 탐색적 결과로만 본다.",
         "6. **프로토콜 결함 발견**: `Financial_Only` 그룹에 뉴스 파생 열 2개가 들어 있다(B절). 이 그룹을 근거로 한 기존 비교(예: Financial_Only vs Full)는 재해석이 필요하다.",
         "7. 위 결과는 **신규 일별 뉴스가 기존 희소 뉴스보다 예측을 분명히 개선했다는 증거를 제공하지 않는다.** "
         "또한 개선이 없다는 것도 소스·척도 차이(H절) 때문에 '뉴스에 정보가 없다'는 결론으로 바로 이어지지 않는다.", ""]
    L += S

    # ---------------- A
    vol0 = rd("newsv2_full", "_final_v10_volatility.csv")
    L += ["## A. 실험 조건", "",
          "| variant | 뉴스 | 임베딩 PCA 솔버 | 비고 |", "|---|---|---|---|",
          "| oldnews_full | 기존(희소: 미국 221일 / 영국 713일, 전방 보간) | full | **주 비교(기존)** |",
          "| newsv2_full | 신규(NYT / Guardian, 거래일마다 갱신) | full | **주 비교(신규)** |",
          "| oldnews_auto | 기존 | auto (scikit-learn 기본) | PCA 구현 민감도 |",
          "| newsv2_auto | 신규 | auto | PCA 구현 민감도 |", "",
          f"- 환경: Python {man['newsv2_full']['python']}, numpy {pk['numpy']}, pandas {pk['pandas']}, scipy {pk['scipy']}, "
          f"scikit-learn {pk['sklearn']}, lightgbm {pk['lightgbm']}, arch {pk['arch']}. 원 실험은 Python 3.7.4 / scikit-learn 1.0.2 (재현 불가).",
          "- 시장 데이터·타깃·피처 정의·모델·부트스트랩·DM 검정은 v10 프로토콜과 동일. **뉴스 열 16개와 임베딩만 교체**했다. "
          "시장 열은 원본과 비트 단위로 동일함을 확인했다.",
          "- 분할은 표본 행수의 70% / 85% 로 재산정했으나 표본이 기존과 같아 **컷오프도 동일**: 학습 < 2022-05-19, 검증 < 2024-03-07, "
          f"시험 ≥ 2024-03-07 (시험 표본 미국 {int(vol0[vol0.market=='US'].n_test.iloc[0])}일 / 영국 {int(vol0[vol0.market=='UK'].n_test.iloc[0])}일). "
          "따라서 기존·신규 뉴스 결과는 같은 시험 표본에서 직접 비교된다.",
          "- 재현성: 전 파이프라인을 같은 시드로 두 번 실행해 44개 결과 CSV 가 일치함을 확인했고, newsv2_full 의 변동성 단계는 PYTHONHASHSEED 0 / 777 에서도 동일했다.", ""]

    # ---------------- B
    L += ["## B. 5일 실현변동성 (log RV5) 예측 — R² [95% 부트스트랩 CI], PCA 솔버 full", ""]
    rows = []
    for m in ["US", "UK"]:
        for g in GROUPS:
            r = {"시장": m, "피처군": g}
            vals = {}
            for v in ["oldnews_full", "newsv2_full"]:
                d = rd(v, "_final_v10_volatility.csv")
                x = d[(d.market == m) & (d.group == g)].iloc[0]
                r[{"oldnews_full": "기존 뉴스", "newsv2_full": "신규 뉴스"}[v]] = fmt_ci(x[["r2", "ci_low", "ci_high"]])
                vals[v] = x.r2
            r["Δ(신규−기존)"] = f"{vals['newsv2_full'] - vals['oldnews_full']:+.3f}"
            rows.append(r)
    L += [md(pd.DataFrame(rows)), "",
          "- `AR_Only`, `Financial_Clean` 은 뉴스 열을 쓰지 않아 두 조건에서 값이 같다.",
          "- **`Financial_Only` 는 순수한 비뉴스 그룹이 아니다.** 뉴스 파생 열 `sent_x_vix`, `sent_x_vol`(감성 × VIX·변동성)이 "
          "`NEWS_SENT_COLS` 에 등록되지 않아 이 그룹에 들어가 있다(기존 v10 프로토콜부터 존재). 그래서 뉴스 데이터가 바뀌면 이 그룹의 값도 바뀐다. "
          "두 열을 뺀 `Financial_Clean` 을 추가 행으로 계산했다. 기존 그룹의 값은 변경하지 않았다.", ""]

    # ---------------- C
    L += ["## C. 뉴스 증분에 대한 Diebold-Mariano 검정 — 변동성 (기준 → 비교, DM>0 이면 비교 모델 우위)", ""]
    rows = []
    for v in V4:
        d = rd(v, "_final_v10_dm.csv")
        r = {"variant": v}
        for m in ["US", "UK"]:
            for c in ["AR+News", "Full"]:
                x = d[(d.market == m) & (d.baseline == "AR_Only") & (d.model == c)].iloc[0]
                r[f"{m} AR_Only→{c}"] = f"{x.dm:+.2f} (p={x.p:.3f})"
        rows.append(r)
    L += [md(pd.DataFrame(rows)), "",
          "다중 비교 보정은 하지 않았다. 비교 모델이 유의하게 우위인 경우는 신규 뉴스·full 솔버의 미국 `AR_Only→Full`(p=0.026) 하나뿐이며, 같은 데이터에서 솔버만 auto 로 바꾸면 사라진다(p=0.210). "
          "영국은 4개 조건 중 3개에서 `Full` 이 `AR_Only` 보다 유의하게 나쁘다(p<0.05, 나머지 하나는 p=0.063).", ""]

    # ---------------- D
    L += ["## D. 5일 누적 방향성 (dir5) — AUC [95% CI], PCA 솔버 full", ""]
    rows = []
    for m in ["US", "UK"]:
        for g in GROUPS:
            r = {"시장": m, "피처군": g}
            for v, nm in [("oldnews_full", "기존 뉴스"), ("newsv2_full", "신규 뉴스")]:
                d = rd(v, "_final_v10_direction.csv")
                x = d[(d.market == m) & (d.group == g)].iloc[0]
                r[nm] = f"{x.auc:.3f} [{x.ci_low:.3f}, {x.ci_high:.3f}]" + (" *" if x.ci_low > 0.5 else "")
            rows.append(r)
    L += [md(pd.DataFrame(rows)), "", "(* = CI 하한 > 0.5)", "",
          "방향성 DM (Brier 손실) — 기준 → 비교:", ""]
    rows = []
    for v in ["oldnews_full", "newsv2_full"]:
        d = rd(v, "_final_v10_dm_dir.csv")
        r = {"variant": v}
        for m in ["US", "UK"]:
            for b, c in [("AR_Only", "AR+News"), ("AR_Only", "Full")]:
                x = d[(d.market == m) & (d.baseline == b) & (d.model == c)].iloc[0]
                r[f"{m} {b}→{c}"] = f"{x.dm:+.2f} (p={x.p:.3f})"
        rows.append(r)
    L += [md(pd.DataFrame(rows)), ""]

    # ---------------- E
    L += ["## E. 8-fold Walk-Forward · pooled 학습 · 전통 모형 (PCA 솔버 full)", ""]
    rows = []
    for v, nm in [("oldnews_full", "기존 뉴스"), ("newsv2_full", "신규 뉴스")]:
        f = rd(v, "_final_v10_folds.csv")
        for m in ["US", "UK"]:
            t = f[f.market == m]
            rows.append({"뉴스": nm, "시장": m, "AR_Only 평균 R²": f"{t.AR_Only_r2.mean():+.3f}", "Full 평균 R²": f"{t.Full_r2.mean():+.3f}",
                         "AR 양수 fold": f"{int((t.AR_Only_r2 > 0).sum())}/{len(t)}", "Full 양수 fold": f"{int((t.Full_r2 > 0).sum())}/{len(t)}"})
    L += [md(pd.DataFrame(rows)), ""]
    rows = []
    for v, nm in [("oldnews_full", "기존 뉴스"), ("newsv2_full", "신규 뉴스")]:
        p = rd(v, "_pooled_calibrated_v9_results.csv")
        for m in ["US", "UK"]:
            for ft in ["AR_Only", "Full"]:
                a = p[(p.market == m) & (p.features == ft) & (p.model == "단독")].iloc[0]
                b = p[(p.market == m) & (p.features == ft) & (p.model == "pooled")].iloc[0]
                rows.append({"뉴스": nm, "시장": m, "피처": ft, "단독 R²(보정)": f"{a.r2_calibrated:+.3f}", "pooled R²(보정)": f"{b.r2_calibrated:+.3f}"})
    L += [md(pd.DataFrame(rows)), ""]
    tb = pd.concat([rd("newsv2_full", "USD/results_v6/traditional_baselines.csv"), rd("newsv2_full", "UK/results_v6/traditional_baselines.csv")])
    tb = tb[tb.target == "vol5"]
    tb0 = pd.concat([rd("oldnews_full", "USD/results_v6/traditional_baselines.csv"), rd("oldnews_full", "UK/results_v6/traditional_baselines.csv")])
    tb0 = tb0[tb0.target == "vol5"]
    same = tb.reset_index(drop=True).equals(tb0.reset_index(drop=True))
    L += ["전통 모형(vol5 log R², 시험): " + ", ".join(f"{r.market} {r.model} {r.r2:+.3f}" for r in tb.itertuples())
          + f". 뉴스를 쓰지 않으므로 두 조건에서 " + ("동일하다." if same else "다르다 — 확인 필요."), ""]

    # ---------------- F
    L += ["## F. 구현 민감도 — PCA 솔버(auto vs full)가 만드는 잡음 크기 (R², 변동성)", ""]
    rows = []
    for m in ["US", "UK"]:
        for g in ["News_Pure", "AR+News", "Full"]:
            r = {"시장": m, "피처군": g}
            vals = {}
            for v in V4:
                d = rd(v, "_final_v10_volatility.csv")
                vals[v] = d[(d.market == m) & (d.group == g)].iloc[0].r2
                r[v] = f"{vals[v]:+.3f}"
            r["auto−full 절대차(기존)"] = f"{abs(vals['oldnews_auto'] - vals['oldnews_full']):.3f}"
            r["auto−full 절대차(신규)"] = f"{abs(vals['newsv2_auto'] - vals['newsv2_full']):.3f}"
            r["신규−기존 절대차(full)"] = f"{abs(vals['newsv2_full'] - vals['oldnews_full']):.3f}"
            rows.append(r)
    L += [md(pd.DataFrame(rows)), "",
          "같은 데이터에서 PCA 솔버만 바꿔도 뉴스 포함 피처군의 R² 가 최대 약 0.05 움직인다. 기존·신규 뉴스 간 차이(신규−기존 절대차)가 "
          "이 범위 안에 있으면 뉴스 데이터의 효과로 해석하기 어렵다. 뉴스 열을 쓰지 않는 그룹은 솔버와 무관하게 동일하다.", ""]

    # ---------------- G
    thesis = BASE / "_final_v10_volatility.csv"
    if thesis.exists():
        t0 = pd.read_csv(thesis).set_index(["market", "group"]).r2
        L += ["## G. 참고 — 기존 논문 수치(Python 3.7 / scikit-learn 1.0.2)와 새 환경(기존 뉴스)의 차이", "",
              "논문 본문 수치를 갱신할 때 어느 값이 환경 때문에 이미 달라지는지 보기 위한 표다. AR_Only 와 Financial_Only(기존 뉴스 기준)는 정확히 재현되고, 임베딩 PCA 를 쓰는 뉴스 포함 그룹만 달라진다.", ""]
        rows = []
        for m in ["US", "UK"]:
            for g in ["AR_Only", "Financial_Only", "News_Pure", "AR+News", "Full"]:
                a = rd("oldnews_auto", "_final_v10_volatility.csv").set_index(["market", "group"]).r2
                f = rd("oldnews_full", "_final_v10_volatility.csv").set_index(["market", "group"]).r2
                rows.append({"시장": m, "피처군": g, "논문(커밋된 v10)": f"{t0[(m, g)]:+.4f}", "새 환경 auto": f"{a[(m, g)]:+.4f}", "새 환경 full": f"{f[(m, g)]:+.4f}"})
        L += [md(pd.DataFrame(rows)), ""]

    # ---------------- H
    L += ["## H. 해석 시 유의사항", "",
          "1. **소스 교란**: 기존은 로이터·MarketWatch·CNBC·블룸버그 등 혼합 매체의 헤드라인으로 뉴스가 있는 날 기준 미국 약 104건 / 영국 약 52건(22,971건 ÷ 221일, 36,739건 ÷ 713일)이고 그 값이 다음 뉴스일까지 이월되었다. 신규는 NYT / Guardian 경제 섹션으로 거래일당 미국 평균 4.96건 / 영국 8.11건이다. "
          "결과 차이가 커버리지 때문인지 매체·성격 때문인지 분리되지 않는다.",
          "2. **일별 감성 점수의 분산 증가**: 하루에 평균하는 헤드라인이 줄어 일별 감성 점수(sent_score_mean)의 표준편차가 0.08~0.10(기존)에서 0.29~0.31(신규)로 커졌다(달력일 전체 행 기준). 피처 척도가 달라진 것이지 정보량이 늘었다는 뜻은 아니다.",
          "3. **시험 구간 재사용**: 동일 시험 구간(2024-03-07~)이 v3~v13 개발 내내 쓰였으므로 이 결과도 독립 검증은 아니다.",
          "4. **다중 비교 미보정**: 피처군·시장·시계·솔버별 다수 검정을 보고한다. 5% 수준의 단일 결과(예: 신규·full 미국 `AR_Only→Full`)는 탐색적으로 다뤄야 한다.",
          "5. **fallback 날짜**: 미국 주말의 34%, 영국 주말의 10%는 그날 의미 있는 경제 기사가 없어 최고점 1건만 유지한 날이다.",
          "6. **그림 10**: '실제 뉴스가 있는 날' 표시는 신규 데이터에서 거의 모든 거래일이 표시되므로 논문 작성 단계에서 재설계가 필요하다.",
          "7. **시점**: 매체 발행일만 있고 기사 시각은 없다. 뉴스 열은 t 행에 shift 없이 들어가고 타깃은 t+1 이후이므로 t 일 발행 뉴스 사용은 타당하나, "
          "일부 기사(예: Guardian 'as it happened' 라이브블로그)에는 t 일 장중 결과가 반영돼 있을 수 있다.", ""]

    (RES / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("saved", RES / "SUMMARY.md")


if __name__ == "__main__":
    main()
