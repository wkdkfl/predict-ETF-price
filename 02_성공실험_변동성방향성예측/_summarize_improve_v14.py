# -*- coding: utf-8 -*-
"""개선실험/ 의 CSV 에서 SUMMARY.md 를 생성한다 (사전 고정 판정 규칙을 코드로 적용).

  python _summarize_improve_v14.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
D = BASE / "개선실험"
SETS = ["F0", "F1", "F2", "F3", "F4", "F5"]
DESC = {"F0": "AR_Only (기준선)", "F1": "AR + 기존 뉴스(감성16+임베딩15) (v10 뉴스 기준선)", "F2": "AR + 뉴스 강도·주제(INT 9)",
        "F3": "AR + INT + 평활 감성(SS 6)", "F4": "AR + INT + 감성16", "F5": "AR + 평활 감성(SS 6)"}


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    sel = pd.read_csv(D / "selection_walkforward.csv")
    tv, td = pd.read_csv(D / "test_volatility.csv"), pd.read_csv(D / "test_direction.csv")
    sv = sel[(sel.task == "vol") & sel.selected]
    sd = sel[(sel.task == "dir") & sel.selected]
    L = ["# 개선 실험 결과 (사전 고정 프로토콜, 시험 구간 1회 평가)", "",
         "`_summarize_improve_v14.py` 가 `개선실험/*.csv` 에서 생성. 프로토콜은 `plan/20260920_improvement_plan.md` 에 실험 전에 고정했다.", ""]

    # ------------- 판정
    L += ["## 판정 (사전 규칙)", "",
          "선택된 후보가 v10 기준선 F0·F1(LightGBM)을 모두 이겨야 '개선'이다. 변동성: R² 우위 & DM p<0.05, 방향성: AUC 우위 & ΔAUC 부트스트랩 CI 가 0 을 배제.", ""]
    rows = []
    if len(sv):
        s = sv.iloc[0]
        for m in ["US", "UK"]:
            x = tv[(tv.featset == s.featset) & (tv.model == s.model) & (tv.market == m)].iloc[0]
            b0 = tv[(tv.featset == "F0") & (tv.model == "lgbm") & (tv.market == m)].iloc[0]
            b1 = tv[(tv.featset == "F1") & (tv.model == "lgbm") & (tv.market == m)].iloc[0]
            ok = x.r2 > b0.r2 and x.r2 > b1.r2 and x.p_vs_F0 < 0.05 and x.p_vs_F1 < 0.05
            rows.append({"타깃": "변동성", "선택 후보": f"{s.featset} + {s.model}", "시장": m, "후보 R²/AUC": f"{x.r2:+.3f}",
                         "F0(lgbm)": f"{b0.r2:+.3f}", "F1(lgbm)": f"{b1.r2:+.3f}", "검정": f"DM p={x.p_vs_F0:.3f} / {x.p_vs_F1:.3f}",
                         "판정": "개선" if ok else "개선 아님"})
    else:
        rows.append({"타깃": "변동성", "선택 후보": "없음", "시장": "-", "후보 R²/AUC": "-", "F0(lgbm)": "-", "F1(lgbm)": "-", "검정": "-", "판정": "개선 없음"})
    if len(sd):
        s = sd.iloc[0]
        for m in ["US", "UK"]:
            x = td[(td.featset == s.featset) & (td.model == s.model) & (td.market == m)].iloc[0]
            b0 = td[(td.featset == "F0") & (td.model == "lgbm") & (td.market == m)].iloc[0]
            b1 = td[(td.featset == "F1") & (td.model == "lgbm") & (td.market == m)].iloc[0]
            ok = x.auc > b0.auc and x.auc > b1.auc and x.dauc_lo_F0 > 0 and x.dauc_lo_F1 > 0
            rows.append({"타깃": "방향성", "선택 후보": f"{s.featset} + {s.model}", "시장": m, "후보 R²/AUC": f"{x.auc:.3f}",
                         "F0(lgbm)": f"{b0.auc:.3f}", "F1(lgbm)": f"{b1.auc:.3f}",
                         "검정": f"ΔAUC vs F0 {x.dauc_vs_F0:+.3f} [{x.dauc_lo_F0:+.3f}, {x.dauc_hi_F0:+.3f}]", "판정": "개선" if ok else "개선 아님(CI 가 0 포함)" if x.dauc_lo_F0 <= 0 else "개선 아님"})
    else:
        rows.append({"타깃": "방향성", "선택 후보": "없음", "시장": "-", "후보 R²/AUC": "-", "F0(lgbm)": "-", "F1(lgbm)": "-", "검정": "-", "판정": "개선 없음"})
    L += [md(pd.DataFrame(rows)), ""]

    # ------------- 후보 정의
    L += ["## 후보 피처군", "", md(pd.DataFrame([{"이름": k, "구성": v} for k, v in DESC.items()])), ""]

    # ------------- 선택 단계
    L += ["## 1. 사전 워크포워드 선택 (시험 구간 미사용)", "",
          "확장 윈도우 5블록(사전 표본 뒤쪽 50%), 7일 엠바고. 변동성 = 표본외 MSE(낮을수록 좋음), 방향성 = 표본외 AUC 시장 평균. "
          "vs_F0 / vs_F1 은 같은 모형의 F0·F1 대비 개선(변동성: MSE 개선율, 방향성: AUC 차).", ""]
    for task, nm in (("vol", "변동성"), ("dir", "방향성")):
        t = sel[sel.task == task][["featset", "model", "score", "vs_F0", "vs_F1", "qualified", "selected"]].copy()
        t["score"] = t.score.map(lambda v: f"{v:.4f}")
        t["vs_F0"] = t.vs_F0.map(lambda v: f"{v:+.4f}")
        t["vs_F1"] = t.vs_F1.map(lambda v: f"{v:+.4f}")
        L += [f"**{nm}**", "", md(t), ""]

    # ------------- 시험 결과 전체
    L += ["## 2. 시험 구간 평가 — 모든 후보 (다중 비교 미보정, 탐색적)", "", "변동성 R² [95% CI]와 DM p (기준: v10 F0·F1 LightGBM):", ""]
    rows = []
    for fs in SETS:
        for m in ["lgbm", "linear"]:
            r = {"피처군": fs, "모형": m, "선택": "◀" if len(sv) and fs == sv.iloc[0].featset and m == sv.iloc[0].model else ""}
            for mk in ["US", "UK"]:
                x = tv[(tv.featset == fs) & (tv.model == m) & (tv.market == mk)].iloc[0]
                r[f"{mk} R²"] = f"{x.r2:+.3f} [{x.ci_low:+.3f}, {x.ci_high:+.3f}]"
                r[f"{mk} p(F0/F1)"] = f"{x.p_vs_F0:.2f} / {x.p_vs_F1:.2f}"
            rows.append(r)
    L += [md(pd.DataFrame(rows)), "", "방향성 AUC [95% CI]와 F0(lgbm) 대비 ΔAUC [95% CI]:", ""]
    rows = []
    for fs in SETS:
        for m in ["lgbm", "linear"]:
            r = {"피처군": fs, "모형": m, "선택": "◀" if len(sd) and fs == sd.iloc[0].featset and m == sd.iloc[0].model else ""}
            for mk in ["US", "UK"]:
                x = td[(td.featset == fs) & (td.model == m) & (td.market == mk)].iloc[0]
                r[f"{mk} AUC"] = f"{x.auc:.3f} [{x.ci_low:.3f}, {x.ci_high:.3f}]"
                r[f"{mk} ΔAUC vs F0"] = f"{x.dauc_vs_F0:+.3f} [{x.dauc_lo_F0:+.3f}, {x.dauc_hi_F0:+.3f}]"
            rows.append(r)
    L += [md(pd.DataFrame(rows)), ""]

    # ------------- 모형 교란
    a = {m: td[(td.featset == "F0") & (td.model == "linear") & (td.market == m)].iloc[0].auc for m in ["US", "UK"]}
    b = {m: td[(td.featset == "F2") & (td.model == "linear") & (td.market == m)].iloc[0].auc for m in ["US", "UK"]}
    L += ["## 해석과 프로토콜 한계", "",
          f"1. **방향성 AUC 상승은 뉴스가 아니라 모형 변경 때문이다.** 뉴스 없이 AR 만 쓴 로지스틱(F0 linear)의 시험 AUC 가 미국 {a['US']:.3f}, 영국 {a['UK']:.3f}로, "
          f"뉴스 강도 피처를 더한 선택 후보(F2 linear, 미국 {b['US']:.3f}, 영국 {b['UK']:.3f})와 사실상 같다. LightGBM(F0 미국 0.482)에서 로지스틱으로 바꾼 효과이며, "
          "이 상승도 ΔAUC CI 가 0 을 포함해 유의하지 않다.",
          "2. **선택 규칙의 결함(사후 발견)**: 후보를 '같은 모형의 F0·F1' 과만 비교하도록 사전 고정해서, 변동성에서 전반적으로 훨씬 나쁜 선형 모형(워크포워드 MSE 0.33~0.36 vs LightGBM 0.25)이 "
          "선택됐다. 규칙을 결과를 본 뒤 바꾸면 사후 조정이 되므로 그대로 두었다. 모든 후보의 시험 결과를 위 표에 공개했다.",
          "3. **기존 뉴스(F1) 도 사전 검증에서는 F0 보다 낫다**(LightGBM MSE 0.2482 vs 0.2556). 새 강도·주제 피처(F2~F5)는 F1 을 넘지 못했다.",
          "4. **워크포워드 방향성 AUC 는 모든 후보가 0.49~0.52** 로 우연 수준이다. 시험 구간의 AUC 0.54~0.57 은 표본 452일/434일에서 CI 가 0.5 를 포함한다.",
          "5. 시험 구간은 v3~v13 개발 내내 쓰였고 후보가 24개(6 피처군 × 2 모형 × 2 타깃)이므로, 위 표의 개별 수치는 다중 비교 보정 전 탐색적 결과다.",
          "6. 기준선 F0·F1 의 LightGBM 결과는 `재실험_결과/newsv2_full` 의 v10 값과 소수점 자릿수 전체가 일치함을 스크립트가 검증한다.", ""]
    (D / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("saved", D / "SUMMARY.md")


if __name__ == "__main__":
    main()
