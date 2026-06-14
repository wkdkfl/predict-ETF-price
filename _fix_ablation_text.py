# -*- coding: utf-8 -*-
"""Fix ablation interpretation paragraphs P449 and P455 in thesis_draft_v20.docx"""
from pathlib import Path
from docx import Document

BASE = Path(__file__).resolve().parent

def replace_para(para, old, new):
    full = para.text
    if old not in full:
        return False
    new_text = full.replace(old, new)
    if para.runs:
        para.runs[0].text = new_text
        for run in para.runs[1:]:
            run.text = ""
    return True

doc = Document(BASE / "thesis_draft_v20.docx")
paras = doc.paragraphs

# ── P449: Ablation interpretation ───────────────────────────────
old_p449 = (
    "<표 10>는 본 연구의 가장 핵심적인 발견을 담는다. Ablation 분석은 XGBoost를 대표 모델로 선정하여 "
    "피처 세트별 변동성 예측 성능을 비교하였다. (i) Financial_Only(거시경제 금융 피처만)는 원래 스케일 R²가 "
    "US 0.052, UK −0.029로 변동성 정보를 거의 담지 못한다. (ii) News_Only(뉴스 감성+FinBE"
)
# Use a unique substring that definitely exists
old_p449_key = "(i) Financial_Only(거시경제 금융 피처만)는 원래 스케일 R²가"

new_p449 = (
    "<표 10>는 본 연구의 핵심 Ablation 결과를 담는다. 5개 피처군(AR_Only, Financial_Only, "
    "News_Pure, AR+News, Full)을 XGBoost로 비교하였다. "
    "(i) AR_Only(가격 기반 자기회귀 피처만)는 변동성 log-R²가 US 0.834, UK 0.939로 단독으로도 "
    "매우 높은 예측력을 보인다. 이는 변동성 클러스터링(volatility clustering)이 HAR-RV/GARCH 문헌과 "
    "일치하는 강한 자기회귀 패턴을 가짐을 반영한다. "
    "(ii) News_Pure(뉴스 감성+FinBERT 임베딩, AR 제외)만으로는 US log-R²=-0.218, UK -0.031로 "
    "변동성 예측에 실질적 기여가 없다. "
    "(iii) AR+News(=기존 'News_Only')가 AR_Only 수준(US 0.821, UK 0.927)을 유지하는 것은 "
    "AR 피처의 압도적 기여 때문이다. "
    "(iv) Financial_Only는 US -0.685, UK 0.004로 낮은 성능을 보인다. "
    "결론적으로, 변동성 예측에서는 가격 자기회귀 피처가 1차적 신호이며, "
    "뉴스 감성은 방향성 예측에서 유의미한 추가 기여(AR_Only AUC 0.724→Full 0.749 US, "
    "0.780→0.832 UK)를 제공한다."
)

# ── P455: Figure description fix ───────────────────────────────
old_p455_key = "News_Only는 Full에 근접하여, 단기 변동성의 예측 가능성이 주로 뉴스 관련 피처군(텍스트 임베딩+자기회귀)에서 비롯됨을 보인다."
new_p455_replacement = (
    "AR_Only가 Full과 유사한 R²를 달성하며, AR+News(구 'News_Only')도 AR_Only 수준을 유지한다. "
    "이는 변동성 예측에서 가격 자기회귀 피처(rolling_absret, leverage 등)가 1차적 신호임을 나타낸다. "
    "News_Pure(AR 제외 순수 뉴스)는 변동성 예측력이 낮다."
)

fixed_449 = False
fixed_455 = False

for i, p in enumerate(paras):
    t = p.text
    if old_p449_key in t and not fixed_449:
        if p.runs:
            p.runs[0].text = new_p449
            for run in p.runs[1:]:
                run.text = ""
        print(f"P{i}: ablation interpretation updated")
        fixed_449 = True

    if old_p455_key in t and not fixed_455:
        if p.runs:
            new_text = t.replace(old_p455_key, new_p455_replacement)
            p.runs[0].text = new_text
            for run in p.runs[1:]:
                run.text = ""
        print(f"P{i}: figure description updated")
        fixed_455 = True

if not fixed_449:
    print("WARNING: P449 key not found — manual edit needed")
if not fixed_455:
    print("WARNING: P455 key not found — manual edit needed")

doc.save(BASE / "thesis_draft_v20.docx")
print("Saved thesis_draft_v20.docx")
