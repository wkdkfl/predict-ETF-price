# -*- coding: utf-8 -*-
"""Add Q15-Q17 to the Q&A Word document."""
from pathlib import Path
from docx import Document
from docx.shared import Pt, RGBColor

BASE = Path(__file__).resolve().parent
doc = Document(BASE / "02_qa_preparation.docx")

new_qas = [
    {
        "q": "Q15. GARCH나 HAR-RV 같은 전통 계량경제 모형과는 비교했나요?",
        "a": (
            "A. 네, 실험에 포함했습니다. GARCH(1,1)은 US R²=-0.213, UK R²=-0.329로 "
            "Mean 예측보다도 낮은 성능을 보였고, GJR-GARCH 역시 음(-)의 R²를 기록했습니다. "
            "HAR-RV(Corsi, 2009)는 US R²=0.034로 전통 모형 중 가장 우수했지만, "
            "ML RandomForest의 R²=0.830과 비교하면 큰 차이가 납니다. "
            "GARCH의 낮은 성능은 Hansen & Lunde(2005)가 지적한 'GARCH의 out-of-sample 일별 예측 한계'와 일치하며, "
            "ML이 HAR 성분 외 다차원 피처를 활용하기 때문에 훨씬 높은 예측력을 보입니다."
        ),
    },
    {
        "q": "Q16. Optuna 하이퍼파라미터 튜닝 시 test 데이터 누출은 없나요?",
        "a": (
            "A. 없습니다. Optuna는 Train(70%)와 Val(15%) 데이터에서만 실행됩니다. "
            "objective 함수가 Val 성능(RMSE)을 최소화하며, Test(15%)는 튜닝 과정에 일절 노출되지 않습니다. "
            "Optuna로 최적 파라미터를 선정한 후, Train+Val 합산 데이터로 최종 모델을 학습하고 Test에서 평가합니다. "
            "이는 표준적인 nested evaluation 절차이며 test leakage가 없습니다."
        ),
    },
    {
        "q": "Q17. 챕터 4(결과)와 챕터 5(분석)의 내용이 겹치는 것 같은데요?",
        "a": (
            "A. 의도적인 구조 분리입니다. 4장은 수치와 표 중심으로 결과를 제시하고, "
            "5장은 이론적 해석·선행연구 비교·통계적 함의를 다룹니다. "
            "예를 들어 §4.5-4.7은 수익률 예측 R² 수치를 보고하고, §5.1-5.7은 그 수치가 왜 EMH와 일치하는지, "
            "다른 시장과 어떻게 다른지를 해석합니다. "
            "다만 일부 텍스트 중복이 있어 최종 제출 전 편집 예정입니다."
        ),
    },
]

for qa in new_qas:
    q_para = doc.add_paragraph()
    q_run = q_para.add_run(qa["q"])
    q_run.font.bold = True
    q_run.font.size = Pt(12)
    q_run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    a_para = doc.add_paragraph()
    a_run = a_para.add_run(qa["a"])
    a_run.font.size = Pt(11)
    doc.add_paragraph()

doc.save(BASE / "02_qa_preparation.docx")
print("Q&A updated: Q15, Q16, Q17 added.")
