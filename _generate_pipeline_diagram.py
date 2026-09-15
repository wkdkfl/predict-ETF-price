# -*- coding: utf-8 -*-
"""연구 파이프라인 전체 흐름도 — 논문용 흑백/회색조 스타일.

데이터 수집 -> 전처리/피처엔지니어링 -> (예비실험: 수익률 예측, 실패) / (본실험: 변동성·방향성 예측, 성공)
-> 평가·검증 -> 결론. 컬러풀한 프레젠테이션용이 아니라 학술논문 삽입용 흑백 도식.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

OUT_DIR = r"C:\Users\a00548169\OneDrive - ONEVIRTUALOFFICE\Desktop\자기주도적\ETFpricepredictmodel\data\2014-2025\그림"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------------- palette
WHITE      = '#FFFFFF'
LIGHT_GRAY = '#F2F2F2'
MID_GRAY   = '#D9D9D9'
DARK_GRAY  = '#595959'
INK        = '#1A1A1A'


def draw_box(ax, x, y, w, h, text, fontsize=9, bold=False, fill=WHITE,
             edge=INK, lw=1.3, ls='solid', style='round,pad=0.08'):
    box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                          facecolor=fill, edgecolor=edge, linewidth=lw, linestyle=ls)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, weight='bold' if bold else 'normal',
            color=INK, linespacing=1.5)
    return box


def arrow(ax, x1, y1, x2, y2, lw=1.3, ls='solid', color=INK):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, linestyle=ls,
                                 mutation_scale=14, shrinkA=0, shrinkB=0))


def section_tag(ax, x, y, text):
    ax.text(x, y, text, ha='left', va='center', fontsize=8.5, weight='bold',
            color=DARK_GRAY, style='italic')


fig, ax = plt.subplots(figsize=(13, 13.0))
ax.set_xlim(0, 13)
ax.set_ylim(4.7, 20.6)
ax.axis('off')

# ============================================================ Title
ax.text(6.5, 20.0, '연구 파이프라인 전체 흐름도', ha='center', fontsize=16, weight='bold', color=INK)
ax.text(6.5, 19.5, 'Research Pipeline Overview', ha='center', fontsize=10, color=DARK_GRAY, style='italic')

# ============================================================ Row 1: Data collection
y1 = 17.6
draw_box(ax, 0.3, y1, 3.7, 1.35,
         '뉴스 헤드라인 수집\nReuters·Bloomberg·CNBC·Guardian\n미국 22,971건 · 영국 36,739건',
         fontsize=8.3, fill=LIGHT_GRAY)
draw_box(ax, 4.4, y1, 3.7, 1.35,
         'ETF 가격 데이터\nyfinance\nSPY(US) · FTSE100 ETF(UK)',
         fontsize=8.3, fill=LIGHT_GRAY)
draw_box(ax, 8.5, y1, 4.2, 1.35,
         '금융지표 수집\nVIX·Oil·Bond·환율·섹터ETF 등\n40여 종 (2014–2024)',
         fontsize=8.3, fill=LIGHT_GRAY)
section_tag(ax, 0.3, y1 + 1.55, '① 데이터 수집 (2014–2024, 미국·영국)')

for cx in (2.15, 6.25, 10.6):
    arrow(ax, cx, y1, cx, y1 - 0.55)

# ============================================================ Row 2: Preprocessing / feature engineering
y2 = 15.85
draw_box(ax, 0.3, y2, 6.0, 1.3,
         '텍스트 표현\nTF-IDF(500dim) · FinBERT 다중헤드라인 집계(768dim + 감성통계 16종)',
         fontsize=8.3, fill=LIGHT_GRAY)
draw_box(ax, 6.7, y2, 6.0, 1.3,
         '금융 피처 엔지니어링\nHAR형 변동성 성분 · leverage effect · RSI · VIX dynamics\n(전 외생 피처 t-1 시차 적용)',
         fontsize=8.1, fill=LIGHT_GRAY)
section_tag(ax, 0.3, y2 + 1.5, '② 전처리 & 피처 엔지니어링 (strict t-1 forecasting)')

arrow(ax, 3.3, y2, 3.3, y2 - 0.45)
arrow(ax, 9.7, y2, 9.7, y2 - 0.45)
ax.plot([3.3, 6.5, 9.7], [y2 - 0.45, y2 - 0.45, y2 - 0.45], color=INK, lw=1.3)
arrow(ax, 6.5, y2 - 0.45, 6.5, y2 - 0.85)

# ============================================================ Branch header
yb = 14.2
draw_box(ax, 4.7, yb, 3.6, 0.55, '예측 타깃 분기', fontsize=9.5, bold=True,
         fill=MID_GRAY, edge=INK, style='round,pad=0.06')
arrow(ax, 5.6, yb, 2.6, yb - 0.55)
arrow(ax, 7.4, yb, 10.4, yb - 0.55)

# ============================================================ Left branch: preliminary (failed)
lx, lw_ = 0.3, 5.0
section_tag(ax, lx, yb - 0.85, '③-A 예비 실험 (Preliminary)')
draw_box(ax, lx, yb - 2.15, lw_, 1.15,
         '타깃: 일별 수익률 수준 r_t\nTF-IDF / FinBERT × Baseline·Tree·LSTM·\nTransformer·Ensemble (21개 모델)',
         fontsize=8.0, fill=WHITE)
arrow(ax, lx + lw_ / 2, yb - 2.15, lx + lw_ / 2, yb - 2.7)
draw_box(ax, lx, yb - 3.9, lw_, 1.15,
         '결과: 21개 모델 모두 R² < 0.03\nNaive Mean 대비 유의한 개선 없음\n(EMH와 일치)',
         fontsize=8.0, fill=LIGHT_GRAY, edge=DARK_GRAY, ls='dashed')
draw_box(ax, lx + 0.9, yb - 4.75, lw_ - 1.8, 0.55, '[X] 실패 → 타깃 전환의 근거',
         fontsize=8.3, bold=True, fill=WHITE, edge=INK)

# ============================================================ Right branch: main (success)
rx, rw_ = 7.7, 5.0
section_tag(ax, rx, yb - 0.85, '③-B 본 실험 (Main)')
draw_box(ax, rx, yb - 2.15, rw_, 1.15,
         '타깃 전환: 로그변동성 |r_{t+1}| · 방향성 sign(r_{t+1})\nStacking(XGBoost+LightGBM+RF → Ridge/Logistic)\n+ Optuna 100 trials 베이지안 탐색',
         fontsize=7.8, fill=WHITE, lw=1.6)
arrow(ax, rx + rw_ / 2, yb - 2.15, rx + rw_ / 2, yb - 2.7)
draw_box(ax, rx, yb - 3.9, rw_, 1.15,
         '결과: 변동성 R²=0.830(US)/0.932(UK)\n방향성 AUC=0.749(US)/0.844(UK)\n뉴스 피처 방향성 증분기여 유의(p<0.01)',
         fontsize=7.8, fill=LIGHT_GRAY, edge=INK, lw=1.6)
draw_box(ax, rx + 0.7, yb - 4.75, rw_ - 1.4, 0.55, '[O] 유의한 예측력 확인 — 논문 핵심 성과',
         fontsize=8.3, bold=True, fill=MID_GRAY, edge=INK, lw=1.6)

# converge: clean elbow/bus connector (no diagonal crossing), built top-down
y_label_bottom = yb - 4.75           # bottom edge of the [X]/[O] label boxes
y_bus = y_label_bottom - 0.35        # merge bus line just under the two boxes
xL = lx + lw_ / 2                    # 2.8  (center of left branch)
xR = rx + rw_ / 2                    # 10.2 (center of right branch)
ax.plot([xL, xL], [y_label_bottom, y_bus], color=DARK_GRAY, lw=1.3, ls='dashed')
ax.plot([xR, xR], [y_label_bottom, y_bus], color=INK, lw=1.3)
ax.plot([xL, xR], [y_bus, y_bus], color=INK, lw=1.3)

y_fan = y_bus - 0.7                  # fan-out line feeding the 4 evaluation boxes
ax.plot([6.5, 6.5], [y_bus, y_fan], color=INK, lw=1.3)

# ============================================================ Evaluation row
eval_h = 1.15
ye = y_fan - 0.55 - eval_h           # top of eval boxes sits 0.55 below the fan line
section_tag(ax, 0.3, y_fan + 0.35, '④ 평가 & 검증 — 본 실험 결과에 대해 수행')
boxes_eval = [
    ('5-Fold\nWalk-Forward\nValidation', 0.3),
    ('Block Bootstrap\n95% CI\n(5,000회)', 3.55),
    ('Diebold–Mariano\n검정\n(Full vs Ablation)', 6.8),
    ('피처군 Ablation\n(AR_Only/News_Only\n/Full)', 10.05),
]
centers = [bx + 2.65 / 2 for _, bx in boxes_eval]
ax.plot([min(centers), max(centers)], [y_fan, y_fan], color=INK, lw=1.3)
for (label, bx), cx in zip(boxes_eval, centers):
    draw_box(ax, bx, ye, 2.65, eval_h, label, fontsize=7.8, fill=LIGHT_GRAY)
    arrow(ax, cx, y_fan, cx, ye + eval_h, lw=1.0, color=DARK_GRAY)

# ============================================================ Final conclusion
yf = ye - 1.55
arrow(ax, 6.5, ye, 6.5, yf + 1.15, lw=1.3)
draw_box(ax, 0.6, yf, 11.8, 1.15,
         '결론: 미국·영국 교차시장 일반성 검증 · 뉴스 피처의 방향성 증분 기여 규명 ·\n'
         '미래정보편향(look-ahead bias) 크기 정량화 (ΔR² ~ 0.62–0.66)',
         fontsize=9.3, bold=True, fill=MID_GRAY, edge=INK, lw=1.6)

# ============================================================ side note: applied to both markets
ax.annotate('', xy=(12.85, yb - 5.0), xytext=(12.85, y1 + 1.5),
            arrowprops=dict(arrowstyle='-', color=DARK_GRAY, lw=1.0))
ax.text(12.95, (y1 + 1.5 + yb - 5.0) / 2, '전 과정 미국(SPY)·영국(FTSE100 ETF)에\n동일 파이프라인 적용',
        ha='left', va='center', fontsize=7.8, color=DARK_GRAY, rotation=90, style='italic')

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'fig_research_pipeline.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print('saved:', out_path)
