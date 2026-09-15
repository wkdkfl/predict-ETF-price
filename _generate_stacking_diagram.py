# -*- coding: utf-8 -*-
"""Stacking 앙상블 구조도 — 논문용 흑백/회색조, 큰 구조만 보이는 단순화 버전.

입력 피처 -> Level-1 베이스 학습기(XGBoost/LightGBM/RandomForest) -> Level-2 메타학습기
(Ridge=변동성 / Logistic=방향성) -> 최종 출력. 하이퍼파라미터·holdout 메커니즘 등 세부사항은
본문 표로 남기고, 그림에는 큰 구조만 표현.
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

WHITE      = '#FFFFFF'
LIGHT_GRAY = '#F2F2F2'
MID_GRAY   = '#D9D9D9'
DARK_GRAY  = '#595959'
INK        = '#1A1A1A'


def draw_box(ax, x, y, w, h, text, fontsize=11, bold=False, fill=WHITE,
             edge=INK, lw=1.5, style='round,pad=0.1'):
    box = FancyBboxPatch((x, y), w, h, boxstyle=style,
                          facecolor=fill, edgecolor=edge, linewidth=lw)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fontsize, weight='bold' if bold else 'normal',
            color=INK, linespacing=1.6)


def arrow(ax, x1, y1, x2, y2, lw=1.6, color=INK):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=16))


def section_tag(ax, x, y, text):
    ax.text(x, y, text, ha='left', va='bottom', fontsize=10, weight='bold',
            color=DARK_GRAY, style='italic')


fig, ax = plt.subplots(figsize=(11, 10.5))
ax.set_xlim(0, 11)
ax.axis('off')

X0, X1 = 0.4, 10.6
CX = (X0 + X1) / 2

# ============================================================ Title
ax.text(CX, 12.7, 'Stacking 앙상블 구조', ha='center', fontsize=18, weight='bold', color=INK)

# ============================================================ Input
h = 1.0
y = 11.5
draw_box(ax, X0, y, X1 - X0, h, '입력 피처  —  뉴스 텍스트 임베딩(FinBERT) + 금융지표', fontsize=12, fill=LIGHT_GRAY)
arrow(ax, CX, y, CX, y - 0.7)

# ============================================================ Level 1: base learners
y1 = y - 0.7 - 1.4
section_tag(ax, X0, y1 + 1.5, 'Level 1 — 베이스 학습기')
bw = (X1 - X0 - 0.6) / 3
xs = [X0, X0 + bw + 0.3, X0 + 2 * (bw + 0.3)]
for bx, name in zip(xs, ['XGBoost', 'LightGBM', 'Random Forest']):
    draw_box(ax, bx, y1, bw, 1.4, name, fontsize=13, bold=True, fill=LIGHT_GRAY, lw=1.8)
centers1 = [bx + bw / 2 for bx in xs]

y_bus = y1 - 0.45
for cx in centers1:
    ax.plot([cx, cx], [y1, y_bus], color=DARK_GRAY, lw=1.3)
ax.plot([centers1[0], centers1[-1]], [y_bus, y_bus], color=INK, lw=1.6)
arrow(ax, CX, y_bus, CX, y_bus - 0.55)

# ============================================================ Level 2: meta-learner
y2 = y_bus - 0.55 - 1.4
section_tag(ax, X0, y2 + 1.5, 'Level 2 — 메타학습기')
draw_box(ax, 1.1, y2, 3.9, 1.4, 'Ridge\n(변동성 회귀)', fontsize=13, bold=True, fill=WHITE, lw=1.8)
draw_box(ax, 5.9, y2, 3.9, 1.4, 'Logistic\n(방향성 분류)', fontsize=13, bold=True, fill=WHITE, lw=1.8)

xL, xR = 1.1 + 3.9 / 2, 5.9 + 3.9 / 2
y_bus2 = y2 - 0.45
ax.plot([xL, xL], [y2, y_bus2], color=DARK_GRAY, lw=1.3)
ax.plot([xR, xR], [y2, y_bus2], color=DARK_GRAY, lw=1.3)
arrow(ax, xL, y_bus2, xL, y_bus2 - 0.55, lw=1.4)
arrow(ax, xR, y_bus2, xR, y_bus2 - 0.55, lw=1.4)

# ============================================================ Output
y3 = y_bus2 - 0.55 - 1.3
draw_box(ax, 0.4, y3, 4.9, 1.3, '변동성 예측\nR² = 0.830(US) / 0.932(UK)',
         fontsize=11.5, bold=True, fill=MID_GRAY, lw=1.8)
draw_box(ax, 5.7, y3, 4.9, 1.3, '방향성 예측\nAUC = 0.749(US) / 0.844(UK)',
         fontsize=11.5, bold=True, fill=MID_GRAY, lw=1.8)

ax.set_ylim(y3 - 0.4, 13.1)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, 'fig_stacking_architecture.png')
plt.savefig(out_path, bbox_inches='tight', dpi=300, facecolor='white')
plt.close()
print('saved:', out_path)
