"""
Generate high-quality thesis diagrams using matplotlib.
1. Research Pipeline Flowchart (그림 2 replacement)
2. Stacking Ensemble Architecture (그림 3 replacement)
3. Walk-Forward Validation (그림 4 replacement)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

os.chdir(r"C:\Users\a00548169\OneDrive - ONEVIRTUALOFFICE\Desktop\자기주도적\ETFpricepredictmodel\data\2014-2025")

# Set Korean font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

# ============================================================
# DIAGRAM 1: Research Pipeline
# ============================================================
def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    data_color = '#E3F2FD'    # light blue
    feat_color = '#FFF3E0'    # light orange
    model_color = '#E8F5E9'   # light green
    eval_color = '#FCE4EC'    # light pink
    result_color = '#F3E5F5'  # light purple
    border_color = '#37474F'
    arrow_color = '#546E7A'
    
    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x, y), w, h, 
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor=border_color, linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, weight=weight, wrap=True)
    
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    
    # Title
    ax.text(7, 7.6, '연구 파이프라인 흐름도', ha='center', fontsize=14, weight='bold')
    
    # Row 1: Data Collection
    draw_box(0.5, 6.2, 2.5, 1.0, '뉴스 헤드라인 수집\n(Reuters, Bloomberg\nCNBC, Guardian)', data_color, 8)
    draw_box(3.5, 6.2, 2.5, 1.0, 'ETF 가격 데이터\n(yfinance)\nSPY / FTSE 100', data_color, 8)
    draw_box(6.5, 6.2, 2.5, 1.0, '금융지표 수집\n(VIX, Oil, Bond\n거시경제 지표)', data_color, 8)
    
    # Section label
    ax.text(13.5, 6.7, '데이터 수집\n(2014-2025)', ha='center', fontsize=8, style='italic', color='gray')
    
    # Arrows down
    draw_arrow(1.75, 6.2, 1.75, 5.5)
    draw_arrow(4.75, 6.2, 4.75, 5.5)
    draw_arrow(7.75, 6.2, 7.75, 5.5)
    
    # Row 2: Feature Engineering
    draw_box(0.5, 4.4, 2.5, 1.0, 'FinBERT 다중 헤드라인\n768-dim 임베딩 집계\n+ TF-IDF (500-dim)', feat_color, 8)
    draw_box(3.5, 4.4, 2.5, 1.0, '수익률 산출\n로그 변동성 타깃\n방향성 레이블', feat_color, 8)
    draw_box(6.5, 4.4, 2.5, 1.0, 'HAR 피처, RSI\nLeverage Effect\nVIX Dynamics', feat_color, 8)
    
    ax.text(13.5, 4.9, '피처 엔지니어링\n(t-1 시차 적용)', ha='center', fontsize=8, style='italic', color='gray')
    
    # Merge arrows to center
    draw_arrow(1.75, 4.4, 4.75, 3.8)
    draw_arrow(4.75, 4.4, 4.75, 3.8)
    draw_arrow(7.75, 4.4, 4.75, 3.8)
    
    # Row 3: Models (center)
    draw_box(2.0, 2.7, 5.5, 1.0, 
             'Stacking 앙상블 프레임워크\nXGBoost + LightGBM + RandomForest → Ridge / Logistic 메타학습기',
             model_color, 9, bold=True)
    
    # Side box: Optuna
    draw_box(8.5, 2.7, 2.5, 1.0, 'Optuna (100 trials)\n하이퍼파라미터\n자동 최적화', model_color, 8)
    draw_arrow(8.5, 3.2, 7.5, 3.2)
    
    ax.text(13.5, 3.2, '모델 학습\n(Optuna 최적화)', ha='center', fontsize=8, style='italic', color='gray')
    
    # Arrow down
    draw_arrow(4.75, 2.7, 4.75, 2.1)
    
    # Row 4: Evaluation
    draw_box(1.0, 1.0, 3.0, 1.0, '5-Fold Walk-Forward\nValidation\n(확장 윈도우)', eval_color, 9)
    draw_box(4.5, 1.0, 3.0, 1.0, 'Bootstrap CI (5,000회)\nDiebold-Mariano 검정\nAblation Study', eval_color, 9)
    
    ax.text(13.5, 1.5, '평가 & 검증', ha='center', fontsize=8, style='italic', color='gray')
    
    # Final arrow
    draw_arrow(3.0, 1.0, 3.0, 0.4)
    draw_arrow(6.0, 1.0, 6.0, 0.4)
    
    # Row 5: Results
    draw_box(1.5, -0.5, 6.0, 0.8,
             '변동성 R² = 0.830 (US) / 0.932 (UK)  |  방향성 AUC = 0.749 (US) / 0.844 (UK)',
             result_color, 10, bold=True)
    
    # Dual market labels
    draw_box(9.5, 4.4, 1.8, 1.0, 'US (SPY)\n미국 시장', '#BBDEFB', 8, bold=True)
    draw_box(11.5, 4.4, 1.8, 1.0, 'UK (FTSE)\n영국 시장', '#BBDEFB', 8, bold=True)
    ax.text(10.4, 5.2, '교차 시장 비교', ha='center', fontsize=8, weight='bold')
    draw_arrow(10.4, 4.4, 10.4, 3.8)
    draw_arrow(12.4, 4.4, 12.4, 3.8)
    
    plt.tight_layout()
    plt.savefig('fig_pipeline.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Pipeline diagram saved: fig_pipeline.png")

# ============================================================
# DIAGRAM 2: Stacking Ensemble Architecture
# ============================================================
def draw_stacking():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    colors = {
        'input': '#E3F2FD',
        'base': '#C8E6C9',
        'meta': '#FFE0B2',
        'output': '#F3E5F5',
        'border': '#37474F'
    }
    
    def draw_box(x, y, w, h, text, color, fontsize=9, bold=False):
        box = FancyBboxPatch((x, y), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor=colors['border'], linewidth=1.5)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, weight=weight)
    
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='#546E7A', lw=1.5))
    
    # Title
    ax.text(6, 6.7, 'Stacking 앙상블 구조', ha='center', fontsize=14, weight='bold')
    
    # Input layer
    draw_box(0.5, 5.5, 4.5, 0.8, '입력 피처: FinBERT 임베딩 + 금융지표 + HAR 피처', colors['input'], 9, True)
    draw_box(5.5, 5.5, 5.5, 0.8, '뉴스 감성(pos/neg/neu) + VIX dynamics + leverage + RSI', colors['input'], 8)
    
    # Arrows to base learners
    for x_target in [1.5, 5.0, 8.5]:
        draw_arrow(2.75, 5.5, x_target, 4.8)
    
    # Level-1: Base Learners
    ax.text(0.3, 4.2, 'Level-1\n(Base)', ha='center', fontsize=8, weight='bold', color='#2E7D32')
    
    draw_box(0.5, 3.7, 2.5, 1.0, 'XGBoost\n(depth=6, lr=0.03\nn_est=500)', colors['base'], 8)
    draw_box(3.5, 3.7, 2.5, 1.0, 'LightGBM\n(depth=7, lr=0.05\nn_est=400)', colors['base'], 8)
    draw_box(6.5, 3.7, 2.5, 1.0, 'Random Forest\n(depth=15\nn_est=100)', colors['base'], 8)
    
    # CV predictions
    ax.text(1.75, 3.5, 'CV 예측₁', ha='center', fontsize=7, style='italic')
    ax.text(4.75, 3.5, 'CV 예측₂', ha='center', fontsize=7, style='italic')
    ax.text(7.75, 3.5, 'CV 예측₃', ha='center', fontsize=7, style='italic')
    
    # Arrows to meta-learner
    for x_source in [1.75, 4.75, 7.75]:
        draw_arrow(x_source, 3.7, 4.75, 2.8)
    
    # Level-2: Meta-Learner
    ax.text(0.3, 2.2, 'Level-2\n(Meta)', ha='center', fontsize=8, weight='bold', color='#E65100')
    
    # Dual path: regression and classification
    draw_box(2.0, 1.8, 2.5, 0.9, '변동성 예측\nRidge Regression\n(α = auto)', colors['meta'], 8, True)
    draw_box(5.5, 1.8, 2.5, 0.9, '방향성 예측\nLogistic Regression\n(C = auto)', colors['meta'], 8, True)
    
    draw_arrow(4.75, 2.8, 3.25, 2.7)
    draw_arrow(4.75, 2.8, 6.75, 2.7)
    
    # Outputs
    draw_arrow(3.25, 1.8, 3.25, 1.2)
    draw_arrow(6.75, 1.8, 6.75, 1.2)
    
    draw_box(1.5, 0.3, 3.5, 0.8, 'log|r_{t+1}| 예측\nR² = 0.830 (US) / 0.932 (UK)', colors['output'], 9, True)
    draw_box(5.0, 0.3, 3.5, 0.8, 'sign(r_{t+1}) 예측\nAUC = 0.749 (US) / 0.844 (UK)', colors['output'], 9, True)
    
    # Optuna box
    draw_box(9.5, 3.5, 2.0, 1.4, 'Optuna\nTPE 알고리즘\n100 trials\n3-fold TS CV', '#FFECB3', 8, True)
    draw_arrow(9.5, 4.2, 9.0, 4.2)
    
    plt.tight_layout()
    plt.savefig('fig_stacking.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Stacking diagram saved: fig_stacking.png")

# ============================================================
# DIAGRAM 3: Walk-Forward Validation
# ============================================================
def draw_walkforward():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')
    
    # Title
    ax.text(6, 5.2, '5-Fold Walk-Forward Validation (확장 윈도우)', ha='center', fontsize=13, weight='bold')
    
    # Timeline
    years = ['2014', '2016', '2018', '2020', '2022', '2024', '2025']
    positions = [0.5, 2.4, 4.3, 6.2, 8.1, 10.0, 11.5]
    
    # Draw timeline
    ax.plot([0.5, 11.5], [0.2, 0.2], color='gray', lw=1)
    for x, yr in zip(positions, years):
        ax.plot(x, 0.2, 'o', color='gray', markersize=4)
        ax.text(x, -0.1, yr, ha='center', fontsize=8, color='gray')
    
    train_color = '#4FC3F7'
    test_color = '#FF7043'
    
    fold_configs = [
        # (train_start, train_end, test_start, test_end, fold_num)
        (0.5, 4.3, 4.3, 5.7, 'Fold 1'),
        (0.5, 5.7, 5.7, 7.0, 'Fold 2'),
        (0.5, 7.0, 7.0, 8.5, 'Fold 3'),
        (0.5, 8.5, 8.5, 10.0, 'Fold 4'),
        (0.5, 10.0, 10.0, 11.5, 'Fold 5'),
    ]
    
    for i, (ts, te, vs, ve, label) in enumerate(fold_configs):
        y = 4.2 - i * 0.8
        h = 0.5
        
        # Train bar
        train_rect = plt.Rectangle((ts, y), te - ts, h,
                                    facecolor=train_color, edgecolor='white', alpha=0.8)
        ax.add_patch(train_rect)
        ax.text((ts + te)/2, y + h/2, f'Train ({int((te-ts)*100/(11))}%)',
                ha='center', va='center', fontsize=7, color='white', weight='bold')
        
        # Test bar
        test_rect = plt.Rectangle((vs, y), ve - vs, h,
                                   facecolor=test_color, edgecolor='white', alpha=0.8)
        ax.add_patch(test_rect)
        ax.text((vs + ve)/2, y + h/2, 'Test',
                ha='center', va='center', fontsize=7, color='white', weight='bold')
        
        # Fold label
        ax.text(11.8, y + h/2, label, ha='left', va='center', fontsize=8, weight='bold')
    
    # Legend
    train_patch = mpatches.Patch(color=train_color, label='학습 데이터 (확장)')
    test_patch = mpatches.Patch(color=test_color, label='시험 데이터 (고정 크기)')
    ax.legend(handles=[train_patch, test_patch], loc='lower left',
             fontsize=9, framealpha=0.9)
    
    # Annotation
    ax.text(6, -0.4, '* 최소 학습 비율 40% → 첫 fold부터 충분한 데이터 확보',
            ha='center', fontsize=8, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig('fig_walkforward.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Walk-Forward diagram saved: fig_walkforward.png")

# Run all
draw_pipeline()
draw_stacking()
draw_walkforward()
print("\nAll 3 diagrams generated successfully!")
