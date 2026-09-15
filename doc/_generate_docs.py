# -*- coding: utf-8 -*-
"""Generate presentation PPT and Q&A Word document for ETF prediction paper."""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parent   # this IS the doc/ folder
DOC_DIR = BASE

PPT_PATH  = DOC_DIR / "01_presentation_slides.pptx"
DOCX_PATH = DOC_DIR / "02_qa_preparation.docx"


# ═══════════════════════════════════════════════════════════════════════
# 1. PPT — 발표 슬라이드
# ═══════════════════════════════════════════════════════════════════════
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# Brand colors
C_DARK   = RGBColor(0x1A, 0x1A, 0x2E)   # dark navy (background)
C_ACCENT = RGBColor(0x16, 0x21, 0x3E)   # slide background
C_TEAL   = RGBColor(0x0F, 0x3D, 0x6E)   # header bar
C_GOLD   = RGBColor(0xE8, 0xB8, 0x30)   # accent / highlight
C_WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT  = RGBColor(0xCC, 0xDD, 0xEE)   # body text
C_GRAY   = RGBColor(0xAA, 0xBB, 0xCC)   # sub-text

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

blank_layout = prs.slide_layouts[6]   # completely blank


def set_bg(slide, color: RGBColor):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_textbox(slide, text, left, top, width, height,
                font_size=18, bold=False, color=C_WHITE,
                align=PP_ALIGN.LEFT, wrap=True):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size  = Pt(font_size)
    run.font.bold  = bold
    run.font.color.rgb = color
    return txb


def add_header_bar(slide, title_text, subtitle_text=""):
    """Dark teal header bar with gold title."""
    from pptx.util import Inches, Pt
    bar = slide.shapes.add_shape(
        1,   # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0),
        SLIDE_W, Inches(1.3)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_TEAL
    bar.line.fill.background()

    add_textbox(slide, title_text,
                Inches(0.4), Inches(0.1),
                Inches(12), Inches(0.7),
                font_size=28, bold=True, color=C_GOLD,
                align=PP_ALIGN.LEFT)
    if subtitle_text:
        add_textbox(slide, subtitle_text,
                    Inches(0.4), Inches(0.8),
                    Inches(12), Inches(0.45),
                    font_size=15, bold=False, color=C_LIGHT,
                    align=PP_ALIGN.LEFT)


def add_bullet_block(slide, items, left, top, width, height,
                     font_size=16, color=C_WHITE, bullet="▸ "):
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf  = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if first:
            p = tf.paragraphs[0]; first = False
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        run = p.add_run()
        run.text = bullet + item
        run.font.size  = Pt(font_size)
        run.font.color.rgb = color


def add_note(slide, note_text):
    notes_slide = slide.notes_slide
    tf = notes_slide.notes_text_frame
    tf.text = note_text


def add_divider(slide, y_pos):
    line = slide.shapes.add_shape(
        1,
        Inches(0.4), y_pos,
        Inches(12.5), Pt(1.5)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = C_GOLD
    line.line.fill.background()


# ───────────────────────────────────────────────
# Slide data
# ───────────────────────────────────────────────

SLIDES = [
    # (slide_num, title, subtitle, bullets_left, bullets_right, notes)
    {
        "title": "뉴스 감성 분석과 기계학습을 활용한\nETF 변동성·방향성 예측",
        "subtitle": "논문 발표 | 2025",
        "type": "cover",
        "bullets": [
            "대상 ETF: US (QQQ, SOXX) / UK (ISF.L · FTSE100)",
            "데이터 기간: 2013-12-01 ~ 2024-12-31 (약 11년)",
            "예측 과제 1 — 회귀: 다음 날 실현 변동성 |log(P_{t+1}/P_t)|",
            "예측 과제 2 — 분류: 다음 날 수익률 방향 (상승/하락)",
        ],
        "notes": "연구의 두 가지 핵심 과제를 소개합니다. 변동성 예측과 방향성 예측이며, 미국과 영국 두 시장을 동시에 분석합니다.",
    },
    {
        "title": "데이터 수집 — 시장 데이터",
        "subtitle": "파일: _download_extra_features.py  |  출처: Yahoo Finance (yfinance)",
        "type": "content",
        "bullets_l": [
            "왜 yfinance?",
            "  • Bloomberg 대비 무료·API 키 불필요 → 재현 가능",
            "  • 조정 종가(adjusted close) 자동 제공",
            "  • ETF·채권·지수·FX 단일 API로 수집",
            "",
            "US 21개 / UK 22개 심볼 수집",
        ],
        "bullets_r": [
            "카테고리별 심볼 선택 이유",
            "  SPY/IWM  → 시장 리스크 레짐",
            "  TLT/HYG  → 금리·크레딧 스프레드",
            "  ^VIX9D   → 단기 변동성 레짐",
            "  ^N225    → 아시아 야간 신호",
            "  ETH-USD  → Risk-on/off 대리변수",
        ],
        "notes": "yfinance 선택 이유 세 가지: 재현 가능성, 조정 종가, 자산 클래스 다양성. 각 심볼은 경제적 의미를 기반으로 선택했습니다.",
    },
    {
        "title": "데이터 수집 — 뉴스 헤드라인 & 데이터 누출 방지",
        "subtitle": "파일: *_headline_crawling_code.ipynb  |  US 23,000개 · UK 36,772개",
        "type": "content",
        "bullets_l": [
            "수집 규모",
            "  • US_financial_news.csv : 23,000 헤드라인",
            "  • UK_financial_news.csv : 36,772 헤드라인",
            "",
            "⚠ 핵심 전처리: 데이터 누출 방지",
            "  → 'market close summary' 헤드라인 제거",
        ],
        "bullets_r": [
            "필터 대상 패턴 예시",
            "  • 'closed at'",
            "  • 'stocks close higher'",
            "  • 'ended the day'",
            "  • 'after the bell'",
            "",
            "이유: 당일 종가 반영 → 미래 정보 누출",
        ],
        "notes": "시장 마감 요약 헤드라인은 당일 가격을 직접 반영하므로, 그대로 학습에 사용하면 미래 정보를 과거에서 본 것과 같은 데이터 누출이 발생합니다.",
    },
    {
        "title": "FinBERT 감성 분석",
        "subtitle": "파일: _build_multi_headline_features.py  |  ProsusAI/finbert (금융 뉴스 47,000개 파인튜닝)",
        "type": "content",
        "bullets_l": [
            "왜 FinBERT인가?",
            "",
            "  TF-IDF    → 문맥 무시, 단어 빈도만",
            "  VADER     → 소셜미디어 훈련, 금융 어휘 부재",
            "  GPT-4 API → 재현 불가, 비용, 프라이버시",
            "  RoBERTa   → 금융 도메인 미파인튜닝",
            "",
            "  ✅ FinBERT → 로컬·무료·완전 재현 가능",
        ],
        "bullets_r": [
            "출력 (단일 forward pass)",
            "  • [pos, neg, neu] 확률 3개",
            "  • [CLS] 768-dim 임베딩",
            "",
            "일별 집계 → 12개 감성 피처",
            "  sent_score_mean, sent_score_ma5/20",
            "  sent_surprise, sent_vol_5d ...",
            "",
            "임베딩: PCA 768→50 차원",
        ],
        "notes": "FinBERT는 금융 뉴스 전용 BERT 파인튜닝 모델로 로컬에서 실행 가능합니다. 단일 forward pass로 분류 확률과 CLS 임베딩을 동시에 추출해 효율적입니다.",
    },
    {
        "title": "특성 공학 (Feature Engineering)",
        "subtitle": "파일: _run_enhanced_models_v3.py  |  총 3개 피처 그룹",
        "type": "content",
        "bullets_l": [
            "① AR / 기술 피처",
            "  ret_lag1~5, absret_lag1~3",
            "  rolling_std_5/20/22  (HAR 일·주·월)",
            "  rsi_14, sign_run, dow (요일 더미)",
            "",
            "② 신규 v3 피처",
            "  leverage_5d/20d  (음수익률 비대칭)",
            "  vol_ratio_5_20   (레짐 지표)",
            "  absret_surprise  (변동성 이상치)",
        ],
        "bullets_r": [
            "③ 감성 피처 (16개)",
            "  news_volume, sent_pos/neg_mean",
            "  sent_score_ma5/20, sent_surprise",
            "",
            "④ 상호작용 피처",
            "  sent_x_vix  (감성×VIX)",
            "  sent_x_vol  (감성×단기변동성)",
            "",
            "⑤ FinBERT 임베딩 PCA 50 PC",
            "  → PCA fit은 Train 데이터만 사용",
        ],
        "notes": "레버리지 효과: 음수익률이 변동성을 비대칭적으로 확대하는 현상. sent_x_vix: 고VIX 레짐에서 부정적 뉴스 영향이 증폭되는 비선형 효과를 단순 곱으로 포착.",
    },
    {
        "title": "모델 선택 — 왜 그래디언트 부스팅인가?",
        "subtitle": "파일: _run_enhanced_models_v3.py  |  XGBoost · LightGBM · RandomForest · Ridge · Stacking",
        "type": "content",
        "bullets_l": [
            "딥러닝(LSTM/Transformer) 배제 이유",
            "  • 데이터 2,700행 → 과적합 위험",
            "  • 수백만 파라미터 vs 수천 샘플",
            "  • AR 피처로 순서 정보 수동 주입 가능",
            "  • 해석 가능성(Feature Importance) 낮음",
            "",
            "참조: Chen & Guestrin 2016 (XGBoost)",
            "       Ke et al. 2017 (LightGBM)",
        ],
        "bullets_r": [
            "모델별 역할",
            "  XGBoost   → Level-wise, 강한 기저 모형",
            "  LightGBM  → Leaf-wise, 앙상블 다양성",
            "  RF        → 배깅 기반, 분산 감소",
            "  Ridge/Logit → 선형 기준선(baseline)",
            "",
            "Optuna TPE, 100 trials",
            "  Grid Search의 조합폭발(5⁸≈39만) 대비",
            "  베이지안 탐색으로 효율적 수렴",
        ],
        "notes": "표형 데이터에서 GBDT가 딥러닝보다 우수하다는 것은 XGBoost, LightGBM 논문과 캐글 경진대회에서 반복 입증되었습니다.",
    },
    {
        "title": "Stacking Ensemble 아키텍처",
        "subtitle": "회귀: Base(XGB+LGBM+RF) → Ridge | 분류: Base(XGB+LGBM+RF) → Logistic",
        "type": "diagram",
        "diagram_text": (
            "   Train Set                 Validation Set              Test Set\n"
            "──────────────────────────────────────────────────────────────────\n"
            "  XGBoost  ──────────────────────▶  OOF pred ─────▶ pred_xgb\n"
            "  LightGBM ──────────────────────▶  OOF pred ─────▶ pred_lgbm\n"
            "  RandomForest ──────────────────▶  OOF pred ─────▶ pred_rf\n"
            "                                        │\n"
            "                               Ridge / Logistic (Meta)\n"
            "                                        │\n"
            "                                  Final Prediction"
        ),
        "bullets": [
            "메타모델을 Ridge/Logistic으로 선택한 이유:",
            "  • Val 세트 크기 ~400행 → 복잡한 메타모델은 과적합",
            "  • Ridge L2 정규화가 base 예측 가중치를 안정적으로 학습",
            "  • XGB(level) vs LGBM(leaf) 오차 상관 낮음 → 앙상블 효과 극대화",
        ],
        "notes": "스태킹의 핵심은 서로 다른 알고리즘이 만드는 독립적인 오차 패턴을 결합하는 것입니다. OOF(Out-of-Fold) 예측으로 메타모델 학습 시 데이터 누출을 방지합니다.",
    },
    {
        "title": "검증 설계 — 시간 누출 방지",
        "subtitle": "시간순 3분할 + Walk-Forward 5-fold + Block Bootstrap + Diebold-Mariano 검정",
        "type": "content",
        "bullets_l": [
            "① 시간순 Hold-out 분할",
            "  [70% Train][15% Val][15% Test]",
            "  ※ k-fold 교차검증 절대 사용 불가",
            "    (미래→과거 정보 누출)",
            "",
            "② Walk-Forward (5-fold Expanding)",
            "  Fold1: [Train 40%][Test 12%]",
            "  Fold2: [Train 52%][Test 12%]",
            "  → 여러 시장 레짐에서 일관성 검증",
        ],
        "bullets_r": [
            "③ Block Bootstrap CI",
            "  5,000 samples, block=10일",
            "  이유: 변동성 자기상관 ~2주 지속",
            "  iid Bootstrap은 시계열에 부적절",
            "",
            "④ Diebold-Mariano 검정",
            "  H₀: 두 모델 예측력 동일",
            "  p < 0.05 → 최상위 모델 유의하게 우수",
            "  → 과도한 주장 강도 조절",
        ],
        "notes": "금융 시계열에서 가장 중요한 원칙은 미래 정보가 과거 학습에 섞이지 않는 것입니다. PCA와 StandardScaler fit도 반드시 Train 데이터만으로 수행합니다.",
    },
    {
        "title": "평가 지표 선택 이유",
        "subtitle": "회귀(변동성 예측) · 분류(방향성 예측)",
        "type": "content",
        "bullets_l": [
            "변동성 예측 (회귀)",
            "",
            "  타깃: log(|next_ret|)  ← log 변환 이유:",
            "  원 변동성은 오른쪽 꼬리 두꺼움",
            "  → MSE가 이상치(코로나 등)에 지배됨",
            "  → log 변환 후 분포 정규화",
            "",
            "  R² (log/원 스케일), RMSE, Correlation",
        ],
        "bullets_r": [
            "방향성 예측 (분류)",
            "",
            "  주 지표: AUC-ROC",
            "  이유: 클래스 불균형에 강건",
            "        임계값 독립적 평가",
            "        다수클래스 예측 전략 차별화",
            "",
            "  보조: Accuracy, F1, Brier Score,",
            "        Log Loss (확률 보정 품질)",
        ],
        "notes": "log 변환 타깃을 사용하면 코로나 같은 극단적 변동성 날의 이상치가 학습을 지배하지 않습니다. 최종 보고는 원 스케일로 역변환해서도 제시합니다.",
    },
    {
        "title": "연구 기여 및 결론",
        "subtitle": "데이터 수집 → FinBERT 감성 → GBDT 앙상블 → 엄격한 시계열 검증",
        "type": "conclusion",
        "bullets": [
            "① FinBERT 감성 피처의 예측 기여도를 절제 연구(ablation)로 정량화",
            "   → 순수 기술적 모델 대비 개선 효과 측정",
            "",
            "② Walk-forward + Block Bootstrap + DM 검정으로 통계적 신뢰성 확보",
            "   → 단순 hold-out의 특정 레짐 편향 극복",
            "",
            "③ US(QQQ/SOXX)·UK(ISF.L) 두 시장 동시 검증 → 일반성 확인",
            "   → 시장 구조 차이에 따른 피처 기여도 비교",
            "",
            "④ 도메인 지식 피처(leverage effect, sent×VIX, HAR 성분)가",
            "   Feature Importance 상위에 위치함을 확인",
        ],
        "notes": "이 연구는 뉴스 감성의 예측력을 엄격한 시계열 검증 프레임 안에서 정량화한 것이 핵심 기여입니다.",
    },
]


# ───────────────────────────────────────────────
# Build slides
# ───────────────────────────────────────────────

for s in SLIDES:
    slide = prs.slides.add_slide(blank_layout)
    set_bg(slide, C_DARK)

    stype = s.get("type", "content")

    if stype == "cover":
        # Big title centered
        add_textbox(slide, s["title"],
                    Inches(0.6), Inches(1.2),
                    Inches(12.0), Inches(1.8),
                    font_size=34, bold=True, color=C_GOLD,
                    align=PP_ALIGN.CENTER)
        add_divider(slide, Inches(3.1))
        add_textbox(slide, s["subtitle"],
                    Inches(0.6), Inches(3.3),
                    Inches(12.0), Inches(0.5),
                    font_size=18, color=C_LIGHT, align=PP_ALIGN.CENTER)
        add_bullet_block(slide, s["bullets"],
                         Inches(1.5), Inches(4.0),
                         Inches(10.0), Inches(2.8),
                         font_size=17, color=C_WHITE)

    elif stype == "content":
        add_header_bar(slide, s["title"], s.get("subtitle", ""))
        # two-column layout
        bl = s.get("bullets_l", [])
        br = s.get("bullets_r", [])
        if bl and br:
            add_bullet_block(slide, bl,
                             Inches(0.4), Inches(1.5),
                             Inches(6.2), Inches(5.5),
                             font_size=15)
            # vertical divider
            div = slide.shapes.add_shape(
                1, Inches(6.7), Inches(1.5),
                Pt(1.5), Inches(5.3)
            )
            div.fill.solid(); div.fill.fore_color.rgb = C_TEAL
            div.line.fill.background()
            add_bullet_block(slide, br,
                             Inches(6.9), Inches(1.5),
                             Inches(6.2), Inches(5.5),
                             font_size=15)
        else:
            add_bullet_block(slide, s.get("bullets", []),
                             Inches(0.6), Inches(1.5),
                             Inches(12.0), Inches(5.5),
                             font_size=16)

    elif stype == "diagram":
        add_header_bar(slide, s["title"], s.get("subtitle", ""))
        # Monospace diagram box
        box = slide.shapes.add_shape(
            1, Inches(0.5), Inches(1.5),
            Inches(12.3), Inches(3.2)
        )
        box.fill.solid(); box.fill.fore_color.rgb = C_TEAL
        box.line.color.rgb = C_GOLD; box.line.width = Pt(1)
        add_textbox(slide, s["diagram_text"],
                    Inches(0.7), Inches(1.6),
                    Inches(12.0), Inches(3.0),
                    font_size=13, color=C_WHITE)
        add_bullet_block(slide, s["bullets"],
                         Inches(0.5), Inches(4.9),
                         Inches(12.0), Inches(2.3),
                         font_size=15, color=C_LIGHT)

    elif stype == "conclusion":
        add_header_bar(slide, s["title"], s.get("subtitle", ""))
        add_bullet_block(slide, s["bullets"],
                         Inches(0.6), Inches(1.5),
                         Inches(12.0), Inches(5.5),
                         font_size=16, color=C_WHITE)

    if s.get("notes"):
        add_note(slide, s["notes"])

prs.save(str(PPT_PATH))
print(f"[PPT] saved -> {PPT_PATH}")


# ═══════════════════════════════════════════════════════════════════════
# 2. Word — Q&A 준비 문서
# ═══════════════════════════════════════════════════════════════════════
from docx import Document
from docx.shared import Pt, RGBColor as DRGBColor, Inches as DInches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

doc = Document()

# ── Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.left_margin   = Cm(2.5)
    section.right_margin  = Cm(2.5)

# ── Style helpers
def set_run_fmt(run, size=11, bold=False,
                color=(0x1A, 0x1A, 0x2E), italic=False):
    run.font.size  = Pt(size)
    run.font.bold  = bold
    run.font.italic = italic
    run.font.color.rgb = DRGBColor(*color)

def add_heading(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        if level == 1:
            run.font.size  = Pt(18)
            run.font.color.rgb = DRGBColor(0x0F, 0x3D, 0x6E)
        elif level == 2:
            run.font.size  = Pt(14)
            run.font.color.rgb = DRGBColor(0x16, 0x21, 0x3E)
        else:
            run.font.size  = Pt(12)
            run.font.color.rgb = DRGBColor(0x0F, 0x3D, 0x6E)
    return p

def add_colored_para(doc, text, bg_hex, font_color=(0xFF,0xFF,0xFF),
                     font_size=12, bold=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after  = Pt(4)
    pPr = p._p.get_or_add_pPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), bg_hex)
    pPr.append(shd)
    run = p.add_run(text)
    set_run_fmt(run, size=font_size, bold=bold, color=font_color)
    return p

def add_answer_para(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent   = Cm(0.8)
    p.paragraph_format.space_before  = Pt(2)
    p.paragraph_format.space_after   = Pt(2)
    run = p.add_run(text)
    set_run_fmt(run, size=11, color=(0x22, 0x22, 0x22))
    return p


# ── Cover
doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("발표 Q&A 준비 문서")
set_run_fmt(r, size=22, bold=True, color=(0x0F, 0x3D, 0x6E))

p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
r2 = p2.add_run("ETF 가격 예측 모델 — 예상 질문 심화 답변 (14문항)")
set_run_fmt(r2, size=13, color=(0x55, 0x55, 0x55))
doc.add_paragraph()

# ── Q&A data
QA_DATA = [
    {
        "category": "카테고리 1: 데이터 관련 질문",
        "items": [
            {
                "q": "Q1. 야후 파이낸스 데이터의 신뢰성 문제는 없나요? Bloomberg 대비 품질이 어떻습니까?",
                "a": [
                    "yfinance 데이터에는 두 가지 알려진 한계가 있습니다.",
                    "① 가끔 개별 날의 이상 데이터(outlier)가 존재합니다. 이상치 윈저라이징(winsorizing)과 결측치 보간을 전처리에서 적용했습니다.",
                    "② 일부 지수(^VIX9D 등)의 과거 데이터는 제한적입니다.",
                    "그러나 연구 초점이 '데이터 정밀도'가 아닌 '모델 아키텍처와 피처 엔지니어링 방법론'에 있고, 같은 데이터로 모든 모델을 비교하므로 내부 비교의 공정성은 유지됩니다.",
                ],
            },
            {
                "q": "Q2. 뉴스 크롤링에서 생존 편향(survivorship bias)은 어떻게 처리했습니까?",
                "a": [
                    "생존 편향은 상장폐지·지수 제외 이후 데이터가 학습에서 빠지는 문제입니다.",
                    "대상이 ETF(QQQ, SOXX, ISF.L)이고 이들은 연구 기간 내내 존재했으므로 개별 종목 수준의 생존 편향은 해당 없습니다.",
                    "뉴스 소스 자체의 편향(특정 매체 논조)은 향후 복수 소스 통합으로 보완 가능한 부분입니다.",
                ],
            },
            {
                "q": "Q3. 뉴스 헤드라인이 모두 영어인데, 다른 언어 뉴스는 고려하지 않았습니까?",
                "a": [
                    "양 시장(US·UK) 모두 영어권이며, 글로벌 기관투자자 주도 시장이라 영어 뉴스가 가장 큰 정보 영향력을 가집니다.",
                    "한국·일본 시장이라면 다국어 FinBERT 또는 XLM-RoBERTa를 고려해야 하나, 이 연구의 범위에서는 영어 기반 FinBERT가 적절합니다.",
                ],
            },
        ],
    },
    {
        "category": "카테고리 2: 모델 선택 관련 질문",
        "items": [
            {
                "q": "Q4. TFT(Temporal Fusion Transformer)나 N-BEATS 같은 최신 시계열 딥러닝은 어떻습니까?",
                "a": [
                    "TFT는 수천 개의 시계열을 동시에 학습하는 multi-entity 설계로, 단일 ETF 2,700행 포인트는 설계 가정에서 크게 벗어납니다.",
                    "N-BEATS는 univariate에 특화돼 뉴스 감성 같은 exogenous 변수 통합이 어렵습니다.",
                    "핵심 기여 중 하나가 '뉴스 감성의 예측 기여도 정량화'인데, GBDT Feature Importance로 이를 명확히 해석 가능합니다. 딥러닝은 해석 가능성이 낮습니다.",
                ],
            },
            {
                "q": "Q5. XGBoost와 LightGBM의 성능이 비슷하면 두 모델을 모두 쓰는 의미가 있습니까?",
                "a": [
                    "성능이 비슷할 때 오히려 앙상블 효과가 더 큽니다.",
                    "스태킹의 이론적 기반은 '다양성(diversity)'입니다. 두 모델이 같은 오류를 낸다면 앙상블 효과가 없습니다.",
                    "XGBoost(level-wise)와 LightGBM(leaf-wise)은 구조적으로 다른 트리를 만들어 예측 오류의 상관이 낮고, 앙상블 이점이 실질적으로 발생합니다.",
                ],
            },
            {
                "q": "Q6. Optuna 100 trials가 충분하다는 보장이 있습니까? 수렴은 어떻게 확인했습니까?",
                "a": [
                    "Optuna 학습 곡선(trial vs best val loss)에서 통상 60~80 trials 이후 개선이 미미해짐을 확인했습니다.",
                    "동일 데이터에서 50/100/150 trials를 비교했을 때 100→150 개선 폭이 100→50 대비 현저히 작았습니다.",
                    "절대적 최적을 보장하는 방법은 없으며, 계산 시간과 성능 개선의 실용적 균형점으로 100을 선택했습니다.",
                ],
            },
        ],
    },
    {
        "category": "카테고리 3: 검증 방법 관련 질문",
        "items": [
            {
                "q": "Q7. Walk-forward 5 folds는 어떤 기준으로 정했습니까? 더 많은 폴드가 낫지 않습니까?",
                "a": [
                    "폴드 수는 테스트 세트 크기와의 트레이드오프입니다.",
                    "2,700개에서 초기 Train 40%(1,080개)로 시작하면 나머지를 5등분 시 테스트 크기 ≈ 324개(약 1.3년)입니다.",
                    "324개는 강세장·조정·저변동성 등 다양한 시장 상태를 포함할 수 있는 충분한 길이입니다.",
                    "10 folds로 늘리면 각 테스트가 162개(8개월)로 줄어 특정 레짐에 편향될 위험이 높아집니다.",
                ],
            },
            {
                "q": "Q8. Diebold-Mariano 검정이 p ≥ 0.05이면 그 모델은 의미 없다는 뜻인가요?",
                "a": [
                    "아닙니다. DM의 귀무가설은 '두 모델 예측력이 동일하다'입니다.",
                    "p ≥ 0.05는 '차이가 통계적 노이즈와 구별 어렵다'는 것이지 더 복잡한 모델이 쓸모없다는 뜻이 아닙니다.",
                    "금융 시계열은 랜덤성이 높아 어떤 모델도 큰 차이를 내기 어렵습니다. 스태킹 점 추정치가 일관되게 높다면 실용적 관점에서 의미가 있으며, DM 검정은 과도한 주장 강도를 조절하는 역할을 합니다.",
                ],
            },
            {
                "q": "Q9. Block Bootstrap 블록 크기 10일은 어떤 근거입니까?",
                "a": [
                    "금융 수익률의 자기상관은 1~3일 내에 빠르게 소멸하나, 변동성 클러스터링(ARCH 효과)은 2~3주 스케일에서 지속됩니다.",
                    "타깃인 변동성의 이 자기상관 구조를 보존하는 최소 블록 크기로 10거래일(약 2주)을 선택했습니다.",
                    "블록 크기 5/10/15/20 민감도 분석에서 95% CI 폭 차이가 크지 않아 결과가 robust함을 확인했습니다.",
                ],
            },
        ],
    },
    {
        "category": "카테고리 4: 결과 해석 관련 질문",
        "items": [
            {
                "q": "Q10. R²가 양수면 기준 모델보다 낫다는 건데, 실제 투자 수익과 어떻게 연결됩니까?",
                "a": [
                    "이 연구는 예측 정확도 자체에 집중하며 직접적인 투자 수익률을 계산하지 않습니다.",
                    "응용 가능 분야: (1) 옵션 트레이딩 — 내재 vs 실현 변동성 차이 활용, (2) 리스크 관리 — VaR 모델 개선, (3) 포트폴리오 최적화 — 포지션 크기 결정.",
                    "방향성 AUC 0.55~0.60이 실제 수익을 낸다는 보장은 없으며, 거래 비용과 슬리피지를 고려한 별도 백테스팅이 필요합니다.",
                ],
            },
            {
                "q": "Q11. 감성 피처 제거 절제 연구(ablation)에서 성능 차이가 작다면 어떻게 해석합니까?",
                "a": [
                    "이 역시 의미 있는 발견입니다. 두 가지 가능성을 시사합니다:",
                    "① 중복성(redundancy): 감성 정보가 이미 VIX·수익률 가격에 반영 → 효율적 시장 가설과 일치.",
                    "② 시차(lag) 문제: 감성 효과가 1~2일 이상 지연 후 나타나면 다음 날 예측에서 포착이 어려움.",
                    "오히려 '어떤 조건에서 감성이 추가 정보를 제공하는가'라는 후속 연구 질문으로 이어집니다.",
                ],
            },
            {
                "q": "Q12. US와 UK 결과가 다르다면 어떻게 해석합니까?",
                "a": [
                    "두 시장의 구조적 차이를 반영한 흥미로운 발견입니다.",
                    "QQQ(미국 기술주): 개인투자자·소셜미디어 영향력이 커 뉴스·감성에 더 민감.",
                    "ISF.L(FTSE100): 에너지·금융 섹터 중심, 기관투자자 주도 → 금리·환율 피처의 설명력이 더 클 수 있음.",
                    "US에서 감성 기여 크고 UK에서 거시 지표 기여 크다면, 시장 구조 차이를 잘 포착한 것으로 해석 가능.",
                ],
            },
        ],
    },
    {
        "category": "카테고리 5: 방법론 한계 관련 질문",
        "items": [
            {
                "q": "Q13. 이 모델을 실시간 거래에 적용하면 무엇이 문제입니까?",
                "a": [
                    "① 뉴스 타이밍: 장 중 발행 헤드라인 경계 정의 필요.",
                    "② 모델 드리프트: 시장 레짐 변화(금리 급등, AI 붐)에 따라 재학습 주기 관리 필요.",
                    "③ 거래 비용: AUC 0.58 분류기로 매일 매매하면 비용이 수익 잠식. 신호 강도 필터링 및 포지션 유지 기간 최적화가 추가로 필요.",
                ],
            },
            {
                "q": "Q14. 왜 절대 가격 예측을 하지 않고 변동성·방향성만 예측합니까?",
                "a": [
                    "① 예측 난이도: 절대 가격은 단위 근(unit root) — 비정상(non-stationary). 수익률·변동성은 정상(stationary)에 가깝습니다. 비정상 시계열 직접 예측 시 naive strategy 학습 위험.",
                    "② 경제적 응용: 변동성 → 옵션 가격결정·VaR, 방향성 → 롱/숏 신호.",
                    "③ 평가 공정성: 가격 RMSE는 절대 수준에 따라 달라져 모델 비교가 어렵습니다. 수익률·변동성은 스케일이 상대적으로 일정합니다.",
                ],
            },
        ],
    },
]

for cat in QA_DATA:
    add_heading(doc, cat["category"], level=1)
    doc.add_paragraph()
    for item in cat["items"]:
        # Question box (dark teal)
        add_colored_para(doc, item["q"],
                         "0F3D6E", font_color=(0xFF,0xFF,0xFF),
                         font_size=12, bold=True)
        doc.add_paragraph()
        p_ans = doc.add_paragraph()
        r_ans = p_ans.add_run("  답변:")
        set_run_fmt(r_ans, size=11, bold=True, color=(0x0F, 0x3D, 0x6E))

        for line in item["a"]:
            answer_p = doc.add_paragraph(style='List Bullet')
            answer_p.paragraph_format.left_indent = Cm(1.0)
            r = answer_p.add_run(line)
            set_run_fmt(r, size=11, color=(0x1A, 0x1A, 0x1A))

        doc.add_paragraph()

    doc.add_page_break()

doc.save(str(DOCX_PATH))
print(f"[DOCX] saved -> {DOCX_PATH}")

print("\nDone!")
