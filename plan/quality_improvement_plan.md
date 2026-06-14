# 논문 퀄리티 개선 플랜
**작성일**: 2026-06-14  
**대상 파일**: `thesis_draft_v20.docx`  
**현재 상태**: 실험 완료, AR_Only ablation 반영, Items 4-8 수정 완료  

---

## 현재 논문의 핵심 강점 (유지할 것)
- Walk-Forward Validation + Block Bootstrap + Diebold-Mariano 3단 검증
- GARCH/HAR-RV 대비 ML 압도적 우위 (HAR-RV R²=0.034 vs RF R²=0.830)
- AR_Only ablation으로 "뉴스 기여"를 정직하게 분리한 설계
- 미국/영국 교차 시장 비교의 일관성

---

## TIER 1 — 심사 통과에 필수 (제출 전 반드시 완료)

### ~~T1-1. ETF 명칭 통일~~ ✅ 확인 완료 — 수정 불필요
- **재검토 결과**: US `ETF` 컬럼 값 ~$148 (2014-01) = SPY backward-adjusted 가격과 일치
  - `qqq_close` 컬럼은 예측 타겟이 아닌 **입력 피처**였음 (혼동 오류)
  - 논문의 "SPY (S&P 500)" 표기는 데이터와 정확히 일치
- **UK 소수 이슈**: UK `ETF` 컬럼 값 ~6717 = FTSE 100 **지수(index)** 레벨, ISF.L ETF 가격(~660p)과 수치가 다름
  - 실질적 차이 없음 (상관계수 >0.99), 필요시 §3 데이터 설명에 "FTSE 100 가격 지수를 UK 예측 타겟으로 사용" 명시로 족함

### T1-2. RQ3 결론 수정 (AR_Only 반영)
- **문제**: P576(결론), P577(RQ3)에 "News_Only ≈ Full → 뉴스가 지배적" 주장이 아직 남아있음
- **올바른 결론**:
  - 변동성: AR 피처가 1차적 신호 (AR_Only R²=0.83-0.94), 뉴스 기여 미미
  - 방향성: 뉴스가 유의미한 기여 (AR_Only AUC 0.724→Full 0.749 US, 0.780→0.832 UK)
- **수정 대상 단락**: P576, P577, P579, P581 (결론 섹션)
- **난이도**: 중간 (논리 구조 변경)

### T1-3. Table 번호 순서 정렬
- **문제**: 표 번호가 `1, 2, 2a, 2b, 2c, 3, 3a ... 11c, 11d ... 8`로 완전히 뒤섞임
- **올바른 순서**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (서브테이블은 a/b 허용하되 순서 준수)
- **방법**: Word에서 Ctrl+H로 일괄 수정 또는 python-docx 스크립트
- **난이도**: 낮음 (기계적 작업)

### T1-4. Figure TOC 업데이트
- **문제**: 그림 9-14가 목차의 그림 목록에 없음
- **방법**: 논문 내 그림 목록 섹션에 Figure 9-14 제목 추가
- **난이도**: 낮음

### T1-5. Chapter 4/5 중복 해소
- **문제**: §4.5-4.7(결과 수치 제시) vs §5.1-5.7(동일 결과 재해석) — 텍스트가 거의 동일
- **해결 전략**:
  - §4.5-4.7: **표와 수치만 남기고** 해석 문장 삭제 (예: "이는 ~를 의미한다" 류 문장 삭제)
  - §5.1-5.7: 수치 반복 없이 **의미·함의·비교** 위주로만 작성
- **예시**: §4.5에서 "RandomForest R²=0.026, p>0.25로 EMH와 일치한다" 종류 문장 → §5.1로 이동
- **난이도**: 높음 (주의 깊은 수동 편집 필요)

---

## TIER 2 — 논문 완성도 향상 (가능하면 완료)

### T2-1. 그림 품질 개선 및 다양화
- **문제**: 현재 그림이 실험 출력 결과 플롯만 존재, 설명적 그림 부족
- **추가해야 할 그림**:
  1. **파이프라인 전체 흐름도** (데이터 수집 → FinBERT → Feature Engineering → Models → Evaluation)
     - `fig_pipeline.png` 파일이 있으나 논문 내 삽입 확인 필요
  2. **Ablation 막대그래프** (5그룹 × 2시장 × 변동성+방향성) — 새로운 AR_Only 결과 반영
  3. **예측값 vs 실제값 시계열 플롯** (Test 구간의 predicted vs actual volatility, 월별)
  4. **GARCH vs ML 비교 막대그래프** (논문에 시각적 증거 추가)
  5. **Walk-Forward fold별 성능 추이 선그래프** (시간에 따른 모델 안정성)
  6. **FinBERT 임베딩 t-SNE 시각화** (감성 클러스터 분포)
- **방법**: matplotlib/seaborn으로 생성, `_generate_figures.py` 스크립트 작성
- **난이도**: 중간

### T2-2. SHAP 기반 피처 중요도 분석 추가
- **문제**: 현재 built-in feature importance(XGBoost gain)만 사용 — biased, unstable
- **개선**: SHAP (SHapley Additive exPlanations) 값 계산
  - 뉴스 피처 vs AR 피처 vs 금융 피처의 SHAP 기여 비교
  - 시장별 (US vs UK) SHAP 비교
  - AR_Only ablation 결과와 SHAP 해석의 일관성 확인
- **방법**: `pip install shap`, XGBoost 모델에 `shap.TreeExplainer` 적용
- **코드 위치**: `_run_enhanced_models_v3.py` 또는 별도 `_shap_analysis.py`
- **난이도**: 중간

### T2-3. Realized Volatility 프록시 정당화
- **문제**: `|r_t|`(절대 수익률)을 realized volatility proxy로 사용 — 업계 표준은 5-min RV 또는 `√(Σr²)`
- **개선 방법**:
  - §3 또는 §4.8.4에 "|r_t| 사용 이유" 명시: "일별 데이터만 접근 가능한 환경에서 가장 널리 쓰이는 daily RV proxy (Andersen et al., 2003)"
  - 또는 `high-low range estimator` (Garman-Klass) 추가 비교
  - 주석에 "`target_vol = |r_{t+1}|` is a noisy but accessible proxy for realized volatility" 추가
- **난이도**: 낮음 (문헌 추가 + 설명 문장)

### T2-4. Multiple Testing 보정 언급
- **문제**: 21개 모델 × 2 시장 × 여러 지표를 비교하면서 multiple testing correction 없음
- **개선**: 
  - 주석 또는 한계 절에 "다수 모델 비교의 multiple testing 문제를 인식하며, Bonferroni 보정 적용 시 가장 유의한 결과도 임계값 이하임을 확인" 추가
  - 또는 실제로 Bonferroni 보정 p-value 보고
- **난이도**: 낮음

### T2-5. 결론 부분 AR_Only 완전 반영
- **현재 P579/P581 결론**: "뉴스 텍스트와 금융지표를 결합한 프레임워크가 높은 R² 달성" — 뉴스 기여를 과대 서술
- **올바른 결론 구조**:
  ```
  1. 변동성: 가격 AR 패턴이 1차 드라이버 (HAR-RV 대비 ML 우위는 AR 피처의 풍부함에서)
  2. 방향성: 뉴스 감성이 AR 베이스라인 대비 통계적으로 유의한 향상 제공
  3. 공통: 전통 GARCH/HAR-RV 대비 ML의 명확한 우위 (R² +0.80 차이)
  ```
- **난이도**: 중간

### T2-6. 참고문헌 보완
- **누락된 중요 문헌**:
  - Hansen, P. R., & Lunde, A. (2005). A forecast comparison of volatility models: Does anything beat a GARCH(1,1)? *Journal of Applied Econometrics*
  - Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics* ← 이미 [44]로 인용되어 있을 수 있음
  - Andersen, T. G., Bollerslev, T., Diebold, F. X., & Labys, P. (2003). Modeling and forecasting realized volatility. *Econometrica*
  - Lundberg, S., & Lee, S. I. (2017). A unified approach to interpreting model predictions (SHAP). *NeurIPS* ← T2-2 추가 시 필요
- **방법**: 참고문헌 섹션에 추가, 본문에서 인용

---

## TIER 3 — 선택적 향상 (시간 여유 시)

### T3-1. 실거래 전략 시뮬레이션 추가
- **내용**: 방향성 예측 AUC=0.75-0.84를 실제 Long/Short 전략으로 변환
  - 예측 방향에 따라 매수/매도 시그널 생성
  - 연간 수익률, 최대낙폭(MDD), Sharpe ratio 계산
  - Buy-and-Hold 대비 초과수익 측정
- **목적**: "실무 활용 가능성" 절(§5.10.4)의 주장에 실증적 근거 제공
- **주의**: 거래비용 포함 필수 (없으면 심사위원 지적)
- **난이도**: 높음

### T3-2. 추가 ETF 검증 (외부 타당성)
- **문제**: US는 QQQ(NASDAQ-100), UK는 ISF(FTSE 100) — 각각 1개 ETF만 사용
- **개선**: 
  - US: SPY(S&P 500) 추가 또는 교체
  - UK: VUKE(Vanguard FTSE 100) 추가
  - 결과 일관성 확인 → 외부 타당성(external validity) 강화
- **난이도**: 중간 (데이터 재수집 + 재실행 필요)

### T3-3. 텍스트 감성의 시장 영향 시차 분석 심화
- **현재**: 뉴스 t-0 적용(당일 뉴스를 당일 데이터로), 금융 t-1
- **추가 분석**: lag 0, 1, 2, 3일로 뉴스 시차를 바꿔가며 AUC 변화 측정
  - "뉴스가 시장에 반영되는 시차"를 데이터로 규명
  - §5.5 "뉴스 시차 효과 분석" 절 강화
- **난이도**: 낮음 (코드 수정 소폭)

### T3-4. Stacking Ensemble 메타러너 개선
- **현재**: Ridge (회귀) / Logistic (분류) 메타러너
- **개선**: 
  - 메타러너도 XGBoost로 교체하여 비선형 앙상블 가능성 탐색
  - Out-of-fold predictions로 메타 피처 생성 (현재 KFold shuffle=False 사용 — 시간적 순서 미보장)
  - TimeSeriesSplit으로 교체하여 stacking의 시간 누출 완전 차단
- **난이도**: 중간

---

## 수정 우선순위 로드맵

```
[즉시 — 제출 전 필수]
T1-1 ETF 명칭 통일 (1시간)
T1-2 RQ3/결론 AR_Only 반영 (3시간)
T1-3 Table 번호 정렬 (1시간)
T1-4 Figure TOC 업데이트 (30분)
T1-5 Chapter 4/5 중복 해소 (반나절)

[제출 전 권장]
T2-1 그림 추가/개선 (하루)
T2-2 SHAP 분석 추가 (반나절)
T2-3 RV proxy 정당화 문장 추가 (30분)
T2-4 Multiple testing 보정 언급 (30분)
T2-5 결론 AR_Only 완전 반영 (1시간)
T2-6 참고문헌 보완 (1시간)

[여유 있을 때]
T3-1 거래 전략 시뮬레이션
T3-2 추가 ETF 검증
T3-3 뉴스 시차 분석 심화
T3-4 Stacking 메타러너 개선
```

---

## 그림 추가 가이드 (T2-1 상세)

### 추가 필요 그림 목록

| 번호 | 제목 | 내용 | 삽입 위치 |
|---|---|---|---|
| Fig A | 연구 파이프라인 전체 흐름도 | 데이터→NLP→FE→모델→검증 박스 다이어그램 | §3 서두 |
| Fig B | AR_Only Ablation 결과 비교 | 5그룹 × vol R² + dir AUC 막대그래프 | §4.8.8 |
| Fig C | GARCH vs HAR-RV vs ML 비교 | R² 비교 막대그래프 (전통 vs ML) | §4.8.6 |
| Fig D | 예측값 vs 실제값 시계열 | Test 구간 predicted/actual volatility | §4.8.6 |
| Fig E | Walk-Forward fold별 성능 | 시간축에 따른 AUC 변화 선그래프 | §4.4 |

### 코드 참조
```python
# 실행 방법
.venv\Scripts\python.exe _generate_diagrams.py  # 기존 다이어그램 생성 스크립트
# 새 그림은 _generate_figures_v2.py 작성 권장
```

---

## 현재 논문 버전 기록

| 버전 | 주요 변경 |
|---|---|
| v19 | 원본 (AR 피처 문제 포함) |
| v20 | Items 4-8 수정, AR_Only ablation 해석 반영, GARCH 비교 추가 |
| **v21 목표** | T1-1~T1-5 완료 (ETF명칭, RQ3결론, 표번호, 챕터중복 해소) |
| **v22 목표** | T2-1~T2-6 완료 (그림, SHAP, 참고문헌) |

---

## 심사 예상 질문 대비 현황

| 질문 | 현재 준비 상태 |
|---|---|
| "왜 GARCH 안 쓰고 ML?" | ✅ GARCH R²=-0.21 실험 결과 있음 |
| "뉴스가 진짜 도움됨?" | ✅ AR_Only ablation으로 방향성+0.025-0.052 AUC 증명 |
| "Optuna에서 test leakage?" | ✅ Train+Val만 사용 명시됨 |
| "R²=0.83이 너무 높지 않나?" | ⚠️ AR 피처의 변동성 클러스터링으로 설명 가능하나 문장 보강 필요 |
| "ETF가 SPY인가 QQQ인가?" | ❌ T1-1 수정 필요 |
| "결론에서 뉴스 기여 과장?" | ⚠️ T1-2, T2-5 수정 필요 |
| "그림이 왜 이렇게 적나?" | ⚠️ T2-1 그림 추가 필요 |
| "Multiple testing 했나?" | ⚠️ T2-4 언급 추가 필요 |
