# 논문 발표 문서화 플랜
## ETF 가격 예측 모델 — 코드·실험 설명 문서

---

## 목표
코드와 실험 전 과정을 논문 발표에서 자신있게 설명할 수 있도록
**데이터 수집 → 전처리 → 모델 선택 이유 → 실험 설계 → 결과 해석**
순서로 이어지는 완성도 높은 설명 문서를 만든다.

---

## 문서 구조 (6개 섹션)

### Section 1. 연구 개요
- 연구 문제: ETF(QQQ/SOXX – US, ISF.L – UK) 다음 날 **변동성(volatility)** 과 **방향성(direction)** 예측
- 데이터 기간: 2013-12-01 ~ 2024-12-31 (약 11년)
- 두 과제(Task):
  - **회귀**: 다음 날 실현 변동성 = `|log(P_t+1/P_t)|`
  - **분류**: 다음 날 수익률 방향 = `1(log return > 0)`

---

### Section 2. 데이터 수집 단계
**파일:** `_download_extra_features.py`

#### 2-1. 왜 yfinance인가?
- Bloomberg/Refinitiv 대비 **무료·API 키 불필요** → 재현 가능성 보장
- 개인 연구 규모에서 충분한 품질 (daily OHLCV, adjusted close)
- 대안(Quandl FRED 직접 API)은 일부 지표 누락 → yfinance가 포괄성 우수

#### 2-2. 수집 대상 (US 21개 / UK 22개 심볼)
| 카테고리 | 심볼 예시 | 선택 이유 |
|---|---|---|
| 광의 시장 지수 | SPY, IWM, DIA | 시장 전반 모멘텀·위험 레짐 포착 |
| 섹터 ETF | XLK, SOXX, XLF, XLE | QQQ(테크), FTSE100(에너지·금융) 설명력 |
| 금리·크레딧 | TLT, HYG, LQD | 듀레이션·크레딧 스프레드 = 변동성 선행지표 |
| 달러·수익률 곡선 | UUP, ^FVX, ^TYX | 금리 환경이 ETF 변동성에 직접 영향 |
| 변동성 지수 | ^VIX9D, ^VVIX | 단기 변동성 레짐 탐지 |
| 아시아 야간 신호 | ^N225, ^HSI | 미국 장 개장 전 글로벌 리스크 전이 |
| 암호화폐 심리 | ETH-USD | 위험 선호도(risk-on/off) 대리변수 |

#### 2-3. 뉴스 데이터
- US: `US_financial_news.csv` (약 23,000 헤드라인)
- UK: `UK_financial_news.csv` (약 36,772 헤드라인)
- 크롤링: `*_headline_crawling_code.ipynb` 참조

---

### Section 3. 전처리 및 특성 공학
**파일:** `_build_multi_headline_features.py`, `_run_enhanced_models_v3.py`

#### 3-1. 뉴스 데이터 전처리
- **데이터 누출 방지**: "market close summary" 패턴 정규식 필터
  - 예) "closed at", "stocks close", "ended the day" 등
  - 이유: 해당 헤드라인은 **당일 종가 반영 → 미래 정보 누출**
- 필터 후 잔여 헤드라인만 FinBERT에 투입

#### 3-2. FinBERT 감성 분석
**왜 FinBERT인가? (vs TF-IDF, VADER, GPT-4)**

| 비교 대상 | 한계 | FinBERT 우위 |
|---|---|---|
| TF-IDF | 어순·문맥 무시, 단순 단어 빈도 | 문맥 기반 transformer |
| VADER | 일반 소셜미디어 훈련 | 금융 도메인 전용 파인튜닝 |
| GPT-4 API | 재현 불가, 비용, 데이터 프라이버시 | 로컬 실행, 비용 0, 완전 재현 가능 |
| RoBERTa-general | 금융 어휘 미반영 | ProsusAI/finbert: 금융 뉴스 47,000개 파인튜닝 |

- 출력: headline당 `[pos, neg, neu]` 확률 + `[CLS]` 768-dim 임베딩
- 단일 forward pass로 분류기 출력 + 인코더 hidden state 동시 추출 → 효율적

#### 3-3. 일별 집계 (12개 감성 피처)
```
news_volume, sent_pos_mean, sent_neg_mean, sent_neu_mean,
sent_score_mean (= pos - neg), sent_score_std, sent_max_neg,
sent_pos_ratio, sent_neg_ratio,
sent_score_ma5, sent_score_ma20, sent_surprise (= score - ma5),
sent_momentum_3d, sent_vol_5d, news_volume_chg
```

#### 3-4. 임베딩 PCA 압축
- 768-dim → **50 PCs** (학습 데이터 분산의 약 80%+ 보존)
- 왜 PCA? 768-dim을 그대로 쓰면 차원의 저주 → 모델 학습 불안정
- PCA는 **학습 데이터만으로 fit**, 검증/테스트에는 transform만 적용 → 누출 없음

#### 3-5. 가격 기반 AR/기술 피처
| 피처 군 | 피처명 | 경제적 의미 |
|---|---|---|
| HAR 성분 | absret_5/20/22d | 일·주·월 변동성 자기회귀 (Corsi 2009) |
| 레버리지 효과 | leverage_5d/20d | 음(-)수익률이 변동성을 비대칭 확대 |
| 기술 지표 | rsi_14 | 과매수/과매도 모멘텀 |
| 부호 연속성 | sign_run | 추세 지속성 포착 |
| 레짐 지표 | vol_ratio_5_20 | 단기 vs 장기 변동성 비율 |
| 감성×VIX | sent_x_vix | 고변동성 레짐에서 감성 영향 증폭 |
| 요일 더미 | dow | 요일 효과 (월요일 효과 등) |

---

### Section 4. 모델 선택 및 실험 설계
**파일:** `_run_enhanced_models_v3.py`

#### 4-1. 왜 그래디언트 부스팅 계열인가? (vs LSTM/Transformer)

| 모델 유형 | 장점 | 이 연구에서의 한계 |
|---|---|---|
| LSTM/GRU | 시계열 순서 자동 학습 | 일별 금융데이터 2,700행 수준 → 과적합 위험, 튜닝 비용 큼 |
| Transformer | 장거리 의존성 | 동일한 데이터 규모 문제, 해석 어려움 |
| **XGBoost/LightGBM** | 표형 데이터 최강, 해석 가능, 결측치 강건 | 순서 정보 수동 피처로 대체 (AR 피처) |
| Random Forest | 분산 감소, 앙상블 기반 | 개별 트리보다 부스팅에 밀림 |
| Ridge/Logistic | 선형 기저모형 | 비선형 패턴 미포착 → 기준선(baseline) 역할 |

→ **핵심 선택 이유**: 표형(tabular) 데이터에서 그래디언트 부스팅이 딥러닝보다 우수함은 Chen & Guestrin (2016, XGBoost), Ke et al. (2017, LightGBM) 논문 및 다수 캐글 우승 사례로 입증. 금융 시계열의 경우 AR 피처를 명시적으로 구성하면 LSTM의 순서 학습 이점을 대부분 커버 가능.

#### 4-2. XGBoost vs LightGBM
| 비교 항목 | XGBoost | LightGBM |
|---|---|---|
| 트리 성장 방식 | Level-wise | Leaf-wise (GOSS+EFB) |
| 속도 | 느림 | 빠름 (대규모 데이터에서) |
| 정확도 | 균형적 | 리프 방식 → 더 깊은 트리 가능 |
| 이 연구에서 | 강한 기저모형 | 앙상블 다양성 기여 |
→ **둘 다 사용하는 이유**: 스태킹 앙상블에서 두 모델의 **예측 오차가 상관되지 않을수록** 앙상블 효과 극대화

#### 4-3. Stacking Ensemble 설계
```
Base Layer:  XGBoost + LightGBM + Random Forest
            (각각 독립 학습, validation에서 OOF 예측 생성)

Meta Layer:  Ridge (회귀) / Logistic Regression (분류)
            (base 예측 3개를 입력으로 linear combination 학습)
```
- 왜 Ridge/Logistic을 메타모델로? → 심플한 선형 결합이 과적합 방지, OOF 예측 수(N_val)가 작아 복잡한 메타모델 불필요

#### 4-4. 하이퍼파라미터 최적화: Optuna (TPE Sampler)
**왜 Optuna인가? (vs GridSearch, RandomSearch)**
- GridSearch: 조합 폭발 (8개 파라미터 × 5값 = 390,625 조합)
- RandomSearch: 효율적이나 사전 평가 정보 미활용
- **Optuna TPE**: 이전 시도 결과를 베이지안 방식으로 학습 → 100 trials만으로도 수렴
- Early stopping(40 rounds)으로 각 trial 조기 종료 → 총 시간 절감

---

### Section 5. 검증 설계 (시간 누출 방지)

#### 5-1. 시간순 3분할
```
[70% Train] | [15% Val] | [15% Test]  ← 시간 순서 유지
```
- 절대로 k-fold cross-validation 사용 X → 미래 데이터가 과거 학습에 섞임
- PCA fit, StandardScaler fit 모두 Train에서만 → transform만 Val/Test에 적용

#### 5-2. Walk-Forward Validation (5-fold Expanding Window)
```
Fold 1: [--Train 40%--] | [Test 12%]
Fold 2: [----Train 52%----] | [Test 12%]
Fold 3: [------Train 64%------] | [Test 12%]
...
```
- 단순 hold-out의 단점: 특정 기간 우연히 쉬운 시장 → 성능 과대평가 가능
- Walk-forward: 여러 시장 레짐에 걸쳐 **일관성** 검증

#### 5-3. Block Bootstrap (5,000 samples, block size=10)
- 금융 시계열은 **자기상관** → 단순 iid bootstrap은 부적절
- Block bootstrap: 연속 10일을 하나의 블록으로 리샘플링
- 신뢰구간(CI) 계산 → 포인트 추정치의 통계적 불확실성 정량화

#### 5-4. Diebold-Mariano 검정
- 두 모델의 예측 정확도 차이가 **통계적으로 유의**한지 검증
- H0: 두 모델 예측력 동일 (p < 0.05면 최상위 모델이 유의하게 우수)

---

### Section 6. 평가 지표 선택 이유

#### 6-1. 변동성 예측 (회귀)
| 지표 | 이유 |
|---|---|
| **R²** (log-vol scale) | 로그 변환된 타깃의 설명력 → 분포 정규화 후 평가 |
| R² (원래 scale) | 실제 변동성 단위로 경제적 해석 |
| RMSE | 큰 오차에 민감 → 꼬리 리스크 포착 |
| Correlation | 예측 방향성 일치도 |

- 왜 **log(vol)**을 타깃으로? → 원래 변동성은 오른쪽 꼬리가 두꺼워 MSE 최소화 시 이상치에 지배됨. log 변환 후 분포가 정규분포에 가까워짐

#### 6-2. 방향성 예측 (분류)
| 지표 | 이유 |
|---|---|
| **AUC-ROC** | 클래스 불균형에 강건, 임계값 독립적 |
| Accuracy | 직관적 해석 |
| F1-score | 정밀도-재현율 조화 평균 |
| Brier Score | 확률 보정(calibration) 품질 |
| Log Loss | 확률 분포의 정보 손실 측정 |

---

## 문서 작성 순서 (작업 계획)

| 단계 | 작업 | 출력물 |
|---|---|---|
| 1 | 전체 파이프라인 다이어그램 설명 텍스트 작성 | Section 1 초안 |
| 2 | 데이터 수집 코드 (`_download_extra_features.py`) 라인별 주석 + 설명 | Section 2 완성 |
| 3 | FinBERT 파이프라인 (`_build_multi_headline_features.py`) 설명 | Section 3 완성 |
| 4 | 모델 선택 비교표 + 코드 근거 연결 | Section 4 완성 |
| 5 | 검증 설계 시각화 설명 + 통계 검정 해설 | Section 5 완성 |
| 6 | 지표 선택 이유 + 실험 결과 해석 가이드 | Section 6 완성 |
| 7 | 전체 문서 통합 → 발표 스크립트 형식 변환 | 최종 발표 문서 |

---

## 발표 시 예상 질문 & 답변 준비 포인트

1. **"왜 LSTM/Transformer 안 쓰나요?"**
   → 데이터 행 수(~2,700일) 대비 딥러닝은 과적합. AR 피처로 순서 정보 명시 대체. 표형 데이터는 GBDT가 학술·실무에서 일관되게 우수.

2. **"FinBERT말고 ChatGPT 같은 LLM은요?"**
   → 재현 불가 (API 버전 변경), 비용, 입력 데이터 프라이버시, 배치 처리 한계. FinBERT는 로컬 실행·완전 재현·비용 0.

3. **"walk-forward와 hold-out 결과가 다르면 어떻게 해석?"**
   → Hold-out은 특정 시장 레짐에 편향될 수 있음. Walk-forward 평균이 더 견고한 추정치. 두 결과의 방향이 일치하면 신뢰도 높음.

4. **"스태킹의 메타모델을 왜 Ridge/Logistic으로 선택?"**
   → OOF 예측 데이터 수가 Val 크기(~400행)로 제한적 → 복잡한 메타모델은 과적합. Ridge의 L2 정규화가 base 예측들의 가중치를 안정적으로 학습.

5. **"블록 부트스트랩 블록 크기 10을 왜 선택?"**
   → 일별 금융 수익률의 자기상관은 통상 2~3주 내 소멸. 10일(2주) 블록이 이 구조를 보존하면서 충분한 리샘플 다양성 확보.
