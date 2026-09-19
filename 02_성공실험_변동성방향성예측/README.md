# 본 실험 — ETF 실현변동성·방향성 예측

석사학위논문 §4.5의 실험 코드와 결과. 최종 확정 프로토콜은 **v10**이다.

---

## 확정 프로토콜 요약

| 항목 | 내용 |
|---|---|
| 표본 | 거래일 필터 적용 — 미국 3,018일 / 영국 3,031일 (원자료 4,382 / 4,383) |
| 모델링 표본 | 미국 3,007일 / 영국 2,886일 (피처 결측 제거 후) |
| 분할 | 학습 < 2022-05-19, 검증 < 2024-03-07, 시험 ≥ 2024-03-07 |
| 시험 표본 | 미국 452일 / 영국 434일 |
| 주 표적 | 5일 실현변동성 `log RV5`, 차분 형태로 학습 후 복원 |
| 학습 | 두 시장 공통 컷오프 이전 데이터를 합쳐 학습(pooled) |
| 보정 | 검증 구간에서 Mincer-Zarnowitz 선형 보정계수 추정 |
| 검정 | 블록 부트스트랩 95% CI (n=5,000), Diebold-Mariano (HAC + HLN 수정) |

### 반드시 알아야 할 두 가지 자료 문제

1. **달력일 전방 보간** — 원자료가 주말·공휴일을 포함한 모든 달력일을 행으로 갖고 비거래일 종가가
   전방 보간되어 있었다. 이 상태에서는 비거래일 수익률이 정확히 0이 되어
   `log|r|` 이 극단값(-18.4)을 갖고, 모델이 시장 동학이 아니라 "내일이 주말인가"를 학습한다.
   통제 이전 성능은 변동성 R² 0.93, 방향성 AUC 0.84였으나 거래일 필터 적용 후 0.37 / 0.59로 낮아졌다.
   → `_trading_days.py`
2. **`eth` 열로 인한 표본 손실** — 이더리움 계열이 2017-11부터만 존재하여 `dropna()` 가
   2014~2017 구간 전체를 삭제하고 있었다(미국 967행, 영국 925행). 결측률 5% 초과 열을 먼저 제거하면
   표본이 미국 1,797 → 3,007행으로 회복된다.

3. **컬럼 순서 비결정성 (2026-09-09 수정)** — `_run_enhanced_models_v4.py`의
   `ar_cols = list(set(ar_cols))`가 파이썬 문자열 집합의 순회 순서에 의존하고 있었다.
   이 순서는 `PYTHONHASHSEED`에 따라 프로세스마다 달라지며, LightGBM의 `colsample_bytree`가
   컬럼 순서를 기준으로 피처를 표집하므로 **같은 시드로 같은 스크립트를 돌려도 실행마다 다른 결과**가
   나왔다. 편차는 소수점 셋째 자리가 아니라 fold 단위로 R² 0.1~0.3에 달했다
   (예: 미국 fold5 Full −1.023 / −1.312 / −1.224).
   `list(dict.fromkeys(ar_cols))`로 교체하여 순서를 보존하도록 고쳤고, 이후 프로세스 간
   완전 재현을 확인하였다. **논문 §4.5의 모든 수치는 이 수정 이후 재실행한 값이다.**
   → `_run_enhanced_models_v4.py:176`

---

## 실행 순서

```
1) _download_extra_features.py        # yfinance 보조 피처 수집
2) _build_multi_headline_features.py  # 다중 헤드라인 FinBERT 감성·임베딩 생성
3) _final_v10.py                      # 변동성 최종 결과  -> _final_v10_volatility.csv, _final_v10_dm.csv
4) _final_v10_direction.py            # 방향성 최종 결과  -> _final_v10_direction.csv, _final_v10_dm_dir.csv
5) _save_folds_v10.py                 # 8-fold Walk-Forward -> _final_v10_folds.csv
6) _garch_aligned_v6.py               # 전통 모형 베이스라인(동일 표본·분할)
7) _pooled_calibrated_v9.py           # pooled vs 단독 학습 비교
8) _figs_8_10_v10.py                  # 그림 8·10 생성
9) _robustness_v13.py                 # 사후 강건성: FTSE 250 대체 표적 + fold 설명변수 -> _robustness_v13_folds.csv
```

## 파일 구성

**핵심 모듈** (다른 스크립트가 import)
- `_trading_days.py` — 거래일 필터. 미국은 yfinance 거래량 변동일, 영국은 ISF.L 가격 파일 기준
- `_run_enhanced_models_v4.py` — 데이터 적재·피처 생성·모델 정의 (v3 + 거래일 필터)
- `_explore_targets_v5.py` — 다일 실현변동성/누적 방향성 타깃 생성 (`make_targets`)
- `_regime_fixes_v7.py` — 최종 피처 프레임 구성 (`prep`) 및 레짐 처방 비교

**최종 분석**
- `_final_v10.py` / `_final_v10_direction.py` — 확정 결과 산출
- `_save_folds_v10.py` — fold별 국면 의존성
- `_garch_aligned_v6.py` — HAR-RV / GARCH / GJR-GARCH
- `_pooled_calibrated_v9.py` — 교차 시장 전이 검정
- `_figs_8_10_v10.py` — 도판

**보조**
- `_collect_gdelt_news.py` — 뉴스 커버리지 보강용 (개인 네트워크에서 실행 필요).
  **2026-09-20 실측: GDELT DOC API 는 2017-01-01 이전 시작일을 거부하고 IP 레이트 리밋(429)이 심해 목적을 달성할 수 없었다.**
  2014~2025 일별 뉴스는 `../공통데이터/collect_economic_news.py` 로 수집했다(아래 '재실험' 절)
- `_구버전_스크립트/` — v3~v9 및 탐색 단계 스크립트 보관

## 결과 파일

| 파일 | 내용 | 논문 |
|---|---|---|
| `_final_v10_volatility.csv` | 5일 변동성 피처군별 R² + 부트스트랩 CI | 표 13 |
| `_final_v10_dm.csv` | 변동성 DM 검정 | 표 14 |
| `_final_v10_direction.csv` | 방향성 AUC + CI | 표 16 |
| `_final_v10_dm_dir.csv` | 방향성 DM 검정 | 표 17 |
| `_pooled_calibrated_v9_results.csv` | 단독 vs pooled 학습 | 표 18 |
| `_robustness_v13_folds.csv` | 8-fold Walk-Forward (수준·변동폭 지표 포함) | 표 19, 그림 9(b) |
| `_final_v10_folds.csv` | 8-fold Walk-Forward (구버전 열 구성) | - |
| `{USD,UK}/results_v6/traditional_baselines.csv` | 전통 모형 | 표 15 |
| `{USD,UK}/results_v6/direction5_predictions_v10.csv` | 방향성 예측값 | 그림 8 |

## 주요 결과

- 미국 5일 실현변동성 **R² = 0.363** (95% CI 0.140~0.476), HAR-RV 0.187 · GJR-GARCH 0.152 대비 우위
- 1일 시계는 R² = 0.116 — 예측 가능성이 예측 시계에 강하게 의존
- 영국은 R² = 0.043으로 점추정은 양수이나 신뢰구간이 0을 포함
- **fold 성능을 지배하는 것은 변동성의 수준이 아니라 시험 구간 내 변동폭**이다.
  변동폭과 R²의 상관은 미국 +0.897(p=0.003) · 영국 +0.870(p=0.005) · 합산 16 fold +0.829(p<0.001).
  수준과의 상관은 미국 −0.434(p=0.283) · 영국 +0.327(p=0.429)으로 부호가 엇갈리고 비유의.
  R² 정의(분모 = 시험 구간 분산)에 따른 동어반복이 아님을 척도 무관 지표로 확인하였다
  (예측-실현 상관 기준 미국 +0.981 · 영국 +0.821).
- 방향성은 미국 AUC 0.587(CI 0.507~0.671)만 유의, 정확도는 다수 클래스 기준을 넘지 못함
- 뉴스는 금융지표·뉴스 단독 대비 유의하나 자기회귀 대비 증분은 유의하지 않음 (p = 0.213)
- pooled 학습으로 미국 R² 0.222 → 0.363 — 변동성 동학의 교차 시장 전이
- **FTSE 250(MIDD) 강건성 검정**: AR_Only +0.007 (CI −0.264~+0.186), Full −0.319.
  영국의 결과는 ISF 고유의 것이 아니며 표적 지수를 바꿔도 재현된다.

## 재실험 — 신규 일별 경제 뉴스 (2026-09-20)

기존 뉴스의 커버리지 공백(미국 221일 / 영국 713일)을 메운 `공통데이터/` 의 2014–2025 일별 경제 뉴스
(미국 NYT / 영국 Guardian, 매 달력일 ≥1건)로 §4.5 실험 전체를 다시 수행했다.
**결과와 해석은 `재실험_결과/SUMMARY.md`.** 위의 기존 결과 파일과 '주요 결과'는 그대로 두었다.

```
python _build_news_features_newsv2.py   # FinBERT(ProsusAI/finbert) + 뉴스 병합 -> *_newsv2 파일 (CPU 약 20분)
python _run_reexperiment.py             # 4개 variant x 7단계 -> 재실험_결과/<variant>/
python _summarize_reexperiment.py       # 재실험_결과/SUMMARY.md 생성
```

- variant 4개: `{oldnews, newsv2} x {PCA solver full, auto}`. 주 비교는 `oldnews_full` vs `newsv2_full`(같은 환경·절차, 뉴스만 다름).
  auto 는 scikit-learn 버전에 따라 내부 구현이 달라지므로 구현 민감도(잡음 하한) 측정용이다.
- 시장 데이터는 고정(원본과 비트 단위 동일 확인). 표본이 같아 분할 컷오프도 동일(학습 <2022-05-19, 검증 <2024-03-07).
- 스크립트는 환경 변수로 입력·출력을 분기한다(`_news_variant.py`). **변수를 주지 않으면 기존 동작과 완전히 같다.**
  `NEWS_DATA=old|new`, `PCA_SOLVER=auto|full`, `RESULT_DIR`, `EXTRA_CLEAN_FIN=1`(뉴스 파생 열 2개를 뺀 `Financial_Clean` 그룹을 **추가 행으로** 계산)
- 병합 단계(뉴스 -> `research_enhanced`)를 만드는 스크립트가 저장소에 없어서 `_build_news_features_newsv2.py` 로 재구성했다.
  비거래일 뉴스는 다음 거래일에 합산하고, 전방 보간은 하지 않는다.
- FinBERT 가중치(`USD/finbert/pytorch_model.bin`, ProsusAI/finbert)는 git 에서 제외돼 있다. `공통데이터/USD/download_finbert.py` 는
  다른 모델(finbert-tone)을 받으므로 쓰지 말고 `huggingface_hub.hf_hub_download('ProsusAI/finbert', 'pytorch_model.bin', local_dir='USD/finbert')` 를 쓴다.
  기존 헤드라인 300건으로 확률을 재계산해 최대 오차 6.9e-06 으로 일치함을 확인했다.
- 실행 환경은 `requirements_reexperiment.txt` (Python 3.12). 아래 '재현 시 유의'의 3.7.4 환경과 다르며,
  임베딩 PCA 를 쓰는 뉴스 포함 피처군은 그 차이만으로 R² 가 최대 약 0.05 달라진다(`SUMMARY.md` F·G절).

### 재실험에서 발견한 프로토콜 이슈
- `Financial_Only` 그룹에 뉴스 파생 열 `sent_x_vix`, `sent_x_vol` 이 섞여 있다(`NEWS_SENT_COLS` 미등록).
  v10 부터 존재하던 문제이며 기존 그룹 값은 바꾸지 않고 `Financial_Clean` 을 추가로 계산했다.

## 재현 시 유의

- 실행 환경은 Python 3.7.4 / lightgbm 4.6.0 / scikit-learn 1.0.2 / xgboost 1.6.2.
  (README가 이전에 명시하던 `../.venv`는 존재하지 않는다.)
- 재현의 전제는 시드가 아니라 **컬럼 순서의 결정성**이다. 피처 목록을 만들 때 `set()`을 거치면
  안 된다. 위 '자료 문제 3'을 참조할 것. 이 조건만 지키면 프로세스 간 완전 재현된다.
- 뉴스 데이터의 실제 커버리지는 미국 221일 / 영국 713일뿐이며, 학습 구간 기준으로는
  각각 3.2% / 12.9%다. 나머지 날짜는 직전 가용일 값이 이월된다. 뉴스 관련 결론을 해석할 때
  반드시 이 한계를 함께 고려할 것.
