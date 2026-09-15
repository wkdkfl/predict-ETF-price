# 실패 실험 — 일별 수익률 수준(level) 예측 (예비 실험, 논문 §4.5–4.7)

TF-IDF/FinBERT + 금융지표로 ETF의 **일별 수익률(%) 자체**를 예측하려던 예비 실험. 21개 모델 모두
Naive Mean 베이스라인을 유의하게 상회하지 못함(R² < 0.03) — 효율적 시장가설과 일치하는 "실패"
결과이며, 이 실패가 논문에서 변동성·방향성 예측(성공 실험)으로 표적을 전환하는 근거가 되었다.

## 구성
- `UK/UK_modeling.ipynb`, `USD/US_modeling.ipynb` — 핵심 실험 코드 (데이터 로드 → TF-IDF/FinBERT
  피처 추출 → Baseline/Tree/LSTM/Transformer/Ensemble 학습 → 평가·시각화 → Walk-Forward →
  통계 검정까지 전 과정이 담긴 단일 노트북)
- `UK/UK_research.csv`, `USD/US_research.csv` — 이 노트북의 입력 데이터
- `USD/US_research.csv.qqq_backup` — 타깃을 QQQ→SPY로 수정하기 전 백업본
- `*/figures/` — 노트북이 생성하는 fig1~fig9 (성능 비교, 예측 vs 실제, 잔차, 학습곡선, 피처중요도 등)
- `*/results/` — 노트북 출력 결과 CSV (t-1 엄밀 시차 버전, 논문 본문 수치)
- `*/results_contemporaneous/` — 동시간대(t-day) 정보를 포함했을 때의 비교 결과 (§5.7 데이터 누출
  진단에 사용 — "당일 정보 포함 시 R²가 부풀려진다"는 근거)
- `*/nltk_data/` — 노트북의 TF-IDF 전처리(불용어 제거 등)에 필요한 NLTK 리소스

## 참고
- FinBERT 모델 가중치(`./finbert`)는 용량이 커서(마켓당 ~1.3GB) 이동하지 않고 원래 위치
  (프로젝트 루트의 `UK/finbert`, `USD/finbert`)에 그대로 둠. 노트북에서는
  `../../UK/finbert` / `../../USD/finbert` 상대경로로 참조하도록 이미 수정되어 있음.
- 노트북을 Jupyter로 열 때 실행 디렉터리(cwd)가 노트북 파일이 있는 폴더(`UK/` 또는 `USD/`)여야
  상대경로(`UK_research.csv`, `results/...` 등)가 정상 동작함.
