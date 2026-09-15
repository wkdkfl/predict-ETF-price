# 공통 데이터 — 실패/성공 실험 모두의 상위 원천 데이터

`01_실패실험_수익률예측`과 `02_성공실험_변동성방향성예측` 어느 한쪽에만 속하지 않는, 두 실험의
공통 원천이거나 더 이상 재실행되지 않는 archival 데이터·코드.

## 구성
- `UK_gold/oil/bond/VIX/CAD/CNY/EUR/JPY/MXN/FTSE100.csv`, `Bitcoin.csv`, `Economic_Lexicon.csv` —
  최초 `research.csv`를 만드는 데 쓰인 원시 매크로 지표
- `{UK,US}_databuilder.ipynb`, `US_dataset.ipynb` — 위 원시 CSV를 병합해 `research.csv`를 만든
  데이터 구축 노트북. **입력 파일(`UK_news.csv`, `US_news.csv` 등)이 이미 삭제되어 현재는
  재실행 불가한 archival 코드**이며, 참고용으로만 보관
- `{UK,US}_headline_crawling_code.ipynb` — 뉴스 헤드라인 크롤링 코드
- `US_news.csv` — 구버전 원시 뉴스 데이터 (UK 쪽 대응 파일은 이미 삭제됨; 현재 파이프라인은
  `*_financial_news.csv`를 사용)
- `USD/download_finbert.py` — FinBERT 모델 최초 다운로드 스크립트 (1회성 셋업 유틸리티)

## 참고
- FinBERT 모델 가중치(`finbert/`)는 여기로 옮기지 않고 프로젝트 루트의 `UK/finbert`,
  `USD/finbert`에 그대로 둠 — 마켓당 ~1.3GB로 두 실험이 그대로 공유해서 씀.
