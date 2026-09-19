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

## 일별 경제 뉴스 (2014-01-01 ~ 2025-12-31, 매일 ≥1건)
기존 `*_financial_news.csv` 의 커버리지 공백(고유 뉴스일 미국 221일 / 영국 713일)을 메우기 위해 새로 수집.
- `collect_economic_news.py` — 수집·선별·검증 스크립트 (`--smoke-test`, `--verify` 등은 파일 상단 docstring 참고).
  GDELT DOC API 는 2017 이전 미지원 + 강한 레이트 리밋이라 소스를 교체했다 (근거: `plan/20260919_economic_news_collection.md`)
- `USD/US_economic_news_daily.csv` — NYT 일별 사이트맵 기반 (business/your-money/upshot)
- `UK/UK_economic_news_daily.csv` — Guardian 비즈니스 일별 아카이브 기반
- `*_economic_news_candidates.csv` — 점수화된 후보 전체 (기준을 바꿔 `--select-only` 로 재선별 가능)
- `economic_news_collection_report.md` — 검증 리포트 (커버리지, 연도별 분포, fallback, 표본)
- `_news_cache/` — 수집 진행 기록 (중단 후 재개용)

컬럼: `year, month, date, headline, source, section, url, relevance_score, day_rank, is_fallback`
(기존 `*_financial_news.csv` 의 year/month/date/headline/source 와 호환).
`is_fallback=1` 은 그날 "의미 있음" 기준(점수 ≥ 3)을 넘는 기사가 없어 최고점 1건만 유지한 날.
날짜는 매체 발행일 기준이며 기사 시각은 없다 — 예측 시점 정렬(당일/전일)은 모델링 단계에서 결정할 것.

## 참고
- FinBERT 모델 가중치(`finbert/`)는 여기로 옮기지 않고 프로젝트 루트의 `UK/finbert`,
  `USD/finbert`에 그대로 둠 — 마켓당 ~1.3GB로 두 실험이 그대로 공유해서 씀.
