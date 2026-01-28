# ETF Price Prediction Models

이 프로젝트는 UK와 USD ETF 가격 예측을 위한 머신러닝/딥러닝 모델을 포함합니다.

## 폴더 구조

- `UK/`: UK ETF (FTSE100) 관련 데이터 및 모델
- `USD/`: US ETF (QQQ, SOXX) 관련 데이터 및 모델

## 각 폴더 포함 내용

### 데이터
- 금융 지표 데이터 (금, 원유, 채권, 환율 등)
- 뉴스 헤드라인 데이터
- ETF 가격 데이터

### 노트북
- `*_headline_crawling_code.ipynb`: 뉴스 헤드라인 크롤링
- `*_databuilder.ipynb`: 데이터 통합 및 전처리
- `*_modeling.ipynb`: 머신러닝/딥러닝 모델링

## 모델

각 폴더에는 다음 모델들이 구현되어 있습니다:

1. **TF-IDF + Random Forest**
2. **TF-IDF + LightGBM**
3. **TF-IDF + BiLSTM**
4. **Hybrid (TF-IDF + RF + BiLSTM)**
5. **FinBERT + Ridge**
6. **FinBERT + BiLSTM**
7. **FinBERT + Hybrid (Ridge + BiLSTM)**
8. **FinBERT + LightGBM**

## 설치 및 실행

```bash
# 필요한 패키지 설치
pip install pandas numpy scikit-learn lightgbm tensorflow transformers torch nltk

# 노트북 실행
jupyter notebook
```

## 주의사항

- FinBERT 모델 파일(pytorch_model.bin, tf_model.h5, flax_model.msgpack)은 용량이 커서 리포지토리에 포함되지 않습니다.
- FinBERT 모델이 필요한 경우 `download_finbert.py`를 실행하거나 Hugging Face에서 다운로드하세요.

## 라이선스

이 프로젝트는 교육 목적으로 만들어졌습니다.
