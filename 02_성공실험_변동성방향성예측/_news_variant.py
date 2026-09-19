# -*- coding: utf-8 -*-
"""재실험 변형(variant) 설정 — 환경 변수로 뉴스 데이터, PCA 솔버, 결과 저장 경로를 바꾼다.

환경 변수를 주지 않으면 기존 동작과 완전히 동일하다(기존 뉴스, PCA auto, 결과는 이 폴더에 기록).
재실험은 `_run_reexperiment.py` 가 변수를 명시적으로 지정해 실행한다.

  NEWS_DATA   old(기본) | new
              old : {US,UK}_research_enhanced.csv, news_embeddings_daily_aligned.npy
              new : {US,UK}_research_enhanced_newsv2.csv, news_embeddings_daily_aligned_newsv2.npy
                    (공통데이터/ 의 2014~2025 일별 경제 뉴스로 재생성, `_build_news_features_newsv2.py`)
  PCA_SOLVER  auto(기본) | full | randomized
              임베딩 PCA 의 SVD 솔버. auto 는 scikit-learn 버전에 따라 내부 구현이 달라 뉴스 포함 피처군의
              R² 가 최대 ±0.05 변한다(2026-09-20 실측). full 은 결정적이라 버전에 무관하다.
  RESULT_DIR  결과 CSV 저장 폴더 (기본: 이 폴더)
"""
from __future__ import annotations

import os
from pathlib import Path

BASE = Path(__file__).resolve().parent

NEWS_DATA = os.environ.get("NEWS_DATA", "old")
if NEWS_DATA not in ("old", "new"):
    raise ValueError("NEWS_DATA must be 'old' or 'new', got %r" % NEWS_DATA)
DATA_SFX = "" if NEWS_DATA == "old" else "_newsv2"

PCA_SOLVER = os.environ.get("PCA_SOLVER", "auto")

RES = Path(os.environ.get("RESULT_DIR", str(BASE)))
RES.mkdir(parents=True, exist_ok=True)


def make_pca(n_components: int, random_state: int):
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=random_state, svd_solver=PCA_SOLVER)
