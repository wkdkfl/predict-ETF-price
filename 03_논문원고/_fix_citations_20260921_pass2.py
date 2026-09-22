# -*- coding: utf-8 -*-
"""인용 표기 정리 2차 (2026-09-21) — 2장 밖의 서술형 저자 인용을 번호식으로.

1차(`_fix_citations_20260921.py`)에서 '저자, 연도 [번호]' 중복 11곳을 정리했다.
이번에는 저자명이 문장의 주어로 쓰인 곳 중 **2장(문헌 검토)과 §5.2.5(유사 선행연구 비교)를 제외한**
부분을 번호식으로 바꾼다. 두 곳은 연구 간 선후·대립 관계를 서술해야 하므로 저자명을 유지한다.

내용(무엇이 보고되었는가)은 그대로 두고 주어만 바꾸므로 사실관계는 변하지 않는다.

실행: python _fix_citations_20260921_pass2.py [--dry-run]
"""
from __future__ import annotations

import sys
from pathlib import Path

import docx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fix_citations_20260921 import iter_paragraphs, replace_in_paragraph  # noqa: E402

DOC = Path(__file__).resolve().parent / "(국문) 석사학위논문_임우현.docx"

REPLACEMENTS = [
    # §1.1 연구 배경 및 필요성
    ("Arnott 등 [7]은 금융 머신러닝 연구에서 백테스팅 프로토콜의 엄격성이 성능 과대평가를 막는 데 "
     "필수적임을 강조하였고, Kapoor와 Narayanan [8]은 여러 분야에 걸쳐 데이터 누출이 머신러닝 기반 "
     "연구의 재현성 위기를 초래해 왔음을 폭넓게 보고하였다.",
     "금융 머신러닝 연구에서는 백테스팅 프로토콜의 엄격성이 성능 과대평가를 막는 데 필수적임이 "
     "강조된 바 있으며 [7], 여러 분야에 걸쳐 데이터 누출이 머신러닝 기반 연구의 재현성 위기를 "
     "초래해 왔음도 폭넓게 보고되었다 [8]."),
    # §3.5 텍스트 표현: TF-IDF
    ("선행연구에서 Xing 등 [19]은 금융 텍스트 기반 예측의 서베이에서 TF-IDF 등 전통적 벡터화 기법이 "
     "금융 도메인에서 유효함을 보인 적 있으며,",
     "금융 텍스트 기반 예측을 다룬 선행 서베이 [19]에서도 TF-IDF 등 전통적 벡터화 기법이 금융 "
     "도메인에서 유효함이 확인된 바 있으며,"),
    # §3.6 텍스트 표현: FinBERT 아키텍처
    ("Araci [4]의 FinBERT는 이를 금융 감성 분류 데이터로 추가 학습하여",
     "FinBERT [4]는 이를 금융 감성 분류 데이터로 추가 학습하여"),
    # <그림 5> 캡션 (그림 목차 + 본문) 및 자료 표기
    ("(Wolpert[27]의 stacked generalization 개념에 기반하여 재작성)",
     "(stacked generalization 개념 [27]에 기반하여 재작성)"),
    ("자료: Wolpert[27]의 stacked generalization 개념을 바탕으로 저자 작성",
     "자료: stacked generalization 개념 [27]을 바탕으로 저자 작성"),
    # §5.2 본 실험 결과
    ("이는 Fama[11]의 약형 효율적 시장가설과 일치하는 결과이다.",
     "이는 약형 효율적 시장가설 [11]과 일치하는 결과이다."),
    # §5.2.3 학술적 시사점
    ("데이터 누출이 분야 전반의 재현성을 훼손해 왔다는 Kapoor와 Narayanan [8]의 지적과 궤를 같이한다.",
     "데이터 누출이 분야 전반의 재현성을 훼손해 왔다는 지적 [8]과 궤를 같이한다."),
]


def main():
    dry = "--dry-run" in sys.argv
    document = docx.Document(str(DOC))
    total = 0
    for old, new in REPLACEMENTS:
        hits = 0
        for par in iter_paragraphs(document):
            hits += replace_in_paragraph(par, old, new)
        total += hits
        print("%s %d회 | %s" % ("OK " if hits else "못찾음", hits, old[:72]))
    print("\n총 치환 %d회" % total)
    if dry:
        print("(--dry-run: 저장하지 않음)")
        return
    document.save(str(DOC))
    print("저장 완료: %s" % DOC.name)


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
