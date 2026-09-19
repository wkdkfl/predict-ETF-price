# -*- coding: utf-8 -*-
"""재실험 러너 — variant 별로 v10 파이프라인 전 단계를 실행하고 결과를 재실험_결과/<variant>/ 에 저장한다.

variant
-------
  oldnews_full  기존(희소) 뉴스 + PCA 솔버 full     ┐ 주 비교쌍: 같은 환경·같은 절차에서 뉴스만 다르다
  newsv2_full   신규(일별) 뉴스 + PCA 솔버 full     ┘
  oldnews_auto  기존 뉴스 + PCA 솔버 auto (scikit-learn 기본)   ┐ 잡음 하한 측정: PCA 구현 세부만 다르다
  newsv2_auto   신규 뉴스 + PCA 솔버 auto                        ┘

PCA 솔버를 둘로 나눈 이유: auto 는 scikit-learn 버전에 따라 내부 구현이 달라 뉴스 포함 피처군의 R² 가
최대 ±0.05 변한다(2026-09-20 실측). full 은 결정적이라 버전에 무관하며, 주 결과는 full 로 보고한다.

실행 순서는 README 의 3~9단계와 같다(1·2단계는 시장 데이터 고정 원칙에 따라 재실행하지 않음).

사용법
------
  python _run_reexperiment.py                      # 4개 variant 전부
  python _run_reexperiment.py --only oldnews_full newsv2_full
  python _run_reexperiment.py --steps _final_v10.py
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT_ROOT = BASE / "재실험_결과"

VARIANTS = {
    "oldnews_full": {"NEWS_DATA": "old", "PCA_SOLVER": "full"},
    "newsv2_full": {"NEWS_DATA": "new", "PCA_SOLVER": "full"},
    "oldnews_auto": {"NEWS_DATA": "old", "PCA_SOLVER": "auto"},
    "newsv2_auto": {"NEWS_DATA": "new", "PCA_SOLVER": "auto"},
}
STEPS = [
    "_final_v10.py",              # 변동성 피처군별 R² + DM
    "_final_v10_direction.py",    # 방향성 AUC + DM
    "_save_folds_v10.py",         # 8-fold Walk-Forward
    "_garch_aligned_v6.py",       # HAR-RV / GARCH / GJR (동일 표본·분할)
    "_pooled_calibrated_v9.py",   # pooled vs 단독 학습
    "_robustness_v13.py",         # FTSE 250 대체 표적 + fold 설명변수
    "_figs_8_10_v10.py",          # 그림 8·10 + 방향성 예측값
]


def pkg_versions():
    out = {}
    for m in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "arch", "torch", "transformers"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def run_variant(name: str, steps: list[str]):
    cfg = VARIANTS[name]
    res = OUT_ROOT / name
    (res / "logs").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8", EXTRA_CLEAN_FIN="1",
               RESULT_DIR=str(res), FIG_OUT=str(res), **cfg)
    manifest = {"variant": name, "settings": cfg, "python": platform.python_version(),
                "packages": pkg_versions(), "steps": []}
    for step in steps:
        t0 = time.time()
        with open(res / "logs" / (Path(step).stem + ".log"), "w", encoding="utf-8") as lf:
            rc = subprocess.run([sys.executable, step], cwd=BASE, env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        manifest["steps"].append({"step": step, "returncode": rc, "seconds": round(dt, 1)})
        print("[%s] %-28s rc=%d  %.0fs" % (name, step, rc, dt), flush=True)
        (res / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return all(s["returncode"] == 0 for s in manifest["steps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", choices=list(VARIANTS))
    ap.add_argument("--steps", nargs="+", choices=STEPS)
    a = ap.parse_args()
    ok = True
    for v in (a.only or list(VARIANTS)):
        ok &= run_variant(v, a.steps or STEPS)
    print("\n전체", "성공" if ok else "일부 실패 — logs 확인")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
