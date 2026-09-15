# Phase 1 사실검증표 (v26 → v27) — 2026-07-15

권위 결과셋: 헤드라인 `results_v3/`, Ablation·DM `results_ar_ablation/`, 수익률/동시간대 `results/`+notebook.

## A. 검증되어 정확 (수정 불필요)
| 항목 | 초록/본문 값 | 실제 근거 | 판정 |
|---|---|---|---|
| 변동성 R²(US, RF) | 0.830 | results_v3 vol_model_comparison r2_log=0.8295 | ✓ |
| 변동성 R²(UK, RF) | 0.932 | r2_log=0.9316 | ✓ |
| 방향성 AUC(US, RF) | 0.749 | dir_model_comparison auc=0.7492 | ✓ |
| 방향성 AUC(UK, LGBM) | 0.844 | auc=0.8441 | ✓ |
| WF 평균 R²(US/UK) | 0.844 / 0.930 | walk_forward_vol 평균 0.8440 / 0.9301 | ✓ |
| WF 평균 AUC(US/UK) | 0.733 / 0.806 | walk_forward_dir 평균 0.7327 / 0.8055 | ✓ |
| 수익률 예측 실패 | R²<0.03 | results/model_comparison 최고 0.026(US)/0.001(UK) | ✓ |
| 동시간대 누출 | R²≈0.65 | US notebook Linear R²=0.6496; ΔR²≈0.62~0.66 | ✓ |

## B. 정정 필요 (요구사항 10)
### F3 — 변동성 R² 신뢰구간 (재계산 완료)
- 초록 기존값(0.806–0.851 / 0.922–0.941)은 어떤 산출물에도 없음(커밋 CI는 원단위 R²·음수).
- **재계산(결정적 RF, block bootstrap n=5000, block=10, SEED=42, 로그-R² 기준)**:
  - US: R²=0.830, **95% CI [0.783, 0.876]** (n_test=392)
  - UK: R²=0.932, **95% CI [0.905, 0.974]** (n_test=259)
  - 저장: `USD|UK/results_v3/volatility_logr2_ci.csv`
- → 초록·본문 CI를 위 재현값으로 교체.

### F1·F4 — 뉴스/FinBERT 기여 (데이터대로 정정)
근거: `results_ar_ablation/ablation_vol.csv | ablation_dir.csv | ablation_dm.csv`
| 그룹 | US vol r2_log | UK vol r2_log | US dir AUC | UK dir AUC |
|---|---|---|---|---|
| AR_Only | 0.834 | 0.939 | 0.724 | 0.779 |
| News_Pure | −0.218 | −0.031 | 0.508 | 0.567 |
| AR+News | 0.821 | 0.927 | 0.730 | 0.820 |
| Full | 0.736 | 0.910 | 0.749 | 0.832 |
- **변동성**: AR_Only 단독이 Full과 대등하거나 우수, News_Pure는 음수 → 변동성 예측력은 **자기회귀(과거 변동성) 피처가 지배**. 뉴스는 변동성에 유의한 개선 없음(DM: US-vol Fin-vs-Full p=0.085 ns).
- **방향성**: 뉴스 추가 시 AUC 상승(UK 0.779→0.832), News_Pure-vs-Full DM 유의(US p=0.000, UK p=0.0042) → 뉴스/FinBERT는 **방향성에서 통계적으로 유의한 증분 기여**. 단, 뉴스 단독은 거의 무작위(0.51/0.57).
- → 기존 "다중 헤드라인 FinBERT가 주된 예측 원천" → **"변동성은 AR 지배, 뉴스는 방향성에서 유의한 증분 기여"** 로 정정.

### F2 — DM "두 시장·두 표적 모두 p<0.01" 은 거짓
근거 `ablation_dm.csv` (Financial_Only vs Full):
| | US | UK |
|---|---|---|
| 변동성 | p=0.085 (ns) | p=0.000 (유의) |
| 방향성 | p=0.000 (유의) | p=0.154 (ns) |
- → "두 시장·두 표적 모두 유의"는 거짓. **US-방향성·UK-변동성에서 유의**로 정확히 재기술.

### F5 — 초록 내부 모순
- para42 "News_Only가 Full의 ~90%(변동성)/~100%(방향성) 재현" ↔ para53 "뉴스+자기회귀가 ~90% 재현" 불일치, 또한 위 ablation과 상충.
- → F1·F4 정정 서사로 통일(변동성=AR 지배, 방향성=뉴스 증분). "~90%/~100%" 문구 삭제/재기술.
