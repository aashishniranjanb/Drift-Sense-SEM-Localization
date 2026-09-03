# V48 BASELINE — IMMUTABLE

**Baseline = V41 FINAL** = `phase2/V41_CALIBRATION/FINAL/v41_predictions.csv`
(chain: V25 localization -> V28-C gate -> V39 pose refinement -> V41 residual-mix calibration)

- Interpreter: Python 3.14 / sklearn 1.8.0
- GT: `data/phase2_dev/pairs.csv` — 180 pairs (70 SetA / 70 SetB / 40 SetC)
- Scorer: `phase2/benchmark_phase2.py` rollup (per `phase2/V27_FINAL/SCORER_AUDIT.md`)
- Reproduced twice; both give identical numbers.

## TOTAL: 88.39 / 100

| Component | Score | Max |
|---|---|---|
| Localization | 40.00 | 40 |
| Pose | 19.20 | 20 |
| Rejection | 8.03 | 15 |
| Calibration | 6.16 | 10 |
| Efficiency | 5.00 | 5 |
| Compliance/Docs | 10.00 | 10 |

## Taxonomy (benchmark_phase2.py)
63 SUBPIXEL_SUCCESS + 13 IN_BOUNDS_SUCCESS + 38 REJECTION_SUCCESS + 64 PRESENCE_FALSE_NEGATIVE + 2 ABSENCE_FALSE_POSITIVE = 180. Zero PERIODIC_REPLICA.

## Rejection detail (positive class = found==0)
TP(absent rejected)=38  FP(present rejected)=64  FN(absent accepted)=2 -> F1 0.5352

## Notes
- Localization is scored over *accepted* present pairs only, so the 64 false-negatives
  do not lower it -> 100% -> 40.00. Present recall is 76/140 = 54%.
- An earlier baseline attempt in this folder used `main`'s `register.py` and the V22
  scorer (coverage-weighted) and reported ~59 — that was the wrong pipeline AND the
  wrong scorer. Disregard any 59.x figure. The correct baseline is 88.39.
