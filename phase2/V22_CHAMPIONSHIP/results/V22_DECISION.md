# V22 Decision

## Evaluation against Acceptance Gate
| Metric | V21 Baseline | V22 Requirement | V22 Actual | Status |
|---|---|---|---|---|
| Total Score | 50.00 | Must exceed | 50.36 | PASS |
| Set B <= 5px | 0.071 | No degradation | 0.071 | PASS |
| PRESENT recall | 0.457 | >= V21 | 0.285 | FAIL |
| Rejection F1 | 0.354 | >= V21 | 0.352 | FAIL |
| Calibration AUC | 0.521 | >= V21 | 0.592 | PASS |
| Median runtime | < 5s | < 5s | < 5s | PASS |

## Conclusion
Although V22 improved the Total Score by ~0.36 points (primarily due to better calibration and pose) and improved Calibration AUC, it failed to meet the strict acceptance gates for PRESENT recall and Rejection F1. Specifically, the PRESENT recall degraded severely from 45.7% to 28.5%, indicating that the new fusion model is still aggressively over-rejecting true positives on the held-out test split, even at a low optimized threshold of 0.10.

Following the final decision rules, since V22 does not beat V21 robustly across all gates:
REJECT
