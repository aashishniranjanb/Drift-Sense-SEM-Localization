# Verification report

## Command

```bash
cd FINAL_SUBMISSION
python register.py --input <released 180-pair pairs.csv> --output verification/predictions.csv
```

- Deterministic: re-running the command reproduces `predictions.csv` byte-for-byte.
- 180 pairs in ~13 s wall (0.07 s/pair median) on the 4-core reference-class
  machine, well inside the 5 s median / 20 s hard-timeout budget.
- Single-threaded BLAS + OpenCV. No network. No file writes outside `--output`.

## Schema / contract checks (all pass)

| Check | Result |
|---|---|
| Columns exactly `pair_id,x,y,theta,scale,found,score` | pass |
| One row per input `pair_id`, no duplicates, no missing | 180 / 180 |
| `found ∈ {0, 1}` | pass |
| `found = 0` ⇒ `x = y = theta = scale = 0` | pass (101 rejected rows) |
| No NaN / infinite values | pass |
| `score ∈ [0, 1]`, 68 distinct values | pass |

## Score (official `benchmark_phase2.py` methodology)

| Block | Points | Max |
|---|---:|---:|
| Localization | 40.00 | 40 |
| Pose | 19.20 | 20 |
| Rejection | 8.03 | 15 |
| Confidence calibration | 8.27 | 10 |
| Efficiency | 5.00 | 5 |
| Generator / citations / failure analysis | 10.00 | 10 |
| **Total** | **90.50** | 100 |

Raw metric dump: `score_report.json`. Metric-by-metric discussion and protocol:
`../documentation/VALIDATION.md`.

## Localization detail

| Set | present | localized | ≤1 px | ≤5 px | median |
|---|---:|---:|---:|---:|---:|
| A | 70 | 40 | 80.0 % | 100 % | 0.224 px |
| B | 70 | 36 | 86.1 % | 100 % | 0.189 px |

Every accepted present pair is within 5 px — no periodic-replica acceptances.

## Failure taxonomy

| Mode | Count |
|---|---:|
| Subpixel success (≤1 px) | 63 |
| In-bounds success (1–5 px) | 13 |
| Rejection success (absent) | 38 |
| Presence false negative | 64 |
| Absence false positive | 2 |
| Periodic replica | 0 |

## Sample I/O contract

`sample_pairs/pairs.csv` (3 pairs) → `sample_pairs/predictions.csv`, produced by
the same `register.py` invocation. Confirms the reader resolves relative image
paths against the input CSV's directory and emits the exact output schema.
