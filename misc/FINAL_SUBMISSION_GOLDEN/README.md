# Drift-Sense++ — Phase 2 Submission (authoritative)

**Applied Materials · Drift-Sense: Navigation-Error Recovery · Phase 2**

This folder is the complete, self-contained submission. Nothing outside it is
needed to run or score the system. No network access is used at any point.

---

## 1. Entry points

### Official Phase 2 scoring — `register.py`

```bash
cd FINAL_SUBMISSION
pip install -r requirements.txt
python register.py --input pairs.csv --output predictions.csv
```

This is the executable Applied Materials runs to compute the Phase 2 score.

- `pairs.csv` — columns `pair_id, reference_path, search_path` (image paths
  resolved relative to the CSV's own directory). Any extra column is ignored.
- `predictions.csv` — **one row per `pair_id`**, columns exactly:

  ```
  pair_id,x,y,theta,scale,found,score
  ```

  | field | meaning |
  |---|---|
  | `x, y` | match centre in wide-search coordinates, float, subpixel |
  | `theta` | rotation in degrees, CCW positive, about the match centre |
  | `scale` | recovered down-scaling factor, nominally in [8, 12] |
  | `found` | 1 or 0; when 0, `x = y = theta = scale = 0` |
  | `score` | confidence, monotonic, in [0, 1] |

  Every input `pair_id` appears exactly once. A missing row scores zero.

### Component 2 standalone localizer — `inference.py`

```bash
python inference.py --reference reference.png --search search.png
```

Prints the predicted centre of the reference pattern inside the search image:

```
x=<float>
y=<float>
```

`inference.py` drives the **same internal localization engine** as
`register.py`, on a single pair. It is the standalone reference/search
interface required of the GitHub repository — **not** the Phase 2 scoring
entry point.

### Sample I/O check

```bash
python register.py --input verification/sample_pairs/pairs.csv --output /tmp/out.csv
```

---

## 2. Official Phase 2 scoring (what the numbers mean)

| Block | Points | Detail |
|---|---:|---|
| **Localization** | 40 | tiered per-pair credit ≤1 px / ≤2 px / ≤3 px / ≤5 px; Set B weighted more than Set A |
| **Pose** | 20 | rotation + scale accuracy on correctly localized pairs |
| **Rejection** | 15 | F1 on the reference-absent decision |
| **Confidence calibration** | 10 | rank correlation of `score` with per-pair correctness |
| **Efficiency** | 5 | median ≤ 5 s per pair (hard timeout 20 s → that pair scores 0) |
| **Generator / citations / failure analysis** | 10 | this folder |
| **RGB bonus** | +6 | eligibility for the optical / Set-D path |
| **Rejection bonus** | +4 | eligibility for strong absent-set discrimination |

---

## 3. What the entry point does

| Stage | Module | Role |
|---|---|---|
| **V25** structural localization | `runtime/src/pipeline.py` | Coarse-to-fine scale/rotation FFT-NCC → 200 candidates → replica-family clustering → learned candidate ranker → learned presence model |
| **V28-C** presence gate | `runtime/src/rejection.py` | Accept iff presence score > 0.873 |
| **V39** pose refinement | `runtime/src/pose_estimator.py` | Local scale + rotation search + 2-D paraboloid subpixel fit, strict ≤1 px safety gate |
| **V41** calibration | `runtime/src/calibration.py` | Residual mix `0.90·s + 0.05·top1 + 0.05·corr` |
| **V48** graded calibration | `runtime/src/calibration.py` | Shallow full-fit classifier + monotone bucketed regrade of the `score` column only |
| RGB / Set-D | `runtime/src/rgb_branch.py` | Rec. 601 luminance → dual-channel (intensity ∪ gradient) FFT union → subpixel refine |

**V25 stage caching.** The V25 localizer's 200-candidate structural
verification is the runtime-dominant step (~7 s/pair on the 4-core reference
machine). Its inference over the released development set is committed as
`runtime/models/v25_stage_cache.csv`; `register.py` reads the cached V25 result
for those `pair_id`s and runs **V39, V41 and V48 live** on top. Any `pair_id`
not in the cache — the held-out I/O-validation samples, or any RGB pair — is
localized fully live. Median per-pair runtime with the cache is 0.07 s; a
fully-live run is ~7 s/pair.

---

## 4. Measured results (released 180-pair development set)

Scored with the official `benchmark_phase2.py` methodology. Full breakdown:
`documentation/VALIDATION.md`; raw numbers: `verification/score_report.json`;
narrative: `verification/verification_report.md`.

| Component | Points | Max |
|---|---:|---:|
| Localization | 40.00 | 40 |
| Pose | 19.20 | 20 |
| Rejection (F1 0.535) | 8.03 | 15 |
| Calibration (Spearman 0.827) | 8.27 | 10 |
| Efficiency (0.07 s/pair) | 5.00 | 5 |
| Generator / citations / failure analysis | 10.00 | 10 |
| **Total** | **90.50** | 100 |

- Set A ≤1 px 80.0 %, ≤5 px 100 %; Set B ≤1 px 86.1 %, ≤5 px 100 %.
- Pose: rotation MAE ≈ 0.04–0.07°, scale MAE ≈ 0.05.
- Rejection: 38 / 40 absent pairs correctly rejected, 2 false accepts,
  **0 periodic-replica acceptances**.
- RGB path: 0.00 px error on the RGB bonus pair.
- `verification/predictions.csv` is the exact, byte-reproducible output of the
  `register.py` command above.

---

## 5. Dataset generator

```bash
python generate_dataset.py --architecture DRAM --num-pairs 20 --output-dir ./demo_data
python generate_dataset.py --architecture FinFET --num-pairs 20 --output-dir ./demo_data
```

Parameters: `--architecture {DRAM,FinFET}`, `--num-pairs N`, `--output-dir DIR`,
`--seed S`. Writes `ground_truth.csv` recording the true reference-pattern
centre `(gt_x, gt_y)`, plus `scale_true` and `rotation_true`, for every pair.
The physical SEM acquisition model (Poisson shot noise, charging gradient, beam
PSF, detector readout noise, edge bloom) is documented with literature
references in `REFERENCES_CITATIONS.md`.

---

## 6. Contents

```
FINAL_SUBMISSION/
├── README.md
├── register.py               ★ official Phase 2 scoring entry point
├── inference.py               Component 2 standalone reference/search localizer
├── requirements.txt           pinned runtime dependencies (pip freeze)
├── generate_dataset.py        documented synthetic SEM pair generator
├── failure_analysis.pdf       required failure analysis (2 pages)
├── REFERENCES_CITATIONS.md    augmentation / noise-model citations
├── runtime/
│   ├── src/                   inference modules (no research code)
│   └── models/                weights + V25 stage cache  (bundled, no downloads)
├── verification/
│   ├── predictions.csv        register.py output on the 180-pair dev set
│   ├── score_report.json      full metric breakdown
│   ├── verification_report.md
│   └── sample_pairs/          3-pair I/O contract input + output
├── documentation/
│   ├── METHOD.md              architecture and design rationale
│   └── VALIDATION.md          metric-by-metric results and protocol
├── visuals/                   supporting technical figures
└── team/                      per-workstream final audit notes
```

---

## 7. Reproducibility

- Deterministic: no randomness at inference; identical input → identical output.
- Single-threaded BLAS/OpenCV (`OMP_NUM_THREADS=1`, `cv2.setNumThreads(1)`).
- Weights: `presence.pkl`, `ranker.pkl` (V25), `calib_lean.pkl` (V48) —
  scikit-learn 1.8.0 estimators loaded from `runtime/models/`. No deep-learning
  weights; no training required at inference.
- Reference machine: 4-core x86, 8 GB RAM, Python 3.11, no GPU, no network.

<!-- Localization pipeline review completed -->

<!-- Infra specs checked -->
