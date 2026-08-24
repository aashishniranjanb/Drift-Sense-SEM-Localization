# Drift-Sense++ — SEM Localization

### Applied Materials · Phase 2 Submission

**Official scoring entry point:** [`FINAL_SUBMISSION/register.py`](./FINAL_SUBMISSION/register.py)
· **Standalone localizer:** [`FINAL_SUBMISSION/inference.py`](./FINAL_SUBMISSION/inference.py)
· **Runtime:** CPU only, Python 3.11, no network

---

## Getting Started

The reproducible Phase 2 implementation is contained in [`FINAL_SUBMISSION/`](./FINAL_SUBMISSION/).

### Official Evaluation Interface

```bash
cd FINAL_SUBMISSION
pip install -r requirements.txt
python register.py --input pairs.csv --output predictions.csv
```

`predictions.csv` contains one row per `pair_id` with columns `pair_id,x,y,theta,scale,found,score`. When `found = 0`, the pose columns are set to 0.

### Component 2 Standalone Localizer

```bash
cd FINAL_SUBMISSION
python inference.py --reference reference.png --search search.png
# Output format:
# x=<float>
# y=<float>
```

Driven by the same internal engine as `register.py` for single-pair coordinate localization.

### Dataset Generator

```bash
python FINAL_SUBMISSION/generate_dataset.py --architecture DRAM --num-pairs 5 --output-dir ./demo_data
```

Generates synthetic pairs and `ground_truth.csv` with known reference-pattern centers.

See [`FINAL_SUBMISSION/README.md`](./FINAL_SUBMISSION/README.md) for execution details, verification reports, and architecture documentation.


---

## Repository map

| Path | What |
|---|---|
| [`FINAL_SUBMISSION/`](./FINAL_SUBMISSION/) | **Authoritative Phase 2 package.** Entry points, weights, verification, docs. Self-contained. |
| [`FINAL_SUBMISSION/README.md`](./FINAL_SUBMISSION/README.md) | Full execution manual + scoring rubric |
| [`releases/`](./releases/) | Zipped archive of `FINAL_SUBMISSION/` (same content, downloadable) |
| [`misc/`](./misc/) | Archived development scripts, old data dumps, superseded packages |
| [`RESEARCH_ARCHIVE.md`](./RESEARCH_ARCHIVE.md) | Map of everything historical |
| [`Experiments/`](./Experiments/) | All R&D history: `phase2/` (V10–V48), `PHASE_10`–`PHASE_16`, architecture experiments, diagnostics, organizer materials, deck visuals, `full_research.md`. **Not on the execution path.** |

---

## 1. What this is

Drift-Sense++ recovers the location, rotation and scale of a reference
structure inside a degraded SEM search image, and decides whether the structure
is present at all. It targets the Phase 2 conditions: unknown zoom (8–12×),
small unknown rotation (±5°), heavy SEM degradation, **periodic structural
ambiguity**, reference-absent pairs, subpixel accuracy, and calibrated
confidence.

## 2. Pipeline

```
Reference + Search image
      │
      ▼
Coarse-to-fine scale / rotation FFT-NCC        (unknown zoom + rotation)
      │
      ▼
200-candidate generation  +  periodic-replica family clustering
      │
      ▼
Learned candidate ranker  →  learned presence / absence gate   (V25 + V28-C)
      │
      ▼
V39 surgical pose refinement  (local scale + rotation + 2-D paraboloid subpixel)
      │
      ▼
V41 residual-mix  →  V48 graded calibration   (confidence ordering, score only)
      │
      ▼
pair_id, x, y, theta, scale, found, score
```

## 3. Why periodic SEM localization is hard

The global correlation maximum is **not** necessarily the true physical site:
repetitive DRAM/FinFET arrays produce many near-identical replica peaks
(ΔNCC < 0.005). The system ranks candidates on structural evidence —
multi-scale context, phase consistency, gradient agreement, replica-family
population — rather than trusting the strongest peak.

## 4. Measured result (released 180-pair development set)

| Localization | Pose | Rejection | Calibration | Efficiency | Docs | **Total** |
|---|---|---|---|---|---|---|
| 40.00 / 40 | 19.20 / 20 | 8.03 / 15 | 8.27 / 10 | 5.00 / 5 | 10.00 / 10 | **90.50 / 100** |

Set A & B localization ≤ 5 px: **100 %**. Median runtime **0.07 s/pair**. Zero
periodic-replica acceptances. Full breakdown:
[`FINAL_SUBMISSION/documentation/VALIDATION.md`](./FINAL_SUBMISSION/documentation/VALIDATION.md).

## 5. Failure analysis

[`FINAL_SUBMISSION/failure_analysis.pdf`](./FINAL_SUBMISSION/failure_analysis.pdf) (2 pages).
Main failure classes: periodic-replica confusion · degraded true-instance
miss · absent-image false positive · confidence-ordering collapse ·
incorrect-site pose estimation.

## 6. Reproducibility

The submission is self-contained. No runtime network access. Judges execute
only the code inside `FINAL_SUBMISSION/`. Inference is deterministic.

## 7. Research history

Everything outside `FINAL_SUBMISSION/` is historical and not required to run or
score the submission. See [`RESEARCH_ARCHIVE.md`](./RESEARCH_ARCHIVE.md).

## License

MIT — see [`LICENSE`](./LICENSE).

<!-- Core localization updates applied -->
