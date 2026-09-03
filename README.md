# Drift-Sense++ — SEM Localization

### Applied Materials · Phase 2 Submission

**Entry point:** [`FINAL_SUBMISSION/register.py`](./FINAL_SUBMISSION/register.py)
· **Language:** Python · **Runtime:** CPU only, no network

---

## ➡️ JUDGE START HERE

The authoritative competition submission is one folder:

## **[FINAL_SUBMISSION/](./FINAL_SUBMISSION/)**

| File | Purpose |
|---|---|
| [`register.py`](./FINAL_SUBMISSION/register.py) | Official inference entry point |
| [`requirements.txt`](./FINAL_SUBMISSION/requirements.txt) | Pinned runtime dependencies |
| [`generate_dataset.py`](./FINAL_SUBMISSION/generate_dataset.py) | Documented synthetic SEM pair generator |
| [`failure_analysis.pdf`](./FINAL_SUBMISSION/failure_analysis.pdf) | Required failure analysis (2 pages) |
| [`README.md`](./FINAL_SUBMISSION/README.md) | Complete execution instructions |
| `runtime/` | Inference modules and model weights (bundled) |
| `documentation/` | Method, validation, references |
| `verification/` | `register.py` output + full score breakdown |
| `visuals/` | Supporting technical figures |

### One-command execution

```bash
cd FINAL_SUBMISSION
pip install -r requirements.txt
python register.py --input pairs.csv --output predictions.csv
```

**Output** — one row per `pair_id`:

```
pair_id,x,y,theta,scale,found,score
```

When `found = 0`: `x = y = theta = scale = 0`.

### Measured result (released 180-pair development set)

| Localization | Pose | Rejection | Calibration | Efficiency | Docs | **Total** |
|---|---|---|---|---|---|---|
| 40.00 / 40 | 19.20 / 20 | 8.03 / 15 | 8.27 / 10 | 5.00 / 5 | 10.00 / 10 | **90.50 / 100** |

Set A & B localization ≤ 5 px: **100 %**. Median runtime **0.07 s/pair**.
See [`FINAL_SUBMISSION/documentation/VALIDATION.md`](./FINAL_SUBMISSION/documentation/VALIDATION.md).

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
(ΔNCC < 0.005). The system therefore ranks candidates on structural evidence —
multi-scale context, phase consistency, gradient agreement, replica-family
population — rather than trusting the strongest peak.

## 4. Failure analysis

Full analysis: [`FINAL_SUBMISSION/failure_analysis.pdf`](./FINAL_SUBMISSION/failure_analysis.pdf).
Main failure classes: periodic-replica confusion · degraded true-instance miss ·
absent-image false positive · confidence-ordering collapse · incorrect-site
pose estimation.

## 5. Reproducibility

The submission is self-contained. No runtime network access. Judges should
execute only the code inside `FINAL_SUBMISSION/`. Inference is deterministic.

## 6. Research history

Everything outside `FINAL_SUBMISSION/` is historical: phase-by-phase
experiments, ablations, rejected approaches, and diagnostic tooling, retained
for scientific transparency. It is **not** required to run or score the
submission. See [`RESEARCH_ARCHIVE.md`](./RESEARCH_ARCHIVE.md) for a map.

## 7. Team

Per-workstream audit notes: [`FINAL_SUBMISSION/team/`](./FINAL_SUBMISSION/team/).

## License

MIT — see [`LICENSE`](./LICENSE).
