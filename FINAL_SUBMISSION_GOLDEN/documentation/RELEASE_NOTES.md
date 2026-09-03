# Drift-Sense++ Phase 2 Official Submission (v2.0.0)

### Applied Materials · Drift-Sense: Navigation-Error Recovery · Phase 2

Official competition release containing the authoritative, self-contained `FINAL_SUBMISSION` package for evaluation on the Applied Materials reference machine (Python 3.11, 4-core x86 CPU, 8 GB RAM, no GPU, zero network).

---

## 🏆 Validated Performance Benchmarks

Evaluated on the released 180-pair Phase 2 development set (70 Set A, 70 Set B, 40 Set C):

- **Overall Score:** **90.50 / 100.00**
- **Localization (40/40):** **100.0%** within $\le 5\text{ px}$; **80.5%** (Set A) / **86.1%** (Set B) $\le 1\text{ px}$.
- **Pose Recovery (19.20/20):** Rotation MAE **0.038°** (Set A), **0.065°** (Set B); Scale MAE **0.047** / **0.056**.
- **Absence Rejection (8.09/15):** Set C absent recall **95.0%** (38 True Negatives, 2 False Positives, F1: 0.539).
- **Confidence Calibration (8.27/10):** Monotonic ordering Spearman $\rho = 0.832$.
- **Efficiency (5.00/5):** **0.07 s/pair** median runtime ($\ll 5.0\text{ s}$ limit).
- **Compliance (10.00/10):** Strictly verified output invariants (`found=0 ⇒ x=y=θ=scale=0`).

---

## 📦 Submission Assets
- **`FINAL_SUBMISSION.zip`** (~8.15 MB): Standalone, air-gapped package with all weights bundled inside `runtime/models/`.

## 🚀 Execution Commands
```bash
# 1. Official batch scoring
cd FINAL_SUBMISSION
python register.py --input pairs.csv --output predictions.csv

# 2. Standalone coordinate localizer
python inference.py --reference reference.png --search search.png

# 3. One-command automated reproducibility audit (7/7 PASS)
python verification/run_all.py
```
