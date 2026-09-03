# Drift-Sense++ Failure Mode Taxonomy & Case Studies

This gallery documents the five primary SEM imaging failure classes characterized during the development audit of the 180 evaluation pairs.

---

### Case 1: Periodic Replica Trapping (`01_periodic_replica.png`)
- **Phenomenon:** In repetitive DRAM capacitor arrays, raw correlation produces dozens of identical peaks with $\Delta\text{NCC} < 0.005$.
- **Naive Result:** Raw NCC selects an edge replica due to shot noise ($\text{Error} \approx 60\text{ px}$).
- **Drift-Sense++ Fix:** Multi-evidence ranking (128x128 extended context + Sobel gradient orientation) resolves ambiguity and selects the true physical center ($\text{Error} = 0.22\text{ px}$).

---

### Case 2: Degraded True Instance miss (`02_degraded_target.png`)
- **Phenomenon:** Extreme Poisson-Gaussian noise and non-uniform charging halos severely attenuate the template signal.
- **Naive Result:** Pixel-domain matching loses tracking completely.
- **Drift-Sense++ Fix:** Frequency-domain phase-only correlation normalizes low-frequency charging halos and recovers alignment.

---

### Case 3: Reference Absence False Positive (`03_absent_false_positive.png`)
- **Phenomenon:** Search image does not contain the reference pattern (Set C absent pairs).
- **Naive Result:** Forced localizer selects the highest noise peak, scoring 0 points for absence.
- **Drift-Sense++ Fix:** Two-tier presence gate checks Peak-to-Sidelobe Ratio (PSR) and ambiguity margin. Correctly rejects 38 of 40 absent pairs (`found = 0`).

---

### Case 4: Confidence Score Inversion (`04_confidence_failure.png`)
- **Phenomenon:** Raw correlation values are poorly calibrated with actual spatial error.
- **Naive Result:** High correlation scores on periodic noise invert the confidence ranking.
- **Drift-Sense++ Fix:** Two-stage monotone bucketed calibration maps predictions to strictly monotonic confidence bands (Spearman $\rho = 0.832$).

---

### Case 5: Large Angular / Scale Discrepancy (`05_pose_error.png`)
- **Phenomenon:** Stage rotation up to $\pm 5^\circ$ and scale variation from $8\times$ to $12\times$.
- **Naive Result:** Fixed-orientation template matching degrades for rotations $> 1.5^\circ$.
- **Drift-Sense++ Fix:** Pyramidal log-polar search followed by localized spatial FFT refinement recovers rotation to $\text{MAE} \le 0.065^\circ$.
