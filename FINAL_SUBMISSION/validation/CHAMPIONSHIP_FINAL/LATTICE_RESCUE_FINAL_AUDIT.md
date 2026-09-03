# LATTICE RESCUE V1 — CHAMPIONSHIP FINAL SHADOW AUDIT

**Golden Baseline:** 91.040 / 100.00
**Target:** 22 near-miss retrieval failures (pool best candidate 5-10px from GT)
**Execution Rule:** ZERO production files modified. Golden baseline untouched.

---

## Final Results Summary

| Metric | Value |
|---|---|
| **22 Retrieval Recoveries** | **1 / 22** |
| **76 Success Regressions** | **3 / 76 BROKEN** — gate failure |
| **Rescue Applied (shadow)** | 2 pairs |
| **Localization Score** | 40.000 (delta +0.000, rescue was net neutral) |
| **Rejection Score** | 8.085 (delta +0.057, minor improvement from 2 new correct accepts) |
| **Total Score** | 91.097 (delta +0.057) |
| **VERDICT** | **DO NOT PROMOTE** |

> [!CAUTION]
> The mandatory safety rule is triggered: **3 of 76 verified successes were broken** (pair_059, pair_067, pair_092 — all with baseline error <0.25 px were pushed to 28–190 px error). Any configuration that breaks verified ≤5px localizations without zero-regression safety is immediately rejected.

---

## Promotion Condition Checklist

| Condition | Status |
|---|---|
| 1. total > 91.040 | YES (+0.057) |
| 2. zero >5px regressions among 76 | **NO — 3 broken** |
| 3. no new false accepts | YES |
| 4. runtime within limits | YES (234s / 138 pairs) |
| 5. single-file scorer confirms improvement | YES (but condition 2 fails) |

---

## 22-Case Recovery Audit

| pair_id | base_err | pool_min | final_err | lat_conf | n_rescue | status |
|---|---|---|---|---|---|---|
| pair_000 | 242.50 | 5.47 | 242.50 | 0.00 | 0 | no_change |
| pair_014 | 279.05 | 5.68 | 279.05 | 0.00 | 0 | no_change |
| pair_018 | 209.10 | 7.67 | 184.33 | 0.50 | 71 | improved (+) |
| pair_024 | 179.66 | 5.03 | 288.87 | 0.50 | 63 | DEGRADED (-) |
| pair_030 | 199.20 | 5.89 | 207.31 | 0.50 | 75 | no_change |
| pair_034 | 23.17 | 6.78 | **113.98** | 0.50 | 63 | **DEGRADED (-)** |
| pair_036 | 223.28 | 6.45 | 223.28 | 0.00 | 0 | no_change |
| pair_040 | 175.70 | 5.19 | 213.31 | 0.75 | 67 | DEGRADED (-) |
| pair_042 | 268.06 | 5.42 | 268.06 | 0.00 | 0 | no_change |
| pair_046 | 200.77 | 9.01 | 212.21 | 0.50 | 63 | no_change |
| **pair_058** | 101.72 | 6.18 | **0.26** | 0.75 | 59 | **RECOVERED** |
| pair_060 | 57.47 | 7.82 | 46.58 | 0.50 | 46 | improved (+) |
| pair_068 | 49.86 | 5.51 | 14.16 | 0.50 | 37 | improved (+) |
| pair_076 | 178.16 | 5.55 | 178.16 | 0.00 | 0 | no_change |
| pair_094 | 176.34 | 6.92 | 286.92 | 0.50 | 48 | DEGRADED (-) |
| pair_102 | 297.97 | 5.22 | 297.97 | 0.00 | 0 | no_change |
| pair_110 | 80.31 | 6.24 | **276.26** | 0.75 | 76 | **DEGRADED (-)** |
| pair_112 | 84.55 | 7.82 | 37.69 | 0.50 | 73 | improved (+) |
| pair_118 | 107.74 | 7.26 | 107.74 | 0.00 | 0 | no_change |
| pair_120 | 61.30 | 7.32 | 40.77 | 0.50 | 44 | improved (+) |
| pair_121 | 116.50 | 7.62 | 116.50 | 0.00 | 0 | no_change |
| pair_127 | 195.70 | 6.02 | 195.70 | 0.00 | 0 | no_change |

**Statistical summary:**
- Lattice found (confidence ≥ 0.25): **14 / 22** pairs
- Lattice not found (confidence = 0): **8 / 22** pairs → zero rescue candidates generated
- RECOVERED to ≤5px: **1 / 22** (pair_058: 101.72 → 0.26 px)
- Improved but not ≤5px: **5 / 22**
- Degraded (rescue made worse): **5 / 22**
- No change: **11 / 22**

---

## Forensic Root Cause: Why Rescue Broke Verified Successes

**The 3 broken success pairs:**
- `pair_059`: base_err = **0.10 px** → final_err = 156.83 px (V25 score: 0.9593)
- `pair_067`: base_err = **0.19 px** → final_err = 28.02 px (V25 score: 0.9203)
- `pair_092`: base_err = **0.24 px** → final_err = 190.84 px (V25 score: 0.9229)

**All 3 broken pairs had extremely accurate baseline answers (< 0.25 px error) and high V25 confidence scores (0.92–0.96).** The rescue gate fired on them because:
1. The local lattice estimator found neighboring peaks in the periodic array
2. A rescue hypothesis landed at a periodic neighbor that happened to have slightly better context/neighborhood scores than the already-correct baseline winner
3. The `is_rescue_decisively_stronger` function passed because it only compares rescue vs baseline signals — it does not consider the V25 confidence level of the existing winner

**The fix is architecturally simple:** add a "leave well enough alone" guard using the V25 calibrated score:
```python
if baseline_v25_score > HIGH_CONFIDENCE_THRESHOLD:  # e.g., 0.90
    skip rescue entirely
```

The 76 success pairs have V25 scores ranging from 0.877 to 0.964 (P50 = 0.9298). A threshold of 0.90 would protect the vast majority of correct answers while still allowing rescue on low-confidence baseline outputs.

---

## Lattice Estimator Analysis

The local lattice estimator **completely failed** on 8 of the 22 target pairs (confidence = 0.00, zero rescue candidates generated). These 8 pairs share a common property: their V25 baseline winner is **far from the GT location** (e.g., pair_000: base=242px), which means the anchor peak used for local lattice estimation is itself a highly incorrect periodic replica. Estimating lattice vectors from a false positive produces no reliable nearby peaks and yields zero confidence.

**This explains the fundamental asymmetry:** pair_058 succeeded because its baseline winner happened to be close enough to the correct region that the local lattice vectors pointed at the GT location.

---

## Engineering Go/No-Go Assessment

Using your stated scale: 0–3 = abandon, 4–7 = weak, 8–11 = promising, 12–15 = strong, 16+ = championship-grade:

**1 / 22 recovered = "ABANDON this approach in its current form"**

However, pair_058's **0.26 px perfect recovery** proves the physical mechanism is correct when the anchor is in the right neighborhood. The failure is in anchor selection and the missing V25 confidence gate.

---

## Strategic Decision Tree

The evidence now satisfies the decision rule stated in the championship plan:

> "If no → freeze the 91.04 grayscale anchor and move directly to the RGB bonus/compliance track"

**Recommended action: Move to the RGB bonus/compliance track.**

The grayscale system is locked at **91.040**. The current audit ladder shows:

| System | Score |
|---|---|
| Locked golden baseline (V25 + V28-C + V39 + V54 + calib_lean) | **91.040** |
| Theoretical top-200 re-rank ceiling | ~93.44 |
| Theoretical full retrieval + re-rank ceiling | ~96.23 |

Three separate experiments (RERANK-V2, RERANK-V3, LATTICE-RESCUE-V1) have all been safely tested and safely rejected. The baseline is intact.

> [!IMPORTANT]
> The +6 RGB bonus verified independently against organizer requirements would already bring the total to **~97.04**, which substantially exceeds the 96+ target without any further grayscale optimization. This is now the highest-ROI remaining path.
