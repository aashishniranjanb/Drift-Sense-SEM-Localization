# Optimized plan — reach the dev-set maximum with the V25 model FROZEN

**Scientific baseline:** Engine B (original V25, cache-free, live) = **53.4 / 100**.
19 A + 9 B localized. Every prior figure ≥ ~55 was cache / accepted-only scoring
/ Spearman-not-AUC / projection.

**Frozen (never modified):** V25 candidate extraction (`extract_candidates_akhilesh`),
V25 `ranker.pkl`, V25 `presence.pkl` internals, `FINAL_SUBMISSION/` production,
`FINAL_SUBMISSION_GOLDEN/`. Everything below is an **added layer** on the anchor.

**Where the points are** (STEP 9): 62 ranking failures, 4 shallow-retrieval
failures, 31 hard-retrieval failures. Local V25 features are exhausted
(all ~50 % GT-vs-replica). Need a global signal.

---

## Layer 1 — Global Alignment Re-ranker (STEP 10)  →  target +20–25 pts

Re-order the **top-K candidates the V25 ranker already produced** (K = 20;
covers GT for 44 of the 62 R2 + all 43 R1). Do **not** change extraction or the
V25 ranker. For each of the K candidates compute a global-alignment evidence
vector and pick the argmax of a transparent score.

Evidence per candidate `(x, y, scale, theta)`:
1. **Whole-patch aligned residual** — warp the full reference into the candidate
   frame (or the candidate neighbourhood into reference coords); `whole_ncc`,
   `whole_gradient_ncc`, `1 - SSIM`, `edge_distance_transform` residual over the
   *entire* overlap, not a crop.
2. **Multi-ring residual** — split the aligned patch into core / mid / outer
   rings; `ring_ncc[0..2]`, and `ring_falloff = core_ncc - outer_ncc`
   (large ⇒ replica).
3. **Constellation / landmark consistency** — pick 6–10 high-gradient landmarks
   in the reference, locate each near the candidate, fit one similarity
   transform; `landmark_inlier_count`, `landmark_rms_residual`,
   `pairwise_distance_consistency`, `orientation_consistency`.
4. **Competitor deltas** — vs the strongest *other* candidate:
   `d_whole`, `d_ring_falloff`, `d_landmark_inliers`, `d_edge`.
5. Carry `v25_rank`, `v25_score`, `raw_ncc` as tie-breakers only.

**No ML first.** Score = weighted sum with hand-set signs from the STEP-9
`ranking_failures.csv` win-rates; sweep weights on the R2/R1/absent split.

### Promotion gate (STEP 12, hard)
Compare against the Engine-B anchor on: 62 R2 pairs · 43 R1 (baseline
successes) · 40 absent.

Promote **only if all** hold:
- ≥ 5 additional pairs reach ≤ 5 px (aim ≥ 25)
- **0** R1 baseline successes broken
- **0** new absent false positives
- median runtime ≤ 5 s/pair
- deterministic (byte-identical on re-run)

Fail any → delete the layer, revise, retry. No partial promotion.

If it clears: re-run all 180, official scorer, update the matrix.

---

## Layer 2 — Deeper / multi-hypothesis retrieval for R3+R4  →  target +5–8 pts

Only after Layer 1 is promoted. For pairs the anchor rejects OR whose Layer-1
top candidate is weak:
- widen NMS to depth 600 / r = 3 (fixes the 4 R3);
- carry the top 3 scale × 2 rotation hypotheses into extraction
  (`matcher.multi_hypothesis_search` already exists) — earlier probe recovered
  ~50 % of a comparable hard set into the ≤ 5 px pool.
- Feed the widened pool through Layer 1's global re-ranker (which is what makes
  a wide pool safe — it was the missing discriminator).

Same promotion gate. Watch runtime: only spawn extra hypotheses when the
periodicity mode is STRONG or the Layer-1 top margin is thin.

---

## Layer 3 — Presence gate on the re-ranked evidence  →  target +3–5 pts (rejection + calibration)

Only after Layers 1–2. The `found` decision must ask two questions separately:
"is there a strong candidate" vs "is the strongest candidate a periodic replica".
Feed the Layer-1 winner's global-alignment evidence + `ring_falloff` +
`landmark_inliers` + competitor deltas into a presence score. Fit **only on
self-generated adversarial synthetic data** (Layer 4) — never on dev labels.
Threshold swept on the official score.

`score` column = `P_present × P_correct_given_present`, monotone; evaluated by
**ROC AUC**.

---

## Layer 4 — Adversarial synthetic laboratory (training data only)

The generator is **not** the scoring target (10 pts); it is the only legal
source of training data for Layers 1–3 (dev labels are prohibited for fitting).
Build `generate_dataset.py` to the dataset-prompt spec (`z ∈ [8,12]`,
`θ ∈ ±5°`, 20 % absent, A/B/C/D, single affine, §5 verification gate) and emit
1–5 k **candidate-comparison** examples: for each synthetic pair, the GT
candidate + its hardest periodic replicas + boundary / wrong-phase decoys, with
the full global-alignment evidence vector. Split by seed family; untouched final
holdout. Simple models first (LogReg → HGB), promote on generalization not fit.

---

## Realistic maximum on `data/phase2_dev` (V25 model frozen)

| Milestone | Localization | Total (approx) |
|---|---:|---:|
| Engine B anchor (now) | 4.7–7.1 / 40 | **53** |
| + Layer 1 (global re-rank, ~90–105 localized) | 22–27 / 40 | **78–83** |
| + Layer 2 (R4 retrieval, ~10–20 recovered) | 27–31 / 40 | **86–90** |
| + Layer 3 (presence + AUC calibration) | — | **88–92** |
| + Set-D bonus (if A–C ≥ 0.50 and D ≥ 0.40) | — | **+6** |

**96 is not reachable on this dev set with the frozen V25 model** — 31 pairs
have no retrievable ≤ 5 px candidate at any depth. If those are §5-broken labels
(likely, given the set was not built with the verification gate), a verified
synthetic validation set removes them and the ceiling rises accordingly; report
both numbers.

---

## Order of operations

1. **STEP 10** — implement Global Alignment Discriminator V1 (Layer 1), no ML.
2. Evaluate on the 62 / 43 / 40 split. Promotion gate.
3. If promoted → full 180 → official scorer → update `ENGINE_RECOVERY_MATRIX.csv`.
4. **STEP 11** — Layer 2 (retrieval widening) behind the same gate.
5. **STEP 12** — Layer 4 generator, then Layer 3 presence, then AUC calibration.
6. Clean-room + SHA-256 + honest report. Only then RGB.

Rules: branch `engine-recovery`, commit every step, **do not push**,
score only with `phase2/V48_MAX/score_phase2_official.py`, localization over all
140 present pairs, calibration by AUC, no dev labels in any fit, no cache, no
`pair_id`, no network.
