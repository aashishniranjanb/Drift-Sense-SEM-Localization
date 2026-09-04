# Verified-Generator 90+ — Implementation Plan

**Anchor:** `engine-recovery` @ `b0cea92`. Production `FINAL_SUBMISSION/`
tree-sha256 `48a9f6ff319c6eda` (294 files) — must stay byte-identical.
Baseline: **Engine B = 53.36 / 100** live, cache-free (`ENGINE_RECOVERY_MATRIX.csv`).

## Why this exists

STEP 9/10 proved `data/phase2_dev`'s degraded labels are **not separable by
correlation at any footprint** (replica and true site both ~0.19 whole-patch NCC).
A large fraction are §5-unverified — unhittable by any correct algorithm. We
cannot measure progress on a broken instrument.

**First milestone is not 90.** It is: *can Engine B do substantially better on a
benchmark whose labels we have independently proved are physically recoverable?*

## Architecture decision — procedural canvas, no giant raster

A `z=12` search FOV is `12000²` canvas px (144 MB) if rasterised. Instead the
pattern is a **procedural field** `f(x, y)` evaluable on an arbitrary float grid:

- **search image** = evaluate `f` at `inverse_affine(search_pixel_grid)`, 3×3
  supersampled and area-averaged (true area integration, better than warping —
  satisfies §3.1 without a 4× intermediate);
- **reference image** = evaluate the *same* `f` on a 1000×1000 grid at 1 nm/px
  centred on the target.

Both come from one function under one transform ⇒ geometry is exact by
construction (§3 R1/R2), no memory blow-up, no resampler-induced label drift.

## Geometry (pinned, dataset-prompt §2.2)

```
p_search = (1/z)·R(theta)·(p_canvas − c_canvas) + c_search
R(theta) = [[ cos t, sin t],
            [−sin t, cos t]]        t = radians(theta)
z ∈ [8,12]   theta ∈ [−5,+5]°   scale column in GT = z
```
GT `(x, y)` = the reference crop centre pushed through **that same** transform.
Target location is chosen in *search* coordinates first and pulled back (R4).

## Independent verification (the whole point)

`verify_ground_truth.py` sees **only**: the re-read reference PNG, the re-read
search PNG, and the declared GT. **No generator metadata.** It renders its own
template by a *different* path (box-blur + rotation-matrix warp, not the
generator's supersampler) and measures:

- intensity NCC and gradient NCC at GT
- global correlation peak location and its distance to GT
- peak prominence, nearest competing peak, **GT-vs-competitor margin**
- GT retrieval rank in a deep NMS pool
- recoverability at ≤1 px / ≤2 px / ≤5 px

**Ship rule:** global peak within 3 px of GT **and margin ≥ 0.12** (floor 0.02
for genuinely degraded pairs, flagged). Anything else is **DISCARDED** — never
written to train/val/test.

## Phases & budget (12 hypotheses max)

| Phase | Deliverable | Gate |
|---|---|---|
| 1 | generator + independent verifier | verifier discards broken samples |
| 2 | **pilot 100** (25 DRAM / 25 FinFET / 25 periodic-hard / 25 degradation-hard), Engine B cache-free, Top-K + ≤1/2/5 px | **STOP & ANALYSE — no training** |
| 3 | 2000 train / 500 val / 500 test, disjoint seeds, balanced | failure taxonomy R1–R6 |
| 4 | pick **one** bottleneck from the data | never attack two at once |
| 5–8 | retrieval V2 / candidate discriminator / hard-negative curriculum / protected fallback | promotion gate |
| 9–11 | presence → calibration → 90+ | ≥ 5 recoveries, 0 broken, 0 new absent FP, ≤ 5 s, deterministic |
| 12 | transfer test vs `data/phase2_dev` (diagnostic only, never tuned on) | report gap honestly |

Every 3 experiments → `RESEARCH_CHECKPOINT_N.md`. Registry:
`EXPERIMENT_REGISTRY.md`. Checkpoints under `checkpoints/`.

## Hard rules

- No model trained / tuned / selected on `data/phase2_dev` — diagnostic only.
- No cache, no `pair_id` logic, no historical prediction lookup, no network.
- One canonical scorer: `phase2/V48_MAX/score_phase2_official.py`
  (localization = mean tiered credit over **all** present pairs; calibration = **AUC**).
- "LOCAL VERIFIED SCORE" is never called an official competition score.
- Commit every experiment; never push; production byte-identical.

## Parallelism

5 workers (6 logical cores). Generation, verification and Engine-B evaluation are
all per-sample independent → `ProcessPoolExecutor`. Single-thread BLAS/OpenCV
inside workers (`OMP_NUM_THREADS=1`, `cv2.setNumThreads(1)`).
