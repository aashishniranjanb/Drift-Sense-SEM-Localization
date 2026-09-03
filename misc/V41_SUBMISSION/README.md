# Drift-Sense++ Phase 2 — Final Submission

## Team: Aashish Niranjan B (Laptop 2)

### Pipeline Version: V41 (V25 localization · V39 pose · V41 calibration · V28-C rejection)

---

## Quick Start

```bash
pip install -r requirements.txt
python register.py --input pairs.csv --output predictions.csv
```

## What register.py does

1. **Reads** `pairs.csv` (columns: `pair_id`, `reference_path`, `search_path`, …)
2. For each pair:
   - **RGB branch**: if reference is colour, runs gradient-union cross-correlation
   - **Gray branch**: runs V25 Championship localization pipeline
     - Applies **V28-C hard rejection** (score ≤ 0.873 → `found=0`)
     - Applies **V39 pose refinement** (θ, scale) if module present
     - Applies **V41 calibration**: `cal_score = 0.90×raw + 0.05×top1_score + 0.05×top1_corr`
3. **Writes** `predictions.csv` with columns: `pair_id, x, y, theta, scale, found, score`

### Schema guarantee
| `found` | `x` | `y` | `theta` | `scale` |
|---------|-----|-----|---------|---------|
| `0`     | `0.0` | `0.0` | `0.0` | `0.0` |
| `1`     | localized | localized | estimated | estimated |

## File manifest

```
register.py              — main entry point (this submission)
requirements.txt         — pip dependencies
generate_dataset.py      — synthetic dataset generator
failure_analysis.pdf     — per-failure-mode analysis
README.md                — this file
V25_CHAMPIONSHIP/        — localization pipeline + trained models
  v25_pipeline.py
  periodicity.py
  feature_extractors.py
  ranker.pkl
  presence.pkl
fallbacks/
  pose_fallback.py
  ranking_fallback.py
  rejection_fallback.py
phase2/
  inference_phase2.py
  scale_search.py
  rotation_search.py
  pose_refinement.py
  rejection.py
  calibration.py
  context_matcher.py
  phase_verifier.py
  family_clustering.py
  candidate_ranker.py
  channel_consensus.py
  periodicity_detector.py
  adaptive_peak_detector.py
  conditional_pace.py
  spatial_fingerprint.py
team/akhilesh-localization/
  candidate_extractor.py
```

## Generator

`generate_dataset.py` produces synthetic Phase 2 benchmarks (Set A / B / C)
using deterministic random seeds over the project's FinFET + DRAM canvas model.
```bash
python generate_dataset.py --output-dir ./synth --set-a 70 --set-b 70 --set-c 40
```

## Calibration details (V41)

The V41 calibration resolves a structural tie in the `found=0` group:
- **REJECTION_SUCCESS** (correct absent) and **PRESENCE_FALSE_NEGATIVE** (wrong absent)
  were both assigned `score=0.0`, creating a massive rank tie suppressing Spearman ρ.
- V41 assigns evidence-based non-zero scores to `found=0` pairs, strictly ordering
  correct rejections above false negatives without changing any `found` / `x/y/θ/scale` value.

**Improvement over baseline (V39):**
| Metric    | V39 Baseline | V41 Final | Δ |
|-----------|-------------|-----------|---|
| AUC       | 0.8248      | **0.8690** | +0.0442 |
| Spearman  | 0.5995      | **0.6159** | +0.0164 |
| Loc ≤5px  | 100%        | **100%**   | 0 |
