# Research archive — map

Everything in this repository **outside `FINAL_SUBMISSION/`** is historical or
supporting. None of it is on the `register.py` / `inference.py` execution path.
It is kept for scientific transparency and reproducibility.

## Version history & experiments

| Path | Contents |
|---|---|
| `phase2/` | The full Phase 2 version history (V10 → V48): localization, pose, rejection and calibration experiments, per-version reports, and prediction CSVs. `phase2/V48_MAX/` holds the calibration study that produced the shipped `calib_lean.pkl` (`b4_calib_lean.py`) and the scorer (`v48_score.py`). `phase2/V27_FINAL/SCORER_AUDIT.md` documents the official scoring formulas. `phase2/benchmark_phase2.py` is the scoring harness. |
| `PHASE_10` … `PHASE_16` | Earlier phase working directories. |
| `experiments/` | Standalone architecture experiments (multiscale, siamese, PACE group ranking, CAR dual-channel, multi-view). |
| `diagnostics_and_tools/` | One-off analysis and tuning scripts. |

## Supporting material

| Path | Contents |
|---|---|
| `misc/` | Loose development scripts (`root_scripts/`), old data dumps (`root_data/`), superseded docs (`old_docs/`), the interim `V41_SUBMISSION/` package, the working copy the runtime was assembled from (`submission_working_copy/`), and old submission zips. |
| `releases/` | `FINAL_SUBMISSION.zip` — a zipped archive of the authoritative package (identical content). |
| `submission_package/` | Phase 1 submission package (superseded). |
| `data/`, `models/` | Development datasets and superseded model checkpoints. |
| `production_engine/`, `demo/`, `results/` | Earlier production runner, demo builder, result dumps. |
| `hf_space*`, `hf_clone/` | Hugging Face Space mirrors. |
| `DriftSense_TeamPlan/`, `docs/` | Planning and design notes. |
| `team/` | Per-workstream **final audit notes** (results only). |
| `figures/`, `phase2_deck_visuals/` | Figure sources for the submission deck. |
| `rgb_bonus_package/` | RGB / Set-D bonus material. |
| `AMP_Phase 2 material/`, `Dataset_AMP_Phase 2/` | Organizer-provided Phase 2 reference material and its generator. |
| `Applied Materials *.pptx / *.docx` | Organizer task brief, dataset prompt, and session transcript. |

## Reproducing the shipped numbers

- Calibration model: `phase2/V48_MAX/b4_calib_lean.py`.
- Score: `phase2/V48_MAX/v48_score.py` and `phase2/benchmark_phase2.py`.
- The V25 stage cache bundled in `FINAL_SUBMISSION/runtime/models/v25_stage_cache.csv`
  is the committed V25 inference over the released development set
  (`phase2/V28_CHAMPIONSHIP/v28_final_predictions.csv` +
  `phase2/V27_REJECTION/v25_rejection_features.csv`).
