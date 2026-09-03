# Research Archive — Map

Everything in this repository **outside `FINAL_SUBMISSION/`** is historical or
supporting. None of it is on the `register.py` / `inference.py` execution path.
It is kept for scientific transparency and reproducibility.

All research and development material lives under [`Experiments/`](./Experiments/).

## Version history & experiments

| Path | Contents |
|---|---|
| `Experiments/phase2/` | The full Phase 2 version history (V10 → V48): localization, pose, rejection and calibration experiments, per-version reports, and prediction CSVs. `phase2/V48_MAX/` holds the calibration study that produced the shipped `calib_lean.pkl` (`b4_calib_lean.py`) and the scorer (`v48_score.py`). `phase2/V27_FINAL/SCORER_AUDIT.md` documents the official scoring formulas. `phase2/benchmark_phase2.py` is the scoring harness. |
| `Experiments/PHASE_10` … `PHASE_16` | Earlier phase working directories. |
| `Experiments/experiments/` | Standalone architecture experiments (multiscale, siamese, PACE group ranking, CAR dual-channel, multi-view). |
| `Experiments/diagnostics_and_tools/` | One-off analysis and tuning scripts. |
| `Experiments/full_research.md` | Complete Phase 2 iteration log with key findings, what worked, what failed, and final architecture deep-dive. |

## Supporting material

| Path | Contents |
|---|---|
| `misc/` | Development scripts (`root_scripts/`), old data dumps (`root_data/`), superseded docs (`old_docs/`), the interim `V41_SUBMISSION/` package, the working copy the runtime was assembled from (`submission_working_copy/`), and old submission zips. |
| `releases/` | `FINAL_SUBMISSION.zip` — a zipped archive of the authoritative package (identical content). |
| `Experiments/submission_package/` | Phase 1 submission package (superseded). |
| `Experiments/data/`, `Experiments/models/` | Development datasets and superseded model checkpoints. |
| `Experiments/production_engine/`, `Experiments/demo/`, `Experiments/results/` | Earlier production runner, demo builder, result dumps. |
| `Experiments/hf_clone/` | Hugging Face Space mirror. |
| `Experiments/DriftSense_TeamPlan/`, `Experiments/docs/` | Planning and design notes. |
| `Experiments/team/` | Per-workstream **final audit notes** (results only). |
| `Experiments/figures/`, `Experiments/phase2_deck_visuals/` | Figure sources for the submission deck. |
| `Experiments/rgb_bonus_package/` | RGB / Set-D bonus material. |
| `Experiments/AMP_Phase 2 material/`, `Experiments/Dataset_AMP_Phase 2/` | Organizer-provided Phase 2 reference material and its generator. |
| `Experiments/Applied Materials *.pptx / *.docx` | Organizer task brief, dataset prompt, and session transcript. |

## Reproducing the shipped numbers

- Calibration model: `Experiments/phase2/V48_MAX/b4_calib_lean.py`.
- Score: `Experiments/phase2/V48_MAX/v48_score.py` and `Experiments/phase2/benchmark_phase2.py`.
- The V25 stage cache bundled in `FINAL_SUBMISSION/runtime/models/v25_stage_cache.csv`
  is the committed V25 inference over the released development set
  (`Experiments/phase2/V28_CHAMPIONSHIP/v28_final_predictions.csv` +
  `Experiments/phase2/V27_REJECTION/v25_rejection_features.csv`).
