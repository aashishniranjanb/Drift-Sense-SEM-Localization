# Research archive — map

Everything in this repository **outside `FINAL_SUBMISSION/`** is historical.
It is retained for scientific transparency and is **not required** to run or
score the submission. Nothing here is on the `register.py` execution path.

| Path | Contents |
|---|---|
| `phase2/` | The full Phase 2 version history (V10 → V48): localization, pose, rejection, calibration experiments, per-version reports and prediction CSVs. `phase2/V48_MAX/` holds the calibration study that produced the shipped `calib_lean.pkl`. `phase2/V27_FINAL/SCORER_AUDIT.md` documents the official scoring formulas. |
| `PHASE_10` … `PHASE_16` | Earlier phase working directories. |
| `experiments/` | Standalone architecture experiments (multiscale, siamese, PACE group ranking, CAR dual-channel, multi-view). |
| `diagnostics_and_tools/` | One-off analysis and tuning scripts. |
| `data/` | Development datasets and generated pairs. |
| `models/` | Superseded model checkpoints. |
| `production_engine/` | Earlier production runner and UI server. |
| `submission/` | Working copy of the runtime that was assembled into `FINAL_SUBMISSION/`. |
| `submission_package/` | Phase 1 submission package (superseded). |
| `V41_SUBMISSION/` | Interim V41-only package (superseded by `FINAL_SUBMISSION/`). |
| `hf_space*`, `hf_clone/` | Hugging Face Space mirrors. |
| `DriftSense_TeamPlan/`, `docs/`, `team/` | Planning, design notes, per-workstream audits. |
| `AMP_Phase 2 material/`, `Dataset_AMP_Phase 2/` | Organizer-provided Phase 2 reference material and its generator. |
| `figures/`, `phase2_deck_visuals/` | Figure sources for the submission deck. |
| `rgb_bonus_package/` | RGB / Set-D bonus material. |

To reproduce the shipped calibration model, see
`phase2/V48_MAX/b4_calib_lean.py`. To reproduce the scoring numbers, see
`phase2/V48_MAX/v48_score.py` and `phase2/benchmark_phase2.py`.
