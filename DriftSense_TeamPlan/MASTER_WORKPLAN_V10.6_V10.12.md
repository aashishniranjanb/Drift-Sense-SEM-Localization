# MASTER PARALLEL WORK PLAN — PHASES 10–16

## Mission
Attack the actual bottlenecks without corrupting the benchmark or creating merge collisions.

Existing evidence identifies retrieval as a critical ceiling: the V10.5 study reported only 39.29% Top-20 candidate recall on its candidate-feature population. Earlier project evidence also shows that candidate dilution and latency can make an apparently richer method worse; therefore every new method must be evaluated for recall, final localization, rejection, and latency together. fileciteturn11file2L148-L167

## Ownership
| Phase | Version | Owner | Focus |
|---|---|---|---|
| 10 | V10.6 | Sai Dharshan | High-recall candidate retrieval |
| 11 | V10.7 | Akhilesh | Replica-aware ranker |
| 12 | V10.8 | Shanganidhi | Presence/absence decision |
| 13 | V10.9 | Sai Dharshan | Hard-negative mining |
| 14 | V10.10 | Akhilesh | Pose-robust retrieval |
| 15 | V10.11 | Shanganidhi | Calibration/confidence gate |
| 16 | V10.12 | Shanganidhi | Independent referee/generalization audit |

## Parallel execution
Phases 10–12 can run immediately and concurrently.
Phase 13 can run immediately from existing main-dataset candidate logs.
Phase 14 is an independent diagnostic.
Phase 15 must not alter x/y/pose.
Phase 16 remains an audit until Aashish requests integration.

## Main-track rule
Aashish alone decides what enters the final `inference_phase2.py` / registration path. A teammate's result is not integrated automatically.

## Acceptance philosophy
The Phase-2 objective is multi-objective:
Localization 40, Pose 20, Rejection 15, Calibration 10, Efficiency 5, Generator/Citations/Failure analysis 10.
