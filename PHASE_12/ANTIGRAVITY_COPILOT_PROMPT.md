# ANTIGRAVITY / COPILOT — SHANGANIDHI — V10.8

You are an isolated research engineer on Drift-Sense++ SAFE-CAR 2.

OWNER: SHANGANIDHI
PHASE: 12
VERSION: V10.8
TASK: Presence / Absence Decision Layer

## HARD DATA LOCK
The ONLY authoritative benchmark is:
`data/phase2_dev/pairs.csv`
plus its referenced images.

You MUST read this main dataset directly.
You MUST NOT modify, regenerate, replace, rebalance, overwrite, or save over it.
You MUST NOT create another benchmark and call it the main benchmark.
Adversarial data may be inspected only as secondary diagnostics; it cannot establish a winning main result.

## NO-COLLISION LOCK
Work only in your assigned phase folder/branch.
DO NOT edit:
- `data/phase2_dev/*`
- `register.py`
- `inference.py`
- `benchmark_phase2.py`
- another teammate's phase
- final integration files

Do not merge automatically. Aashish integrates accepted work.

## OBJECTIVE
Build a separate presence classifier/gate that rejects same-architecture absent references while preserving true positives.

## DELIVERABLES
- presence_classifier_v108.py
- presence_feature_study.py
- V10.8_PRESENCE_DECISION.md
- V10.8_PRESENCE_FEATURES.csv

## PROCEDURE
1. Inspect the current frozen implementation.
2. Establish the baseline on the exact main dataset.
3. State one falsifiable hypothesis.
4. Build a diagnostic first.
5. Run it on the main dataset.
6. Implement the smallest isolated change.
7. Re-run the exact same benchmark population.
8. Report:
   - Set A <=1px / <=5px
   - Set B <=1px / <=5px
   - weighted localization
   - median and P95 error
   - scale MAE
   - rotation MAE
   - Set C rejection F1
   - Spearman rho
   - latency
   - failure taxonomy
   - Top-K retrieval recall when relevant
9. State whether the change is KEEP / MODIFY / REJECT.
10. Create `HANDOFF_TO_AASHISH.md`.

## LEAKAGE RULE
If training a model, split by case/pair, not individual candidate rows. Do not let candidates from the same pair appear in both train and validation/test.

## CONTRACT
Preserve:
`x, y, theta, scale, found, score`

If you discover that `z` is genuinely required by the official challenge interface, do not invent it. Document the source/requirement and ask Aashish to approve the contract change.

## HANDOFF MUST INCLUDE
- hypothesis
- exact dataset
- exact commands
- baseline
- after
- delta
- latency
- changed files
- failure examples
- risks
- KEEP/MODIFY/REJECT
