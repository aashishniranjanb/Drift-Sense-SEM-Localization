# ANTIGRAVITY / COPILOT — SAI_DHARSHAN — V10.9

You are an isolated research engineer on Drift-Sense++ SAFE-CAR 2.

OWNER: SAI_DHARSHAN
PHASE: 13
VERSION: V10.9
TASK: Hard-Negative Mining from Main Dataset

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
Mine the hardest periodic replicas, false positives, false negatives, and near-tie candidates from the frozen main dataset.

## DELIVERABLES
- hard_negative_miner_v109.py
- hard_negative_buffer.csv
- V10.9_HARD_NEGATIVE_REPORT.md

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
