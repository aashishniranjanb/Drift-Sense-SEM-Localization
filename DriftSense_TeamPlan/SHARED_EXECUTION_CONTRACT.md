# DRIFT-SENSE++ V10.6–V10.12 PARALLEL TEAM CONTRACT

## 1. Authority
Aashish owns the MAIN TRACK and final integration. Teammates produce isolated evidence and implementation proposals.

## 2. AUTHORITATIVE DATASET — STRICT
Use ONLY:
`data/phase2_dev/pairs.csv`
and the reference/search images it references.

NEVER regenerate, edit, replace, rebalance, overwrite, or save over this dataset.
Every claimed improvement MUST reproduce on this exact main dataset.
Derived CSVs may be written outside `data/phase2_dev/`.

## 3. NO-COLLISION RULE
Do not edit another owner's phase.
Do not edit `register.py`, `inference.py`, `benchmark_phase2.py`, or final integration files.
Do not silently merge changes.
Your deliverable is a proposal + isolated implementation for Aashish to review.

## 4. EXTERNAL CONTRACT
Preserve:
`x, y, theta, scale, found, score`
Do not add/remove/rename fields casually. If the official challenge specification proves that `z` is required, document the evidence and stop for Aashish's approval before changing the contract.

## 5. SCIENTIFIC RULES
- Same benchmark population before/after.
- No denominator changes.
- No case leakage in learned models.
- Separate retrieval failure from ranking failure.
- Report latency with accuracy.
- Do not optimize one metric in isolation.

## 6. OBJECTIVE WEIGHTS
Localization 40; Pose 20; Rejection 15; Calibration 10; Efficiency 5; Generator/Citations/Failure analysis 10.

## 7. REQUIRED HANDOFF
Create `HANDOFF_TO_AASHISH.md` containing:
baseline, after, deltas, exact command, changed files, latency, failure taxonomy, risks, and KEEP/MODIFY/REJECT.
