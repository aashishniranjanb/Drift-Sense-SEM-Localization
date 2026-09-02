# Phase V17: Tasks & Milestones

- [x] **Task 1: Baseline Freeze**
  - Lock V16 outputs into `results/CONTROL_V16/` with commit ID and full metrics.
- [x] **Task 2: Forensic Feature Extractor**
  - Implement `src/forensic_extractor.py` to extract candidate-level physical, frequency, and structural features (GT, Winner, 2nd, 3rd).
- [x] **Task 3: Pairwise Difference Matrix Generation**
  - Execute `experiments/run_replica_forensics.py` on the 35 periodic failure cases.
- [x] **Task 4: Failure Categorization & Mechanism Assignment**
  - Document failure causes for $\ge 90\%$ of failures in `FAILURE_ANALYSIS.md`.
- [x] **Task 5: Feature Power & Separability Ablation**
  - Tabulate feature separability, win rates, and t-statistics in `ABLATION.md`.
- [x] **Task 6: Formal Scientific Decision & Handoff**
  - Deliver `DECISION.md` and `HANDOFF.md` containing precise mathematical requirements for Phase V18.
