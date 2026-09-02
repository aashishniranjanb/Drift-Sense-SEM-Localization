# Phase V17: Physical Replica Discrimination Forensics

## 1. Objective
Investigate and explain 100% of the remaining 35 periodic-replica localization failures in the V16 frozen baseline.

**Core Scientific Question:**
> **Why does the true physical Ground Truth location lose to a false periodic replica during candidate ranking?**

---

## 2. Directory Structure
```text
phase2/V17_REPLICA_FORENSICS/
├── README.md               # Overview of Phase V17
├── TASK.md                 # Concrete milestone checklist
├── HYPOTHESIS.md           # Hypotheses regarding periodic failure mechanisms
├── EXPERIMENT_PLAN.md      # Multi-candidate forensic extraction protocol
├── baseline/               # Pointers to frozen V16 control
├── src/                    # Forensic feature extraction tools
├── experiments/            # Batch forensic execution scripts
├── results/                # Pairwise difference tables and failure assignments
├── plots/                  # Statistical distributions
├── FAILURE_ANALYSIS.md     # In-depth failure mechanism breakdown (>=90% explained)
├── ABLATION.md             # Feature separability and power analysis
├── DECISION.md             # Scientific verdict and design rules for Phase V18
└── HANDOFF.md              # Standardized handoff specification
```

---

## 3. Success Criteria
- [x] Zero changes to production inference code.
- [x] Extract full feature vectors for GT, Winner (#1), 2nd, and 3rd candidates across all 35 periodic failures.
- [x] Generate comprehensive pairwise diff matrix ($GT - Winner$).
- [x] Formally categorize $\ge 90\%$ of ranking failures into rigorous physical failure mechanisms.
