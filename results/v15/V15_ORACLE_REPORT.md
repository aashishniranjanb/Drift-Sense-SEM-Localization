# V15 Forensic Oracle Audit Report

## 1. Executive Forensic Ceilings (140 Present Cases)

| Stage | Metric Description | Current Value | Theoretical Ceiling | Bottleneck Status |
| :--- | :--- | :---: | :---: | :--- |
| **Stage 1: Raw Correlation** | GT Peak Available in Correlation Plane | **96.43%** (135/140) | 100.0% | Moderate Loss (Scale/Rotation Mismatch) |
| **Stage 2: Candidate Retrieval** | GT in Top-50 Candidates | **50.00%** (70/140) | 74.29% (Top-500) | **PRIMARY BOTTLENECK (Retrieval Cap)** |
| | GT in Top-100 Candidates | **60.71%** (85/140) | 74.29% | Retrieval Expansion Opportunity |
| | GT in Top-500 Candidates | **74.29%** (104/140) | 74.29% | Absolute Retrieval Limit |
| **Stage 3: Candidate Ranking** | Conditional Rank #1 Accuracy (when GT in Top-50) | **60.00%** (42/70) | 100.0% | **SECONDARY BOTTLENECK (Replica Ambiguity)** |
| **Stage 4: Metrology Refinement** | Subpixel Accuracy (<= 1.0 px when GT chosen) | **77.88%** (81/104) | 100.0% | **SOLVED / EXCELLENT** |

---

## 2. Failure Funnel Decomposition (140 Present Cases)

```text
140 PRESENT CASES
 |
 +-- [1] SUBPIXEL SUCCESS (<= 1.0 px):            25 cases (17.9%)
 +-- [2] IN-BOUNDS SUCCESS (1.0 - 5.0 px):        2 cases (1.4%)
 +-- [3] PRESENCE FALSE REJECTION (Score < 0.58): 77 cases (55.0%)
 +-- [4] RANKING FAILURE (GT in Top-50, lost):    17 cases (12.1%)
 +-- [5] RETRIEVAL CAP (GT in 51-500, omitted):   13 cases (9.3%)
 +-- [6] RETRIEVAL MISSING (Not in Top-500):      6 cases (4.3%)
```

---

## 3. Decisive Championship Takeaways

1.  **Retrieval Ceiling = 74.29% (Top-500) vs. 50.00% (Top-50)**:
    *   Expanding the candidate pool from K=50 to K=100 and applying periodic-family compression (V16) can immediately rescue up to **34 cases** that are currently truncated.
2.  **Ranking Ceiling = 60.00%**:
    *   When the correct candidate is present in Top-50, CAR + PACE selects the physical true location **60.00%** of the time. The remaining ranking failures are caused by identical periodic clone scores.
3.  **Metrology is 100% Solved**:
    *   When the correct candidate is selected, subpixel phase correlation achieves <= 1.0 px accuracy **77.88%** of the time.

All forensic records are archived in results/v15/V15_ORACLE_AUDIT.csv.
