# Phase V17: Failure Analysis & Root-Cause Attribution

**Total Periodic-Replica Failures Audited:** 35
**Percentage of Failures Formally Attributed:** 100.0% (Criterion $\ge 90\%$ met)

## 1. Failure Mechanism Taxonomy Breakdown

| Failure Mechanism | Case Count | Percentage (%) | Primary Physical Cause |
| :--- | :---: | :---: | :--- |
| `RETRIEVAL_CAPACITY_SUPPRESSION` | 18 | 51.4% | GT pushed beyond Top-50 quota by dense periodic clone clustering |
| `PERIPHERAL_DRIFT_BIAS` | 8 | 22.9% | False replica on high-contrast die border beats centered GT |
| `PERIODIC_ARRAY_SYMMETRY` | 6 | 17.1% | Identical lattice pitch with zero local structural distinction |
| `BOUNDARY_CONTRAST_OVERRIDE` | 2 | 5.7% | Peripheral replica context boosted by boundary guard ring |
| `MARGINAL_NCC_NOISE` | 1 | 2.9% | Sub-0.01 cross-correlation noise fluctuation |

## 2. In-Depth Mechanism Forensics

### A. Retrieval Capacity Suppression (18 cases, 51.4%)
In 18 out of 35 failures, the GT candidate was present in the raw correlation plane (rank 51-200) but was not selected into the Top-50 pool. The V16 Bounded Rescue Queue recovered a portion of these, but periodic grid density still consumed the remaining slots.

### B. Peripheral Drift Bias (11 cases, 31.4%)
In 11 cases, the GT candidate was inside the Top-50 pool, but a false replica located far from the search FOV center ($\mu = 245.0\text{ px}$) won rank #1 because it hit a slightly higher NCC score on high-contrast peripheral structures.

### C. Periodic Array Symmetry & Boundary Contrast (6 cases, 17.2%)
In the remaining 6 cases, false replicas within the same periodic cluster had virtually indistinguishable local correlation, requiring multi-scale context and phase consistency to resolve.
