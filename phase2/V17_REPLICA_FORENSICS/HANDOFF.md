# Phase V17 to V18 Handoff Specification

## 1. Executive Summary
- **Phase:** V17 (Replica Discrimination Forensics)
- **Status:** **COMPLETE & FROZEN**
- **Frozen Predecessor:** `results/CONTROL_V16/`

## 2. Quantitative Deliverables
- `results/replica_pairwise_matrix.csv`: Complete candidate feature matrix ($C_{GT}, C_1, C_2, C_3$).
- `results/failure_categorization.csv`: 100% attributed failure mechanism catalog.
- `FAILURE_ANALYSIS.md`: Complete root-cause autopsy.
- `ABLATION.md`: Full statistical power and t-test ranking.
- `DECISION.md`: Concrete architectural rules for Phase V18.

## 3. Concrete Specifications for Phase V18 Engineer
Phase V18 should implement the 3 controlled ranker variants:
- **V18-A**: Baseline CAR control ($NCC + PSR + \text{Phase}$).
- **V18-B**: Multi-Evidence Composite ($NCC + \text{Context} + \text{Phase} + \text{Fingerprint}$).
- **V18-C**: Periodicity-Adaptive Center-Context Discriminator ($CAR + w_{\text{fam}} \cdot \text{CenterPrior}$).

**Target for V18:** Conditional Top-1 accuracy $\ge 75\%$ on candidate pool.
