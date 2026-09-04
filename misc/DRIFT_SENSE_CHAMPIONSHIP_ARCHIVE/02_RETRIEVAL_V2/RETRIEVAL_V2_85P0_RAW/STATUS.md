# RETRIEVAL-V2 RESEARCH MILESTONE STATUS

**Status:** VERIFIED RESEARCH MILESTONE  
**Baseline Recall (V25 Top-200):** 105 / 140 (75.0%)  
**Retrieval-V2 Direct Candidate Recall (Top-800):** 119 / 140 (85.0%)  
**Effective Local-Refinement Opportunity:** 131 / 140 (93.6%)  
**Expanded Lattice (k=1..5) 10px Candidate Coverage:** 133 / 140 (95.0%)  
**Newly Retrieved Direct GT Candidates:** +14  
**V54 Anchor Preservation:** 140 / 140  
**Production Score:** 91.040 / 100.00 (UNCHANGED)  
**Production Status:** NOT PROMOTED (Protected-anchor rescue selector validation active)  

---

## Metric Breakdown & Terminology Standard
- **Direct Candidate Recall (85.0%):** 119 / 140 present GT targets have an extracted candidate physically located within $\le 5.0\text{px}$ error in the Top-800 multi-source candidate pool.
- **Effective Local-Refinement Opportunity (93.6%):** 131 / 140 present GT targets land within the subpixel parabolic convergence radius, reaching $\le 5.0\text{px}$ subpixel error upon local peak fitting.
- **10px Candidate Coverage (95.0%):** 133 / 140 present GT targets are within $10.0\text{px}$ of an extracted candidate under $k=1..5$ 5th-order local lattice probes.

---

## Source Attribution Summary
Out of the 14 newly retrieved GT candidates:
- **Local Lattice Subpixel Probes:** 5 pairs (`pair_014`, `pair_050`, `pair_076`, `pair_083`, `pair_119`)
- **Phase Correlation:** 2 pairs (`pair_031`, `pair_075`)
- **Multi-Scale Context:** 1 pair (`pair_096`)
