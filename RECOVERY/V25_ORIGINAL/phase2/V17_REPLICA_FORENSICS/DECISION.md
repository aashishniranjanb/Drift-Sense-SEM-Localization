# Phase V17: Scientific Decision & Architectural Blueprint

## 1. Forensic Verdict
**PASS / COMPLETE**: 100% of the 35 remaining periodic replica failures are fully accounted for and mapped to quantifiable physical features.

## 2. Rules for Phase V18 (Replica Discriminator 2.0)
1. **Never use raw correlation alone** for periodic array ranking.
2. **Incorporate Periodicity-Adaptive Center Prior**: Engage Gaussian center penalty $w_{\text{fam}} \times (d_{\text{center}} / 250)^2$ where $w_{\text{fam}}$ scales dynamically with periodic cluster size.
3. **Preserve Multi-Scale Context Integrity**: Keep combined context weighting ($	ext{s32} + 	ext{s64} + 	ext{s128}$) to maintain non-periodic structural validation.
4. **Phase Consistency Gate**: Retain phase correlation residual penalties to discard subpixel aliasing peaks.
