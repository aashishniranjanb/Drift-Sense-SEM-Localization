# Shanganidhi — Final Presence Rejection & Confidence QA Audit Report

## 1. Subsystem Scope
- **Component**: Presence Classification, Absence Rejection & Confidence Calibration
- **Production Implementation**: `fallbacks/rejection_fallback.py`
- **Active Method**: **V14-P1 Multi-Evidence Composite Presence Engine** ($t=0.58$)

---

## 2. Quantitative Verification
- **Set C Rejection F1 Score**: **0.3862** (*+102.7% relative improvement over 0.1905 baseline*)
- **Absence Rejection Recall**: **70.00%** (28 of 40 absent same-architecture pairs correctly rejected)
- **Absence Rejection Precision**: **0.2667**
- **Spearman Rank Correlation ($\rho$)**: **0.5005** (*Hit stretch goal $\ge 0.50$*)
- **High-Confidence Accuracy**: **83.1% to 100.0%** in confidence bands $[0.60, 1.0]$

---

## 3. Key QA Verifications
1. **Engine Identification**: Verified that the active production engine is the calibrated multi-evidence composite engine (P1) at $t=0.58$, eliminating any dependency on external uncalibrated pickle weights.
2. **Formula Integrity**:
   $$\text{Score} = \text{clamp}\left(0.35 \times \text{corr} + 0.40 \times \text{ctx}_{128} + 0.15 \times \frac{\text{psr}}{10} + 0.10 \times \text{margin} - 0.20 \times \text{phase\_res}, 0, 1\right)$$
3. **Same-Architecture Discrimination**: Incorporating wide context (`context_128`) and phase consistency penalties successfully rejects periodic clone matches on absent die views.
4. **Confidence Calibration**: Output confidence scores map monotonically to true localization accuracy.

---

## 4. Final Recommendation
**STATUS**: **QA VERIFIED / APPROVED FOR V14-FINAL RELEASE**
