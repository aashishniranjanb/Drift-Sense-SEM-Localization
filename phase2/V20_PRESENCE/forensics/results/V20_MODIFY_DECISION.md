# V20 MODIFY DECISION

## Decision
**MODIFY**

## Justification
The previous models (V20-A to V20-E) failed because they relied too heavily on corr_score and context_score, which are naturally high for periodic replicas. The forensics prove that periodic hard negatives can be reliably separated from true degraded cases by penalizing high-ambiguity fields.

## V20-G Recommendations
To build V20-G, we must explicitly incorporate structural uniqueness and ambiguity penalty terms:

1. **Peak Margin Penalty**: Strongly reject candidates with peak_margin < 0.01.
2. **Structural Cut Dependency**: Incorporate 
earest_cut_dist. If 
earest_cut_dist > 20 (no structural breaks nearby), the confidence score should be dramatically scaled down.
3. **Replica Family Suppression**: Penalize score confidence proportional to amily_population if it reaches maximum bounds.

These three new features (peak_margin, 
earest_cut_dist, and amily_population) should form the core of the V20-G classifier input vector.

## Next Experiment
Implement V20-G using a refined multi-evidence formulation or Logistic Regression trained strictly on this new augmented feature set, specifically weighting peak_margin and 
earest_cut_dist to suppress periodic false positives.
