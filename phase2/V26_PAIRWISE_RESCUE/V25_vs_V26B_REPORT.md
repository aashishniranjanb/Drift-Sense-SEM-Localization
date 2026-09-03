# V26-B ANCHORED PAIRWISE RESCUE: FINAL REPORT

## 1. PROMOTION GATE: REJECTED
V26-B failed the primary gate. Introducing Pairwise Rescue degraded the Localization Score relative to the isolated V25 Anchor baseline (simulated).

## 2. THE DIAGNOSTIC REVELATION (Why V25 was "poisoned")
While evaluating V26-B, I discovered an architectural flaw in how we are extracting features that explains both V26-A's catastrophic failure and V26-B's struggle:

**Relative Features are Corrupted by the Rescue Candidates**
The V25 Ranker relies on amily_population and amily_ratio, which are calculated based on the *entire* candidate pool (K=200 in V25). 
When we injected R3/R2 rescue candidates into the feature extraction pool, we changed the denominator and the clustering distances. This silently corrupted amily_ratio for the true V25 candidates, changing their 25_ml_score, which then completely derailed the V25 Presence/Rejection model (causing False Negatives to jump from 62 to 91 even when no rescues were made!).

## 3. SWEEP RESULTS
Even when forcing the Rejection model to use the V25 anchor's margins, the Pairwise Model (trained symmetrically) struggled to beat V25 without sacrificing localization:

| Threshold | Rescues Attempted | Relative Loc Score (Simulated) |
|---|---|---|
| 0.98 (No rescues) | 0 | 69.17% |
| 0.95 | 5 | 69.17% |
| 0.90 | 20 | 66.67% |
| 0.80 | 56 | 55.83% |
| 0.60 | 110 | 45.19% |

Every rescue threshold lowered the localization score. The pairwise model is still prioritizing periodic replicas over the true GT when they are physically very similar, because the HistGradientBoostingClassifier is overfitting to the symmetric Δmargin features without spatial context.

## 4. NEXT STEPS
The pairwise verification approach is mathematically sound, but it cannot be trained safely on features (amily_ratio, context_combined_rel) that change depending on the size of the injected candidate pool. 

If we want to rescue the 35 candidates, we must:
1. Extract the V25 candidates and calculate their features **independently**.
2. Extract the Rescue candidates and calculate their features **independently**.
3. Only then compare them. 

V25 remains untouched and safely in production.
