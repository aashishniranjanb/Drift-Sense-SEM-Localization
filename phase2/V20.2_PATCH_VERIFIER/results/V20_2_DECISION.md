# V20.2 PATCH VERIFIER DECISION

## Decision
**REJECT**

## Justification
While the Two-Stream CNN architecture successfully learned to distinguish and reject periodic replica hard negatives (driving Set C FPR down to ~12%), it entirely failed to maintain reasonable recall on true degraded matches. Even with handcrafted physical priors appended (V20.2-D) and significant data augmentation, the test recall maxed out at ~50.6%.

The local visual patches in the heavily degraded test sets are too corrupted by noise for a local CNN to reliably confirm presence. The network becomes overly conservative, rejecting genuine matches.

## Next Steps
Abandon patch-level visual verification. Pivot to global or semi-global frequency/phase validation (Phase V20.3) where noise can be mitigated across the wider search field.
