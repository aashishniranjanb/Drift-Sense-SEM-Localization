"""EXP002 promotion gate: per-pair comparison of context_combined vs the V25 ranker.

Gate (mission spec): >=5 additional <=5px recoveries, 0 baseline successes
broken, deterministic, median runtime <=5s. Absent-FP is unaffected here because
the selector does not change the presence decision path -- reported separately.
"""
import sys
import numpy as np, pandas as pd

D = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "pool_pilot100.csv")
P = D[D.present == 1].copy()
P["err"] = np.where(P.refined_gterr >= 0, P.refined_gterr, P.gterr)

base = P.loc[P.groupby("pair_id")["v25"].idxmax()].set_index("pair_id")["err"]
new = P.loc[P.groupby("pair_id")["context_combined"].idxmax()].set_index("pair_id")["err"]
C = pd.DataFrame({"base_err": base, "new_err": new})
C["base_ok"] = C.base_err <= 5
C["new_ok"] = C.new_err <= 5

rec = C[~C.base_ok & C.new_ok]
brk = C[C.base_ok & ~C.new_ok]
print(f"pairs={len(C)}  baseline <=5px {int(C.base_ok.sum())}  new <=5px {int(C.new_ok.sum())}")
print(f"RECOVERED {len(rec)}   BROKEN {len(brk)}")
if len(brk):
    print("\nbroken:")
    print(brk[["base_err", "new_err"]].round(3).to_string())
still = C[~C.base_ok & ~C.new_ok]
print(f"\nstill failing (both): {len(still)}")
if len(still):
    print(still[["base_err", "new_err"]].round(2).to_string())
C.round(4).to_csv("gate_per_pair.csv")

# determinism: selector is a pure argmax over deterministic features
d = D.groupby("pair_id")["context_combined"].apply(lambda s: (s == s.max()).sum())
print(f"\nties at argmax (non-determinism risk): pairs with >1 max = {int((d > 1).sum())}")
print(f"runtime median {D.groupby('pair_id').pair_runtime.first().median():.2f}s "
      f"max {D.groupby('pair_id').pair_runtime.first().max():.2f}s")
