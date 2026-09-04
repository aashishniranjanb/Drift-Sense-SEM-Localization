"""Does pose_v3's cost reduction change its answer? Measure, don't assume.

Runs pose_v2 and pose_v3 on the same pairs, serially in one process, and compares
both the estimates (against ground truth, by rubric tier) and the wall time.
"""
import os, sys, time
import numpy as np, pandas as pd, cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "EXP003_pose")))
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
cv2.setNumThreads(1)

from pose_v2 import estimate_pose_v2
from pose_v3 import estimate_pose_v3

DATA = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       os.path.join(_HERE, "..", "EXP001_pilot100", "data"))
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30

pairs = pd.read_csv(os.path.join(DATA, "pairs.csv"))
man = pd.read_csv(os.path.join(DATA, "manifest.csv"))[["pair_id", "z", "theta", "present"]]
M = pairs.merge(man, on="pair_id")
M = M[M.present == 1].head(N)

rows = []
for _, r in M.iterrows():
    ref = cv2.imread(os.path.join(DATA, r["reference_path"]), 0)
    srch = cv2.imread(os.path.join(DATA, r["search_path"]), 0)
    t = time.time(); a = estimate_pose_v2(ref, srch); ta = time.time() - t
    t = time.time(); b = estimate_pose_v3(ref, srch); tb = time.time() - t
    rows.append(dict(pair_id=r["pair_id"], z=r["z"], theta=r["theta"],
                     v2_s=a["best_scale"], v2_t=a["best_theta"], v2_time=ta,
                     v3_s=b["best_scale"], v3_t=b["best_theta"], v3_time=tb))
D = pd.DataFrame(rows)


def sc(p): return np.select([p <= .01, p <= .02, p <= .05], [1., .6, .3], 0.)
def rc(x): return np.select([x <= .25, x <= .5, x <= 1.], [1., .6, .3], 0.)


print(f"n = {len(D)} present pairs, serial, 1 thread\n")
for tag, s, t, tm in (("pose_v2", "v2_s", "v2_t", "v2_time"),
                      ("pose_v3", "v3_s", "v3_t", "v3_time")):
    ds = (D[s] - D.z).abs() / D.z
    dt = (D[t] - D.theta).abs()
    print(f"{tag}  scale<=1% {int((ds<=.01).sum()):2d}/{len(D)}  rot<=0.25d {int((dt<=.25).sum()):2d}/{len(D)}"
          f"  pose-credit {(.5*sc(ds.values)+.5*rc(dt.values)).mean():.4f}"
          f"  |  median {D[tm].median():.2f}s  max {D[tm].max():.2f}s")

same_s = (D.v2_s - D.v3_s).abs() < 1e-9
same_t = (D.v2_t - D.v3_t).abs() < 1e-9
print(f"\nidentical estimate: scale {int(same_s.sum())}/{len(D)}  theta {int(same_t.sum())}/{len(D)}")
d = D[~(same_s & same_t)]
if len(d):
    print("\ndisagreements:")
    print(d[["pair_id", "z", "v2_s", "v3_s", "theta", "v2_t", "v3_t"]].round(4).to_string(index=False))
print(f"\nspeedup: {D.v2_time.median()/D.v3_time.median():.2f}x")
D.round(5).to_csv(os.path.join(_HERE, "pose_equivalence.csv"), index=False)
