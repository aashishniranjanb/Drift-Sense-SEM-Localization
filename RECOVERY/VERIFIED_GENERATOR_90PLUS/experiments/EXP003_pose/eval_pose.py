"""EXP003 evaluation: pose_v2 vs Engine B's sequential pose search.

Pose accuracy only -- localization is held fixed, so this isolates the pose stage.
GT is used to score, never to search.
"""
import os, sys, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_V25 = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "V25_ORIGINAL"))
DATA = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       os.path.join(_HERE, "..", "EXP001_pilot100", "data"))


def work(row):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
    sys.path.insert(0, _HERE)
    from pose_v2 import estimate_pose_v2
    ref = cv2.imread(os.path.join(DATA, row["reference_path"]), 0)
    srch = cv2.imread(os.path.join(DATA, row["search_path"]), 0)
    t0 = time.time()
    r = estimate_pose_v2(ref, srch)
    dt = time.time() - t0
    if r is None:
        return dict(pair_id=row["pair_id"], v2_scale=np.nan, v2_theta=np.nan, v2_t=dt)
    return dict(pair_id=row["pair_id"], v2_scale=r["best_scale"],
                v2_theta=r["best_theta"], v2_score=r["best_score"], v2_t=dt)


def main():
    pairs = pd.read_csv(os.path.join(DATA, "pairs.csv"))
    man = pd.read_csv(os.path.join(DATA, "manifest.csv"))[["pair_id", "z", "theta", "present"]]
    jobs = [r for _, r in pairs.iterrows()]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=5) as ex:
        out = list(ex.map(work, jobs))
    V = pd.DataFrame(out).merge(man, on="pair_id")
    V = V[V.present == 1]

    pool = pd.read_csv(os.path.join(_HERE, "..", "EXP002_selector", "pool_pilot100.csv"))
    b = pool.groupby("pair_id").agg(b_scale=("est_scale", "first"),
                                    b_theta=("est_theta", "first")).reset_index()
    V = V.merge(b, on="pair_id")

    def sc(p): return np.select([p <= .01, p <= .02, p <= .05], [1., .6, .3], 0.)
    def rc(d): return np.select([d <= .25, d <= .5, d <= 1.], [1., .6, .3], 0.)

    for tag, s, t in (("Engine B (sequential)", "b_scale", "b_theta"),
                      ("pose_v2 (joint)", "v2_scale", "v2_theta")):
        ds = (V[s] - V.z).abs() / V.z
        dt = (V[t] - V.theta).abs()
        print(f"{tag:24s} scale: med {100*ds.median():5.2f}%  <=1% {int((ds<=.01).sum()):2d}/{len(V)}"
              f"  |  rot: med {dt.median():.3f}d  <=0.25d {int((dt<=.25).sum()):2d}/{len(V)}"
              f"  |  pose-credit {(.5*sc(ds.values)+.5*rc(dt.values)).mean():.4f}")
    print(f"\npose_v2 runtime: median {V.v2_t.median():.2f}s  max {V.v2_t.max():.2f}s"
          f"  (wall {time.time()-t0:.0f}s, 5 workers)")

    ds_b = (V.b_scale - V.z).abs() / V.z
    ds_v = (V.v2_scale - V.z).abs() / V.z
    V["abstheta"] = V.theta.abs()
    V["b_bad"] = ds_b > .02
    V["v_bad"] = ds_v > .02
    print("\nscale >2% by |theta| bucket:")
    V["bucket"] = pd.cut(V.abstheta, [0, 1, 2, 3, 4, 6])
    print(V.groupby("bucket", observed=True).agg(n=("b_bad", "size"),
          engineB=("b_bad", "sum"), pose_v2=("v_bad", "sum")).to_string())
    V.round(5).to_csv(os.path.join(_HERE, "pose_v2_eval.csv"), index=False)


if __name__ == "__main__":
    main()
