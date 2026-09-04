"""Run engine_v3 over a dataset, emitting predictions + per-stage timings + the
raw evidence a presence model would consume (fitted elsewhere, on train only).

    python run.py --data DIR --out DIR [--workers N] [--serial]

--serial times a single pair with no contention, which is the number that has to
meet the 5 s median budget on the 4-core reference machine.
"""
import argparse, os, sys, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))


def work(job):
    row, data = job
    sys.path.insert(0, _HERE)
    import engine_v3
    engine_v3.setup()
    ref = cv2.imread(os.path.join(data, row["reference_path"]), 0)
    srch = cv2.imread(os.path.join(data, row["search_path"]), 0)
    t0 = time.time()
    r, T = engine_v3.run_pair(ref, srch)
    total = time.time() - t0
    base = dict(pair_id=row["pair_id"], runtime=total,
                **{f"t_{k}": v for k, v in T.items()})
    if r is None:
        return dict(base, x=0.0, y=0.0, theta=0.0, scale=0.0, ctx=np.nan,
                    ctx_margin=np.nan, corr=np.nan, corr_margin=np.nan)
    return dict(base, **r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default=_HERE)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--serial", type=int, default=0, help="time N pairs with 1 worker")
    a = ap.parse_args()
    data = os.path.abspath(a.data)
    pairs = pd.read_csv(os.path.join(data, "pairs.csv"))
    jobs = [(r, data) for _, r in pairs.iterrows()]
    if a.serial:
        jobs = jobs[:a.serial]
        out = [work(j) for j in jobs]
    else:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            out = list(ex.map(work, jobs))
        print(f"wall {time.time()-t0:.0f}s")
    R = pd.DataFrame(out)
    tcols = [c for c in R.columns if c.startswith("t_")]
    print(f"\nruntime median {R.runtime.median():.2f}s  max {R.runtime.max():.2f}s"
          f"  ({'SERIAL, 1 worker' if a.serial else f'{a.workers} workers'})")
    print("stage medians: " + "  ".join(f"{c[2:]} {R[c].median():.2f}s" for c in tcols))
    os.makedirs(a.out, exist_ok=True)
    R.round(5).to_csv(os.path.join(a.out, "evidence.csv"), index=False)


if __name__ == "__main__":
    main()
