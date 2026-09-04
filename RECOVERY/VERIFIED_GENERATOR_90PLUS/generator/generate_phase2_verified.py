"""Phase-2 verified generator.

    python generate_phase2_verified.py --output-dir DIR --n 100 --seed 20260904 [--profile pilot]

Emits reference/search PNGs + pairs.csv + ground_truth.csv + manifest.csv.
Every present sample must pass the INDEPENDENT verifier (verify_ground_truth.py)
before it is written; failures are resampled up to --retries, then dropped and
logged. No unverified label is ever shipped.

Geometry: one canvas->search affine (geometry.py). GT is the reference-crop
centre pushed through that same transform, plus the post-pose raster-drift
displacement (R5).
"""
import argparse, json, os, sys, time
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import patterns as PT                      # noqa: E402
import geometry as G                       # noqa: E402
import degrade as DG                       # noqa: E402
from verify_ground_truth import verify     # noqa: E402

SS = 3                     # supersampling for area integration
BLOCK = 300                # supersampled rows per block (bounds memory)
CANVAS_HALF_NM = 40000.0   # modelled canvas half-extent (procedural: effectively unbounded)


def _render_search(field, z, theta, c_canvas, c_search):
    """Area-integrated search image: evaluate the field on an SSxSS supersampled
    grid of canvas coords and box-average down. True area integration (§3.1)."""
    n = G.SEARCH_PX * SS
    out = np.zeros((G.SEARCH_PX, G.SEARCH_PX), dtype=np.float32)
    off = (np.arange(n) + 0.5) / SS - 0.5
    Rm = G.R(theta).T
    for y0 in range(0, n, BLOCK):
        y1 = min(n, y0 + BLOCK)
        gy, gx = np.meshgrid(off[y0:y1], off, indexing="ij")
        qx = (gx - c_search[0]) * z
        qy = (gy - c_search[1]) * z
        cx = Rm[0, 0] * qx + Rm[0, 1] * qy + c_canvas[0]
        cy = Rm[1, 0] * qx + Rm[1, 1] * qy + c_canvas[1]
        v = field(cx, cy).astype(np.float32)
        # box-average this block down by SS
        vb = v.reshape(-1, SS, G.SEARCH_PX, SS).mean(axis=(1, 3))
        out[y0 // SS:y1 // SS, :] = vb
        del gy, gx, qx, qy, cx, cy, v, vb
    return out


def _render_reference(field, center_canvas, ss=2):
    n = G.REF_PX * ss
    off = (np.arange(n) + 0.5) / ss - G.REF_PX / 2.0
    out = np.zeros((G.REF_PX, G.REF_PX), dtype=np.float32)
    for y0 in range(0, n, BLOCK):
        y1 = min(n, y0 + BLOCK)
        gy, gx = np.meshgrid(off[y0:y1] + center_canvas[1], off + center_canvas[0], indexing="ij")
        v = field(gx, gy).astype(np.float32)
        out[y0 // ss:y1 // ss, :] = v.reshape(-1, ss, G.REF_PX, ss).mean(axis=(1, 3))
        del gy, gx, v
    return out


def _self_ambiguity(ref01):
    """Cheap pre-check: how uniquely does the reference's own core locate itself
    inside the reference? A crop cut from deep inside a uniform periodic mat
    matches itself everywhere -> it can never produce a verifiable label.

    Returns (margin, core_is_distinctive). margin = self-peak minus the best
    competing peak outside an exclusion window."""
    r8 = (np.clip(ref01, 0, 1) * 255).astype(np.uint8)
    n = r8.shape[0]
    c = n // 2
    half = n // 6                      # core patch ~1/3 of the reference
    core = r8[c - half:c + half, c - half:c + half]
    corr = cv2.matchTemplate(r8.astype(np.float32), core.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, peak, _, loc = cv2.minMaxLoc(corr)
    work = corr.copy()
    ex = max(8, n // 20)
    y1, y2 = max(0, loc[1] - ex), min(work.shape[0], loc[1] + ex + 1)
    x1, x2 = max(0, loc[0] - ex), min(work.shape[1], loc[0] + ex + 1)
    work[y1:y2, x1:x2] = -99.0
    _, comp, _, _ = cv2.minMaxLoc(work)
    return float(peak - comp)


def _pick_zone_target(z, theta, c_canvas, c_search, params, rng, lo=280.0, hi=720.0):
    """Choose a reference-crop centre ON the mat/strip lattice (both axes) that
    also maps inside the search frame.

    Measured: crops taken inside a uniform mat have self-ambiguity 0.00 -- they
    repeat everywhere and can never yield a verifiable label (this is exactly
    what is wrong with data/phase2_dev). Crops straddling a zone boundary
    measure 0.22-0.44. So we enumerate the zone lattice rather than snapping and
    hoping.

    Returns (canvas_centre, tx, ty) or None.
    """
    mat = params["zone_mat_nm"]; strip = params["zone_strip_nm"]
    per = mat + strip
    phx, phy = params["zone_phase_x"], params["zone_phase_y"]

    # canvas bounding box of the admissible search window
    corners = np.array([[lo, lo], [hi, lo], [lo, hi], [hi, hi]])
    P = G.search_to_canvas(corners, z, theta, c_canvas, c_search)
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()

    def ks(a0, a1, ph):
        base = ph + mat + strip * 0.5
        k_lo = int(np.floor((a0 - base) / per)) - 1
        k_hi = int(np.ceil((a1 - base) / per)) + 1
        return [base + k * per for k in range(k_lo, k_hi + 1)]

    xs = ks(x0, x1, phx)
    ys = ks(y0, y1, phy)
    if not xs or not ys:
        return None
    cand = [(cx, cy) for cx in xs for cy in ys]
    rng.shuffle(cand)
    for cx, cy in cand:
        # small jitter so the crop is not always dead-centre on the strip
        jx = cx + rng.normal(0, strip * 0.35)
        jy = cy + rng.normal(0, strip * 0.35)
        b = G.canvas_to_search(np.array([[jx, jy]]), z, theta, c_canvas, c_search)[0]
        if lo <= b[0] <= hi and lo <= b[1] <= hi:
            return np.array([jx, jy]), float(b[0]), float(b[1])
    return None


def _sample_spec(rng, profile, idx):
    """Choose preset / pose / severity / presence for one sample."""
    if profile == "pilot":
        # 25 dram | 25 finfet | 25 periodic-hard | 25 degradation-hard
        band = idx // 25
        if band == 0:
            kind, sev, hard = "dram", int(rng.integers(0, 2)), "nominal"
        elif band == 1:
            kind, sev, hard = "finfet", int(rng.integers(0, 2)), "nominal"
        elif band == 2:
            kind, sev, hard = (rng.choice(["dram", "finfet"]), int(rng.integers(0, 2)), "periodic")
        else:
            kind, sev, hard = (rng.choice(["dram", "finfet"]), int(rng.integers(3, 5)), "degraded")
    else:
        kind = rng.choice(["dram", "finfet"])
        sev = int(rng.integers(0, 5))
        hard = rng.choice(["nominal", "periodic", "degraded"], p=[0.5, 0.25, 0.25])
    preset = str(rng.choice(PT.preset_names(kind)))
    z = float(rng.uniform(8.0, 12.0))
    theta = float(rng.uniform(-5.0, 5.0))
    return dict(preset=preset, kind=kind, z=z, theta=theta, severity=sev, hard=hard)


def build_one(args):
    idx, seed, profile, outdir, retries = args
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1)
    pid = f"v{idx:05d}"
    last = None
    # presence is fixed per sample -- it must NOT re-roll on retry, or failed
    # present samples silently become absent samples
    present = bool(np.random.default_rng(seed + 77777 + idx).random() > 0.20)
    for attempt in range(retries):
        rng = np.random.default_rng(seed + idx * 1009 + attempt * 7919)
        spec = _sample_spec(rng, profile, idx)
        spec["present"] = present
        z, theta = spec["z"], spec["theta"]
        # periodic-hard: shrink the zone structure so the field is more uniformly
        # periodic (more replicas competing with the true site)
        zscale = 0.55 if spec["hard"] == "periodic" else float(rng.uniform(0.85, 1.25))
        lw = float(rng.normal(0, 2.0))
        field, kind, params = PT.make_field(spec["preset"], rng, zone_scale=zscale, linewidth_bias=lw)

        c_canvas = np.array([0.0, 0.0])
        c_search = np.array([G.SEARCH_PX / 2.0, G.SEARCH_PX / 2.0])
        ok3, corner_d = G.assert_R3(z, theta, c_canvas, c_search, CANVAS_HALF_NM)
        if not ok3:
            continue

        # R4 + distinctiveness: pick the target in SEARCH coords, pull it back,
        # SNAP the crop onto a mat/strip boundary so it carries structure the
        # lattice does not repeat, push forward, require it still sits well
        # inside the frame.
        picked = _pick_zone_target(z, theta, c_canvas, c_search, params, rng)
        if picked is None:
            continue
        ref_center_canvas, tx, ty = picked

        # render the REFERENCE first (cheap) and reject self-ambiguous crops
        # before paying for the expensive search render
        amb = -1.0
        if spec["present"]:
            ref01 = _render_reference(field, ref_center_canvas)
            amb = _self_ambiguity(ref01)
            if amb < 0.12:
                continue
        else:
            # §4 decoy: SAME architecture family, INDEPENDENT canvas with clearly
            # different large-scale zoning so it carries structure the search lacks
            drng = np.random.default_rng(seed + 900000 + idx * 31 + attempt)
            dpreset = str(drng.choice(PT.preset_names(kind)))
            dfield, _, dparams = PT.make_field(dpreset, drng,
                                               zone_scale=float(drng.uniform(0.45, 0.75)),
                                               linewidth_bias=float(drng.normal(0, 2.5)))
            ref01 = _render_reference(dfield, np.array([drng.uniform(-8000, 8000),
                                                        drng.uniform(-8000, 8000)]))
            params["decoy_preset"] = dpreset

        search01 = _render_search(field, z, theta, c_canvas, c_search)

        srng = np.random.default_rng(seed + 500000 + idx * 13 + attempt)
        ref_u8, _, _ = DG.apply_sem(ref01, spec["severity"], srng, is_reference=True)
        srch_u8, ddx, ddy = DG.apply_sem(search01, spec["severity"], srng, is_reference=False)

        if spec["present"]:
            gt_x, gt_y = tx + ddx, ty + ddy      # R5: label tracks the post-pose drift
            gt = dict(present=1, x=float(gt_x), y=float(gt_y), theta=float(theta), scale=float(z))
        else:
            gt = dict(present=0, x=0.0, y=0.0, theta=0.0, scale=0.0)

        # ---- write, then verify the RE-READ files (never an in-memory render) ----
        rp = os.path.join(outdir, "reference", f"{pid}.png")
        sp = os.path.join(outdir, "search", f"{pid}.png")
        cv2.imwrite(rp, ref_u8); cv2.imwrite(sp, srch_u8)
        vres = verify(rp, sp, gt)
        last = vres
        if vres["ship"]:
            man = dict(pair_id=pid, attempt=attempt, self_amb=amb, **spec, **{f"p_{k}": v for k, v in params.items()},
                       drift_dx=ddx, drift_dy=ddy, corner_max_nm=corner_d,
                       **{f"v_{k}": v for k, v in vres.items()})
            return pid, gt, man, True
        # failed -> resample
    # exhausted retries: remove artefacts, report the failure
    for p in (os.path.join(outdir, "reference", f"{pid}.png"),
              os.path.join(outdir, "search", f"{pid}.png")):
        if os.path.exists(p):
            os.remove(p)
    return pid, None, {"pair_id": pid, "DROPPED": True,
                       **{f"v_{k}": v for k, v in (last or {}).items()}}, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260904)
    ap.add_argument("--profile", default="pilot", choices=["pilot", "bulk"])
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()

    od = os.path.abspath(a.output_dir)
    os.makedirs(os.path.join(od, "reference"), exist_ok=True)
    os.makedirs(os.path.join(od, "search"), exist_ok=True)

    geo = G.assert_R1_R2()
    print("R1/R2:", json.dumps(geo))

    t0 = time.time()
    jobs = [(i, a.seed, a.profile, od, a.retries) for i in range(a.n)]
    pairs, gts, mans, n_ok, n_drop = [], [], [], 0, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (pid, gt, man, ok) in enumerate(ex.map(build_one, jobs)):
            mans.append(man)
            if ok:
                n_ok += 1
                pairs.append(dict(pair_id=pid, reference_path=f"reference/{pid}.png",
                                  search_path=f"search/{pid}.png"))
                gts.append(dict(pair_id=pid, **gt))
            else:
                n_drop += 1
            if (i + 1) % 20 == 0:
                print(f"{i+1}/{a.n}  ok={n_ok} dropped={n_drop}  {time.time()-t0:.0f}s", flush=True)

    import pandas as pd
    pd.DataFrame(pairs).to_csv(os.path.join(od, "pairs.csv"), index=False)
    pd.DataFrame(gts).to_csv(os.path.join(od, "ground_truth.csv"), index=False)
    pd.DataFrame(mans).to_csv(os.path.join(od, "manifest.csv"), index=False)
    rep = {"requested": a.n, "shipped": n_ok, "dropped": n_drop, "seed": a.seed,
           "profile": a.profile, "geometry": geo, "runtime_s": round(time.time() - t0, 1),
           "supersample": SS}
    json.dump(rep, open(os.path.join(od, "generator_report.json"), "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
