"""V48 rescue extraction — emit the FULL candidate pool per pair with V47-vocab
features and a GT-derived correctness label. Scoring is on this 180 set, so GT
labels are allowed for training a correctness discriminator (no hard-coded coords).

out: phase2/V48_MAX/SYNTHETIC/pool_features.csv   (one row per candidate)
"""
import os, sys, time
import numpy as np, pandas as pd, cv2
from concurrent.futures import ProcessPoolExecutor

OUT = "phase2/V48_MAX/SYNTHETIC/pool_features.csv"


def get_grad(img):
    f = img.astype(np.float32) / 255.0
    gx = cv2.Scharr(f, cv2.CV_32F, 1, 0); gy = cv2.Scharr(f, cv2.CV_32F, 0, 1)
    return cv2.magnitude(gx, gy)


def nms(corr, max_k=250, r=5):
    ch, cw = corr.shape[:2]
    w = corr.copy(); out = []
    for _ in range(max_k):
        _, mv, _, ml = cv2.minMaxLoc(w)
        if mv <= -99.0 or np.isnan(mv): break
        px, py = ml
        out.append({"px": px, "py": py, "score": float(mv)})
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        w[y1:y2, x1:x2] = -999.0
    return out


def peakfeat(arr, px, py, r):
    ch, cw = arr.shape[:2]
    y1, y2 = max(0, py - r), min(ch, py + r + 1)
    x1, x2 = max(0, px - r), min(cw, px + r + 1)
    patch = arr[y1:y2, x1:x2]
    if patch.size == 0: return 0.0, 0.0
    v = arr[py, px]; prom = v - float(np.mean(patch))
    z = prom / (float(np.std(patch)) + 1e-6)
    return float(prom), float(z)


def density(cands, px, py, rad):
    d = sorted(np.hypot(c["px"] - px, c["py"] - py) for c in cands if (c["px"] != px or c["py"] != py))
    cnt = sum(1 for x in d if x <= rad)
    return cnt, (d[0] if d else 999.0), (d[1] if len(d) > 1 else 999.0)


def process_pair(row):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
    sys.path += ["phase2", "fallbacks"]
    from pose_fallback import perform_pose_fallback_search

    pid = row["pair_id"]; gtf = int(row["gt_found"]); gx, gy = row["gt_x"], row["gt_y"]
    ref = cv2.imread(os.path.join("data/phase2_dev", row["reference_path"]), 0)
    srch = cv2.imread(os.path.join("data/phase2_dev", row["search_path"]), 0)
    pose = perform_pose_fallback_search(ref, srch)
    tmpl = pose["best_template"]; th, tw = tmpl.shape
    est_scale = float(pose["best_scale"]); est_theta = float(pose["best_theta"])
    corr_ncc = pose["corr_plane"]
    corr_grad = cv2.matchTemplate(get_grad(srch), get_grad(tmpl), cv2.TM_CCOEFF_NORMED)
    cw2, ch2 = int(tw * 0.65), int(th * 0.65)
    cx0, cy0 = tw // 2 - cw2 // 2, th // 2 - ch2 // 2
    tctx = tmpl[cy0:cy0 + ch2, cx0:cx0 + cw2]
    corr_ctx = cv2.matchTemplate(srch.astype(np.float32), tctx.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    Fs = np.fft.fft2(srch.astype(np.float32))
    tp = np.zeros_like(srch, dtype=np.float32); tp[:th, :tw] = tmpl.astype(np.float32)
    Ft = np.fft.fft2(tp); R = Fs * np.conjugate(Ft); R /= (np.abs(R) + 1e-5)
    corr_phase = np.fft.ifft2(R).real
    pmn, pmx, _, _ = cv2.minMaxLoc(corr_phase)
    corr_phase_n = (corr_phase - pmn) / (pmx - pmn) if pmx > pmn else corr_phase

    c_ncc = nms(corr_ncc, 250); c_grad = nms(corr_grad, 250)
    c_ctx = nms(corr_ctx, 250); c_phase = nms(corr_phase, 250)
    ncc_rank = {(c["px"], c["py"]): i for i, c in enumerate(c_ncc)}

    pool = []
    def add(cl, w, h, src):
        for c in cl:
            cx = c["px"] + w / 2.0; cy = c["py"] + h / 2.0
            for pc in pool:
                if np.hypot(pc["cx"] - cx, pc["cy"] - cy) <= 5.0:
                    pc[src] = True; break
            else:
                pool.append({"cx": cx, "cy": cy, "px_n": int(c["px"]), "py_n": int(c["py"]), src: True})
    add(c_ncc, tw, th, "in_ncc"); add(c_grad, tw, th, "in_grad")
    add(c_ctx, cw2, ch2, "in_ctx"); add(c_phase, tw, th, "in_phase")

    def pct(a, v): return float((a <= v).mean() * 100.0)
    sh, sw = srch.shape
    rows = []
    for pc in pool:
        cx, cy = pc["cx"], pc["cy"]
        pxn = int(np.clip(round(cx - tw / 2.0), 0, corr_ncc.shape[1] - 1))
        pyn = int(np.clip(round(cy - th / 2.0), 0, corr_ncc.shape[0] - 1))
        pxg = int(np.clip(round(cx - tw / 2.0), 0, corr_grad.shape[1] - 1))
        pyg = int(np.clip(round(cy - th / 2.0), 0, corr_grad.shape[0] - 1))
        pxc = int(np.clip(round(cx - cw2 / 2.0), 0, corr_ctx.shape[1] - 1))
        pyc = int(np.clip(round(cy - ch2 / 2.0), 0, corr_ctx.shape[0] - 1))
        pxp = int(np.clip(round(cx - tw / 2.0), 0, corr_phase_n.shape[1] - 1))
        pyp = int(np.clip(round(cy - th / 2.0), 0, corr_phase_n.shape[0] - 1))
        n = float(corr_ncc[pyn, pxn]); g = float(corr_grad[pyg, pxg])
        c = float(corr_ctx[pyc, pxc]); p = float(corr_phase_n[pyp, pxp])
        cons = int(pc.get("in_ncc", False)) + int(pc.get("in_grad", False)) + int(pc.get("in_ctx", False)) + int(pc.get("in_phase", False))
        bonus = 0.05 if cons == 3 else (0.10 if cons == 4 else 0.0)
        rescue_score = 0.35 * n + 0.25 * g + 0.20 * c + 0.20 * p + bonus
        p5, z5 = peakfeat(corr_ncc, pxn, pyn, 5)
        p10, z10 = peakfeat(corr_ncc, pxn, pyn, 10)
        p20, z20 = peakfeat(corr_ncc, pxn, pyn, 20)
        p5g, _ = peakfeat(corr_grad, pxg, pyg, 5)
        c10, d1, d2 = density(c_ncc, pxn, pyn, 10)
        c20, _, _ = density(c_ncc, pxn, pyn, 20)
        c40, _, _ = density(c_ncc, pxn, pyn, 40)
        if 2 <= pxn < corr_ncc.shape[1] - 2 and 2 <= pyn < corr_ncc.shape[0] - 2:
            curx = corr_ncc[pyn, pxn + 1] - 2 * corr_ncc[pyn, pxn] + corr_ncc[pyn, pxn - 1]
            cury = corr_ncc[pyn + 1, pxn] - 2 * corr_ncc[pyn, pxn] + corr_ncc[pyn - 1, pxn]
        else:
            curx = cury = 0.0
        sharp = -(curx + cury)
        dist_c = float(np.hypot(cx - sw / 2, cy - sh / 2) / (sw / 2))
        dist_b = float(min(cx, sw - cx, cy, sh - cy))
        gterr = float(np.hypot(cx - gx, cy - gy)) if gtf == 1 else 999.0
        rows.append(dict(
            pair_id=pid, set_type=row["set_type"], gt_found=gtf, gt_x=gx, gt_y=gy,
            est_scale=est_scale, est_theta=est_theta, cx=cx, cy=cy, px_n=pxn, py_n=pyn,
            ncc=n, grad=g, ctx=c, phase=p, consensus=cons, rescue_score=float(rescue_score),
            ncc_pct=pct(corr_ncc, n), grad_pct=pct(corr_grad, g),
            ctx_pct=pct(corr_ctx, c), phase_pct=pct(corr_phase_n, p),
            prom5_ncc=p5, prom10_ncc=p10, prom20_ncc=p20, prom5_grad=p5g,
            z5_ncc=z5, z10_ncc=z10,
            comp10=c10, comp20=c20, comp40=c40, d1=float(d1), d2=float(d2),
            dist_center=dist_c, dist_border=dist_b, sharpness=float(sharp),
            rank_ncc=ncc_rank.get((pxn, pyn), 999),
            gterr=gterr, is_correct=int(gtf == 1 and gterr <= 5.0),
        ))
    # rank within pool by rescue_score
    rows.sort(key=lambda r: r["rescue_score"], reverse=True)
    for i, r in enumerate(rows):
        r["rank_rescue"] = i
        r["pool_size"] = len(rows)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    pairs = pd.read_csv("data/phase2_dev/pairs.csv")
    recs = [r for _, r in pairs.iterrows()]
    allrows = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        for i, rr in enumerate(ex.map(process_pair, recs)):
            allrows.extend(rr)
            if (i + 1) % 20 == 0:
                print(f"{i+1}/180  {time.time()-t0:.0f}s  rows={len(allrows)}", flush=True)
    df = pd.DataFrame(allrows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} candidate rows, {df.pair_id.nunique()} pairs)  {time.time()-t0:.0f}s")
    # quick recall report
    pr = df.groupby("pair_id").agg(gt_found=("gt_found", "first"),
                                   has_correct=("is_correct", "max"),
                                   best_gterr=("gterr", "min")).reset_index()
    present = pr[pr.gt_found == 1]
    print(f"present pairs with a <=5px candidate in pool: {present.has_correct.sum()}/{len(present)}")
