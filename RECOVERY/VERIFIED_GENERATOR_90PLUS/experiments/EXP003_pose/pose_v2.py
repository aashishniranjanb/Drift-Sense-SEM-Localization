"""EXP003 -- pose_v2: joint scale/rotation estimation.

Three defects in Engine B's sequential pose search, measured in EXP002:

  1. The coarse SCALE sweep runs at theta=0. Under the dataset's +/-5 deg rotation
     the template no longer matches, so the first stage's argmax is unreliable --
     median scale error rises 0.28% -> 2.22% as |theta| goes 0-1 deg -> 3-4 deg,
     and 60% of pairs in that band miss the rubric's 2% tier.
  2. Rotation is applied to the ~100 px TEMPLATE, after decimation. Rotating a
     tiny image loses fidelity and BORDER_REFLECT contaminates the edges.
  3. Theta is quantized at 0.25 deg -- exactly the top pose tier's boundary.

pose_v2:
  * keeps the top-N coarse scale hypotheses instead of only the argmax, so a
     rotation-corrupted coarse stage cannot lock the search onto a wrong branch;
  * rotates the FULL-RESOLUTION reference, then decimates (INTER_AREA), so the
     template is built the same way at every angle;
  * alternates fine scale / fine theta refinement to a finer step than either
     tier boundary.

Nothing is fitted. All constants are search-grid parameters, not tuned thresholds.
"""
import numpy as np
import cv2


def _tpl(ref_f, s, theta):
    """Rotate at full resolution, then decimate. Returns None if degenerate."""
    h, w = ref_f.shape[:2]
    if abs(theta) > 1e-6:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), theta, 1.0)
        rot = cv2.warpAffine(ref_f, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    else:
        rot = ref_f
    tw, th = int(round(w / s)), int(round(h / s))
    if tw < 10 or th < 10:
        return None
    return cv2.resize(rot, (tw, th), interpolation=cv2.INTER_AREA)


def _score(ref_f, search_f, s, theta, want_plane=False):
    t = _tpl(ref_f, s, theta)
    if t is None or t.shape[0] >= search_f.shape[0] or t.shape[1] >= search_f.shape[1]:
        return (-1.0, None, None) if want_plane else -1.0
    r = cv2.matchTemplate(search_f, t, cv2.TM_CCOEFF_NORMED)
    _, mv, _, _ = cv2.minMaxLoc(r)
    return (float(mv), r, t) if want_plane else float(mv)


def estimate_pose_v2(ref_img, search_img, scale_min=8.0, scale_max=12.0,
                     coarse_step=0.5, n_hyp=3,
                     coarse_angles=(-4.0, -2.0, 0.0, 2.0, 4.0),
                     theta_min=-5.0, theta_max=5.0, rounds=2):
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)

    # --- stage 1: coarse scale at theta=0, keep the top n_hyp branches
    coarse = [(s, _score(ref_f, search_f, s, 0.0))
              for s in np.arange(scale_min, scale_max + 1e-5, coarse_step)]
    coarse = [c for c in coarse if c[1] > -1.0]
    if not coarse:
        return None
    hyps = [s for s, _ in sorted(coarse, key=lambda c: -c[1])[:n_hyp]]

    # --- stage 2: coarse rotation for each surviving scale branch
    best = (-1.0, hyps[0], 0.0)
    for s in hyps:
        for th in coarse_angles:
            v = _score(ref_f, search_f, s, th)
            if v > best[0]:
                best = (v, s, th)
    _, s_hat, t_hat = best

    # --- stage 3: alternating fine refinement, finer than either tier boundary
    s_step, t_step = coarse_step, 2.0
    for _ in range(rounds):
        s_step /= 5.0            # 0.5 -> 0.1 -> 0.02
        t_step /= 5.0            # 2.0 -> 0.4 -> 0.08
        grid = np.clip(np.arange(s_hat - 2.5 * s_step, s_hat + 2.51 * s_step, s_step),
                       scale_min, scale_max)
        s_hat = max(grid, key=lambda s: _score(ref_f, search_f, s, t_hat))
        grid = np.clip(np.arange(t_hat - 2.5 * t_step, t_hat + 2.51 * t_step, t_step),
                       theta_min, theta_max)
        t_hat = max(grid, key=lambda t: _score(ref_f, search_f, s_hat, t))

    v, plane, tpl = _score(ref_f, search_f, s_hat, t_hat, want_plane=True)
    return {"best_scale": float(s_hat), "best_theta": float(t_hat),
            "best_score": float(v), "best_template": tpl, "corr_plane": plane}
