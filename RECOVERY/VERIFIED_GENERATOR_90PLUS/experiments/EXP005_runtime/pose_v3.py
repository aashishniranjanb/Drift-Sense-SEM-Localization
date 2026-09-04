"""pose_v3 -- pose_v2's estimate at a fraction of the cost.

Profiling engine_v3 showed pose is 95% of per-pair runtime. pose_v2 spends it in
two avoidable places:

  1. It warps the FULL 1000x1000 reference for every angle evaluation (~39 of
     them). The warp only needs enough resolution to survive the subsequent
     decimation to a ~100 px template, so it is done on a pre-decimated
     OVERSAMPLE x template-size image instead -- same "rotate before final
     decimate" fidelity, ~10x less pixel work. The pre-decimation is cached per
     template size, so each unique scale pays for it once.
  2. It runs coarse branch selection at full resolution. Coarse stages only have
     to pick a branch, so they run on a 2x-downsampled search image and template.
     Fine refinement -- the part that decides the reported scale and theta, and
     therefore the pose tier -- stays at full resolution.

Same search structure and same grids as pose_v2, so the estimate is expected to
be identical up to the coarse stages' branch choice; that equivalence is measured,
not assumed (see EQUIVALENCE.md).
"""
import numpy as np
import cv2

OVERSAMPLE = 3.0          # rotate at 3x the template size, then decimate
COARSE_DOWN = 2           # coarse stages run on a 2x-downsampled search image


class _Templates:
    """Builds templates for (scale, theta), caching the per-scale pre-decimation."""

    def __init__(self, ref_f):
        self.ref = ref_f
        self.h, self.w = ref_f.shape[:2]
        self._pre = {}

    def _pre_for(self, tw, th):
        key = (tw, th)
        p = self._pre.get(key)
        if p is None:
            pw, ph = int(round(tw * OVERSAMPLE)), int(round(th * OVERSAMPLE))
            p = (cv2.resize(self.ref, (pw, ph), interpolation=cv2.INTER_AREA)
                 if pw < self.w else self.ref)
            self._pre[key] = p
        return p

    def get(self, s, theta):
        tw, th = int(round(self.w / s)), int(round(self.h / s))
        if tw < 10 or th < 10:
            return None
        pre = self._pre_for(tw, th)
        if abs(theta) > 1e-6:
            ph, pw = pre.shape[:2]
            M = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), theta, 1.0)
            pre = cv2.warpAffine(pre, M, (pw, ph), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        return cv2.resize(pre, (tw, th), interpolation=cv2.INTER_AREA)


def _peak(search_f, tpl, want_plane=False):
    if tpl is None or tpl.shape[0] >= search_f.shape[0] or tpl.shape[1] >= search_f.shape[1]:
        return (-1.0, None) if want_plane else -1.0
    r = cv2.matchTemplate(search_f, tpl, cv2.TM_CCOEFF_NORMED)
    _, mv, _, _ = cv2.minMaxLoc(r)
    return (float(mv), r) if want_plane else float(mv)


def estimate_pose_v3(ref_img, search_img, scale_min=8.0, scale_max=12.0,
                     coarse_step=0.5, n_hyp=3, n_final=2,
                     coarse_angles=(-4.0, -2.0, 0.0, 2.0, 4.0),
                     theta_min=-5.0, theta_max=5.0, rounds=2):
    ref_f = ref_img.astype(np.float32)
    search_f = search_img.astype(np.float32)
    T = _Templates(ref_f)

    # coarse stages on a downsampled search image; a template built at scale s
    # and then downsampled by d is the template for scale s*d on that image
    small = cv2.resize(search_f, (search_f.shape[1] // COARSE_DOWN,
                                  search_f.shape[0] // COARSE_DOWN),
                       interpolation=cv2.INTER_AREA)

    def coarse_score(s, th):
        t = T.get(s * COARSE_DOWN, th)
        return _peak(small, t)

    # --- stage 1: coarse scale at theta=0, keep top n_hyp branches
    cs = [(s, coarse_score(s, 0.0)) for s in np.arange(scale_min, scale_max + 1e-5, coarse_step)]
    cs = [c for c in cs if c[1] > -1.0]
    if not cs:
        return None
    hyps = [s for s, _ in sorted(cs, key=lambda c: -c[1])[:n_hyp]]

    # --- stage 2: coarse rotation per surviving branch
    branches = sorted(((coarse_score(s, th), s, th) for s in hyps for th in coarse_angles),
                      key=lambda b: -b[0])[:n_final]

    # --- stage 3: alternating fine refinement at FULL resolution.
    # The coarse stages run downsampled and can pick the wrong scale branch, so
    # the surviving branches are refined independently and the winner is decided
    # by the full-resolution score -- the downsampled ranking is never final.
    def refine(s_hat, t_hat):
        s_step, t_step = coarse_step, 2.0
        for _ in range(rounds):
            s_step /= 5.0
            t_step /= 5.0
            g = np.clip(np.arange(s_hat - 2.5 * s_step, s_hat + 2.51 * s_step, s_step),
                        scale_min, scale_max)
            s_hat = max(g, key=lambda s: _peak(search_f, T.get(s, t_hat)))
            g = np.clip(np.arange(t_hat - 2.5 * t_step, t_hat + 2.51 * t_step, t_step),
                        theta_min, theta_max)
            t_hat = max(g, key=lambda t: _peak(search_f, T.get(s_hat, t)))
        return _peak(search_f, T.get(s_hat, t_hat)), s_hat, t_hat

    _, s_hat, t_hat = max((refine(s, th) for _, s, th in branches), key=lambda r: r[0])

    tpl = T.get(s_hat, t_hat)
    v, plane = _peak(search_f, tpl, want_plane=True)
    return {"best_scale": float(s_hat), "best_theta": float(t_hat),
            "best_score": float(v), "best_template": tpl, "corr_plane": plane}
