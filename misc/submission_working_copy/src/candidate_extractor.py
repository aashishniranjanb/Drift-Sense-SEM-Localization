"""Top-K correlation-peak candidate extraction with centre/periphery sector
allocation, plus periodic-replica family clustering. Verbatim frozen logic."""
import numpy as np
import cv2


def extract_nms_fast(corr_plane, tw, th, max_k=200, r=5):
    ch, cw = corr_plane.shape[:2]
    work = corr_plane.copy()
    out = []
    for rank in range(max_k):
        _, mv, _, ml = cv2.minMaxLoc(work)
        if mv <= 0.01 or np.isnan(mv):
            break
        px, py = ml
        out.append({"peak_x": px, "peak_y": py, "cx": px + tw / 2.0, "cy": py + th / 2.0,
                    "corr_score": float(mv), "raw_rank": rank + 1})
        y1, y2 = max(0, py - r), min(ch, py + r + 1)
        x1, x2 = max(0, px - r), min(cw, px + r + 1)
        work[y1:y2, x1:x2] = -999.0
    return out


def extract_candidates_akhilesh(corr_plane, tw, th, ref_img=None, search_img=None,
                                est_scale=10.0, est_theta=0.0, max_final_k=200):
    sh, sw = search_img.shape[:2] if search_img is not None else (1024, 1024)
    scx, scy = sw / 2.0, sh / 2.0
    pool = extract_nms_fast(corr_plane, tw, th, max_k=max_final_k * 3, r=5)
    if len(pool) <= max_final_k:
        return pool
    center, periph = [], []
    for c in pool:
        d = float(np.hypot(c["cx"] - scx, c["cy"] - scy))
        c["dist_to_center"] = d
        (center if d <= 260.0 else periph).append(c)
    for c in center:
        c["center_priority"] = c["corr_score"] - 0.05 * (c["dist_to_center"] / 260.0) ** 2
    center.sort(key=lambda x: x["center_priority"], reverse=True)
    periph.sort(key=lambda x: x["corr_score"], reverse=True)
    n_c = min(int(max_final_k * 0.7), len(center))
    n_p = min(max_final_k - n_c, len(periph))
    final = center[:n_c] + periph[:n_p]
    if len(final) < max_final_k:
        rest = center[n_c:] + periph[n_p:]
        rest.sort(key=lambda x: x["corr_score"], reverse=True)
        final += rest[:max_final_k - len(final)]
    return final[:max_final_k]


def cluster_replica_families(candidates, scale):
    if not candidates:
        return candidates
    pitches = np.array([32.0, 36.0, 48.0, 128.0]) / scale
    n = len(candidates)
    adj = np.eye(n, dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(candidates[i]["cx"] - candidates[j]["cx"])
            dy = abs(candidates[i]["cy"] - candidates[j]["cy"])
            xm = any(abs(dx - p * round(dx / p)) < 2.0 and round(dx / p) > 0 for p in pitches)
            ym = any(abs(dy - p * round(dy / p)) < 2.0 and round(dy / p) > 0 for p in pitches)
            if (xm and dy < 2.5) or (ym and dx < 2.5):
                adj[i, j] = adj[j, i] = True
    visited = np.zeros(n, dtype=bool)
    fam = 0
    for i in range(n):
        if visited[i]:
            continue
        q = [i]
        visited[i] = True
        comp = []
        head = 0
        while head < len(q):
            u = q[head]
            head += 1
            comp.append(u)
            for v in range(n):
                if adj[u, v] and not visited[v]:
                    visited[v] = True
                    q.append(v)
        scores = [candidates[k]["corr_score"] for k in comp]
        pop = len(comp)
        var = float(np.var(scores)) if pop > 1 else 0.0
        for k in comp:
            candidates[k]["family_id"] = fam
            candidates[k]["family_population"] = pop
            candidates[k]["family_score_variance"] = var
        fam += 1
    return candidates
