"""Single canvas->search affine + ground-truth derivation (dataset-prompt §2.2/§3).

    p_search = (1/z)·R(theta)·(p_canvas − c_canvas) + c_search
    R(theta) = [[ cos t,  sin t],
                [−sin t,  cos t]]        t = radians(theta)

Location, rotation and scale are NEVER tracked separately -- the GT point is the
reference-crop centre pushed through this same transform, so they cannot
disagree. R1 (invertibility), R2 (recoverability of z/theta) and R3 (no invented
pixels) are asserted here, not eyeballed.
"""
import numpy as np

SEARCH_PX = 1000          # search image is 1000x1000
REF_PX = 1000             # reference image is 1000x1000 at 1 nm/px


def R(theta_deg):
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, s], [-s, c]], dtype=np.float64)


def canvas_to_search(P, z, theta, c_canvas, c_search):
    """P: (...,2) canvas coords (nm) -> search pixel coords."""
    P = np.asarray(P, dtype=np.float64)
    return (R(theta) @ (P - c_canvas).T).T / z + c_search


def search_to_canvas(Q, z, theta, c_canvas, c_search):
    """Exact inverse of canvas_to_search."""
    Q = np.asarray(Q, dtype=np.float64)
    return (R(theta).T @ ((Q - c_search) * z).T).T + c_canvas


def assert_R1_R2(tol_px=1e-9):
    """Round-trip and decomposition tests over the parameter-space corners."""
    worst_rt, worst_z, worst_th = 0.0, 0.0, 0.0
    for z in (8.0, 9.3, 10.0, 12.0):
        for th in (-5.0, 0.0, 2.7, 5.0):
            cc = np.array([5000.0, 5000.0]); cs = np.array([500.0, 500.0])
            P = np.array([[0.0, 0.0], [1234.5, -987.6], [10000.0, 10000.0], [-4321.0, 8765.0]])
            Q = canvas_to_search(P, z, th, cc, cs)
            P2 = search_to_canvas(Q, z, th, cc, cs)
            worst_rt = max(worst_rt, float(np.abs(P2 - P).max()))
            # decompose the linear part: R/z  -> recover z and theta
            M = R(th) / z
            z_hat = 1.0 / np.sqrt(np.linalg.det(M))
            th_hat = np.degrees(np.arctan2(M[0, 1], M[0, 0]))
            worst_z = max(worst_z, abs(z_hat - z))
            worst_th = max(worst_th, abs(th_hat - th))
    assert worst_rt <= tol_px, f"R1 round-trip {worst_rt:.3e} px > {tol_px}"
    assert worst_z < 5e-4 and worst_th < 5e-4, f"R2 z err {worst_z:.2e}, theta err {worst_th:.2e}"
    return {"R1_roundtrip_px": worst_rt, "R2_z_err": worst_z, "R2_theta_err": worst_th}


def search_grid(z, theta, c_canvas, c_search, ss=3):
    """Canvas coords for every search pixel, `ss`x`ss` supersampled.

    Returns (Xc, Yc) each of shape (SEARCH_PX*ss, SEARCH_PX*ss). Area-averaging
    these down by `ss` is true area integration (§3.1) -- no warp, no aliasing
    from a decimated intermediate.
    """
    n = SEARCH_PX * ss
    off = (np.arange(n) + 0.5) / ss - 0.5          # sub-pixel centres in search px
    gx, gy = np.meshgrid(off, off)
    Q = np.stack([gx.ravel(), gy.ravel()], axis=1)
    P = search_to_canvas(Q, z, theta, c_canvas, c_search)
    return P[:, 0].reshape(n, n), P[:, 1].reshape(n, n)


def reference_grid(center_canvas, ss=2):
    """Canvas coords for a REF_PX x REF_PX reference at 1 nm/px around a centre."""
    n = REF_PX * ss
    off = (np.arange(n) + 0.5) / ss - REF_PX / 2.0
    gx, gy = np.meshgrid(off, off)
    return gx + center_canvas[0], gy + center_canvas[1]


def assert_R3(z, theta, c_canvas, c_search, canvas_half_nm):
    """All four search corners must map strictly inside the modelled canvas."""
    corners = np.array([[0, 0], [SEARCH_PX - 1, 0], [0, SEARCH_PX - 1], [SEARCH_PX - 1, SEARCH_PX - 1]],
                       dtype=np.float64)
    P = search_to_canvas(corners, z, theta, c_canvas, c_search)
    d = np.abs(P - c_canvas).max()
    ok = d < canvas_half_nm
    return bool(ok), float(d)


def target_center_search_to_canvas(tx, ty, z, theta, c_canvas, c_search):
    """R4: pick the target in SEARCH coords, pull it back to canvas."""
    return search_to_canvas(np.array([[tx, ty]]), z, theta, c_canvas, c_search)[0]
