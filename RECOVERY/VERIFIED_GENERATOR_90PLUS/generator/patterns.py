"""Procedural SEM layout fields.

Every pattern is a function f(X, Y, preset) -> float array in [0,1], evaluable on
an ARBITRARY float grid of canvas coordinates (nm). Nothing is rasterised to a
full canvas, so a z=12 field of view (12000^2 nm) costs nothing.

Two families, each with several presets (different pitches / linewidths), plus a
large-scale ZONE modulation (mats separated by routing strips) so the field is
not purely periodic -- that global structure is what distinguishes the true site
from a lattice replica.
"""
import numpy as np

# ---------------------------------------------------------------- presets
DRAM_PRESETS = {
    "dram_1x":      dict(bl_pitch=64.0, wl_pitch=88.0, bl_w=26.0, wl_w=32.0, cont_r=17.0),
    "dram_dense":   dict(bl_pitch=48.0, wl_pitch=66.0, bl_w=20.0, wl_w=24.0, cont_r=13.0),
    "dram_loose":   dict(bl_pitch=86.0, wl_pitch=118.0, bl_w=34.0, wl_w=42.0, cont_r=23.0),
    "dram_wide":    dict(bl_pitch=104.0, wl_pitch=76.0, bl_w=44.0, wl_w=28.0, cont_r=21.0),
    "dram_compact": dict(bl_pitch=56.0, wl_pitch=56.0, bl_w=22.0, wl_w=22.0, cont_r=14.0),
    "dram_legacy":  dict(bl_pitch=128.0, wl_pitch=150.0, bl_w=52.0, wl_w=58.0, cont_r=30.0),
}
FINFET_PRESETS = {
    "finfet_7nm":  dict(fin_pitch=34.0, gate_pitch=108.0, fin_w=11.0, gate_w=36.0, cont_r=12.0),
    "finfet_10nm": dict(fin_pitch=44.0, gate_pitch=128.0, fin_w=15.0, gate_w=44.0, cont_r=15.0),
    "finfet_14nm": dict(fin_pitch=56.0, gate_pitch=156.0, fin_w=19.0, gate_w=54.0, cont_r=18.0),
    "finfet_22nm": dict(fin_pitch=76.0, gate_pitch=196.0, fin_w=27.0, gate_w=68.0, cont_r=24.0),
    "finfet_28nm": dict(fin_pitch=94.0, gate_pitch=240.0, fin_w=34.0, gate_w=84.0, cont_r=29.0),
    "finfet_45nm": dict(fin_pitch=136.0, gate_pitch=330.0, fin_w=50.0, gate_w=118.0, cont_r=41.0),
}
ALL_PRESETS = {**{k: ("dram", v) for k, v in DRAM_PRESETS.items()},
               **{k: ("finfet", v) for k, v in FINFET_PRESETS.items()}}


def _stripe(coord, pitch, width, soft=2.0):
    """Smooth periodic stripe: 1 inside a `width` band every `pitch`."""
    d = np.abs(((coord + pitch * 0.5) % pitch) - pitch * 0.5)
    return 1.0 / (1.0 + np.exp((d - width * 0.5) / soft))


def _dots(X, Y, px, py, r, soft=2.0, ox=0.0, oy=0.0, stagger=False):
    """Smooth periodic dot lattice; optional row stagger (6F^2-like)."""
    yy = Y - oy
    row = np.floor((yy + py * 0.5) / py)
    sx = np.where((row % 2 == 1) & stagger, px * 0.5, 0.0) if stagger else 0.0
    dx = ((X - ox - sx + px * 0.5) % px) - px * 0.5
    dy = ((yy + py * 0.5) % py) - py * 0.5
    d = np.hypot(dx, dy)
    return 1.0 / (1.0 + np.exp((d - r) / soft))


def _hash01(ix, iy, salt=0):
    """Deterministic vectorised hash of an integer mat index -> [0,1)."""
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ np.int64(salt * 83492791)
    h = (h ^ (h >> 13)) * np.int64(1274126177)
    h = h ^ (h >> 16)
    return (h & np.int64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def _zone_apply(X, Y, p, mat_nm, strip_nm, phase_x, phase_y, px_key, py_key):
    """Large-scale mat/strip structure WITH per-mat variation.

    Each mat gets its own lattice phase offset and gain, keyed by a hash of its
    index. Without this every mat is interchangeable, a crop taken inside one
    repeats across the whole field, and no label can ever be verified -- which is
    exactly the defect measured in data/phase2_dev. Returns
    (X_shifted, Y_shifted, mat_gain, strip_field, in_mat).
    """
    per = mat_nm + strip_nm
    ix = np.floor((X - phase_x) / per)
    iy = np.floor((Y - phase_y) / per)
    ox = (_hash01(ix, iy, 1) - 0.5) * p[px_key]      # sub-pitch phase, up to +/- half pitch
    oy = (_hash01(ix, iy, 2) - 0.5) * p[py_key]
    gain = 0.72 + 0.56 * _hash01(ix, iy, 3)          # per-mat brightness
    fx = (X - phase_x) % per
    fy = (Y - phase_y) % per
    in_mat = ((fx < mat_nm) & (fy < mat_nm)).astype(np.float32)
    sx = _stripe(X - phase_x - mat_nm - strip_nm * 0.5, per, strip_nm * 0.55, soft=3.0)
    sy = _stripe(Y - phase_y - mat_nm - strip_nm * 0.5, per, strip_nm * 0.55, soft=3.0)
    return X + ox, Y + oy, gain, np.maximum(sx, sy), in_mat


def dram_field(X, Y, p, zone=None):
    if zone is not None:
        Xs, Ys, gain, strip, in_mat = _zone_apply(X, Y, p, *zone, "bl_pitch", "wl_pitch")
    else:
        Xs, Ys, gain, strip, in_mat = X, Y, 1.0, 0.0, 1.0
    bl = _stripe(Xs, p["bl_pitch"], p["bl_w"])
    wl = _stripe(Ys, p["wl_pitch"], p["wl_w"])
    ct = _dots(Xs, Ys, p["bl_pitch"], p["wl_pitch"], p["cont_r"],
               ox=p["bl_pitch"] * 0.5, oy=p["wl_pitch"] * 0.5, stagger=True)
    arr = (0.30 * bl + 0.26 * wl + 0.62 * ct) * gain
    if zone is not None:
        arr = arr * (0.30 + 0.70 * in_mat) + 0.60 * strip
    return np.clip(arr, 0.0, 1.0)


def finfet_field(X, Y, p, zone=None):
    if zone is not None:
        Xs, Ys, gain, strip, in_mat = _zone_apply(X, Y, p, *zone, "fin_pitch", "gate_pitch")
    else:
        Xs, Ys, gain, strip, in_mat = X, Y, 1.0, 0.0, 1.0
    fin = _stripe(Xs, p["fin_pitch"], p["fin_w"], soft=1.6)
    gate = _stripe(Ys, p["gate_pitch"], p["gate_w"], soft=2.4)
    ct = _dots(Xs, Ys, p["fin_pitch"] * 2.0, p["gate_pitch"], p["cont_r"],
               oy=p["gate_pitch"] * 0.5)
    arr = (0.34 * fin + 0.46 * gate + 0.40 * ct) * gain
    if zone is not None:
        arr = arr * (0.30 + 0.70 * in_mat) + 0.60 * strip
    return np.clip(arr, 0.0, 1.0)


def make_field(preset_name, rng, zone_scale=1.0, linewidth_bias=0.0):
    """Return (field_fn, kind, params). field_fn(X, Y) -> [0,1]."""
    kind, base = ALL_PRESETS[preset_name]
    p = dict(base)
    for k in p:
        if k.endswith("_w") or k == "cont_r":
            p[k] = max(4.0, p[k] + linewidth_bias)
    # zone geometry: mat/strip in nm, randomized phase
    mat = float(rng.uniform(1900, 3200) * zone_scale)
    strip = float(rng.uniform(240, 430) * zone_scale)
    px = float(rng.uniform(0, mat + strip))
    py = float(rng.uniform(0, mat + strip))
    zone = (mat, strip, px, py)
    fn = dram_field if kind == "dram" else finfet_field

    def field(X, Y):
        return fn(X, Y, p, zone=zone)
    return field, kind, {"preset": preset_name, "kind": kind, "zone_mat_nm": mat,
                         "zone_strip_nm": strip, "zone_phase_x": px, "zone_phase_y": py,
                         "linewidth_bias_nm": linewidth_bias, **p}


def preset_names(kind=None):
    if kind == "dram":
        return list(DRAM_PRESETS)
    if kind == "finfet":
        return list(FINFET_PRESETS)
    return list(ALL_PRESETS)
