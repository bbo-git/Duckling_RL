"""
gait_lib.py - build a library of random, physically checked gaits for training the
universal tracker (one policy that follows any gait from gait_gen.py).

    python gait_lib.py --n 400 --out gait_library.npz            (~1 s per gait)

Every candidate is drawn around the presets, generated with inverse kinematics, and
kept only if it is balanced (ZMP check) and within the motor speeds. A "stand still"
gait is always included so the same policy also stands.
"""
import argparse
import json
import math
import multiprocessing as mp
from dataclasses import asdict

import mujoco
import numpy as np

import gait_gen as G

RANGES = {                       # uniform sampling ranges
    "cycle_s": (0.32, 0.55), "stride_m": (-0.015, 0.040), "lift_m": (0.006, 0.016),
    "height_m": (0.050, 0.0570), "bob_m": (0.0, 0.004), "sway": (0.85, 1.1),
    "waddle": (0.0, 0.4), "double_support": (0.05, 0.18), "pitch_deg": (-4.0, 5.0),
    "nod_deg": (0.0, 16.0), "nod_counter": (0.4, 1.2), "neck_lean_deg": (-2.0, 8.0),
    "head_tilt_deg": (-16.0, -2.0), "head_sway_deg": (0.0, 12.0), "head_look_deg": (0.0, 12.0),
}
NOLOAD = {"coreless": 31.4, "servo": 10.5}
SERVO_JOINTS = ("knee", "ankle", "neck_pitch")


def check(kin, g):
    """(zmp_fraction_outside_3mm, worst_speed_ratio)"""
    m, d = kin.m, kin.d
    n = len(g["t"])
    feet_low, com = [], np.zeros((n, 3))
    for k in range(n):
        kin._set(g["root_pos"][k], g["root_quat"][k], g["joint_pos"][k])
        mujoco.mj_comPos(m, d)
        com[k] = d.subtree_com[kin.trunk]
        pts = []
        for i, side in enumerate(("left", "right")):
            if g["contact"][k, i]:
                gid = kin.legs[side]["sole"]
                mid = m.geom_dataid[gid]; va = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
                w = m.mesh_vert[va:va + nv] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
                pts.append(w[w[:, 2] < w[:, 2].min() + 0.0015][:, :2])
        feet_low.append(np.vstack(pts) if pts else None)
    dt = 1.0 / g["fps"]
    per = com[:, :2] - np.outer(np.arange(n) * dt, g["vel_xy"])
    acc = (np.roll(per, -1, 0) - 2 * per + np.roll(per, 1, 0)) / dt**2
    zmp = com[:, :2] - com[:, 2].mean() / 9.81 * acc
    from scipy.spatial import ConvexHull
    out = []
    for k, P in enumerate(feet_low):
        if P is None:
            out.append(True); continue
        try:
            eq = ConvexHull(P).equations
            out.append((eq[:, :2] @ zmp[k] + eq[:, 2]).max() > 0.003)
        except Exception:
            out.append(np.min(np.linalg.norm(P - zmp[k], axis=1)) > 0.003)
    q = g["joint_pos"]
    v = np.abs(np.diff(np.vstack([q, q[:1]]), axis=0)) * g["fps"]
    ratio = 0.0
    for i, name in enumerate(g["joint_names"]):
        lim = NOLOAD["servo"] if any(str(name).endswith(s) for s in SERVO_JOINTS) else NOLOAD["coreless"]
        ratio = max(ratio, v[:, i].max() / lim)
    return float(np.mean(out)), float(ratio)


def stand_gait(kin, n=25):
    """Stand still in HOME (both feet down, no motion)."""
    q = np.tile(kin.home, (n, 1))
    return dict(t=np.arange(n) / G.FPS, joint_pos=q, root_pos=np.tile([0, 0, 0.0556], (n, 1)),
                root_quat=np.tile([1.0, 0, 0, 0], (n, 1)), contact=np.ones((n, 2), bool),
                root_height=np.full(n, 0.0556), vel_xy=np.zeros(2), cycle_s=n / G.FPS, fps=G.FPS,
                joint_names=np.array(G.JOINT_ORDER), params=json.dumps({"stand": True}), width_m=0.047)


_kin = None


def _worker(args):
    seed, scene = args
    global _kin
    if _kin is None:
        _kin = G.DuckKinematics(scene)
    rng = np.random.default_rng(seed)
    p = G.GaitParams(width_m=0.0, **{k: float(rng.uniform(*r)) for k, r in RANGES.items()})
    g = _kin.generate(p, com_passes=3)
    zmp_out, speed = check(_kin, g)
    return seed, g, zmp_out, speed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="scene_walk.xml")
    ap.add_argument("--n", type=int, default=400, help="number of candidates")
    ap.add_argument("--out", default="gait_library.npz")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    a = ap.parse_args()
    kin = G.DuckKinematics(a.scene)
    keep = []
    for name, p in G.PRESETS.items():                   # presets always in
        g = kin.generate(p)
        zo, sp = check(kin, g)
        keep.append((name, g, zo, sp))
    rejected = 0
    with mp.Pool(a.workers) as pool:
        for seed, g, zo, sp in pool.imap_unordered(_worker, [(s, a.scene) for s in range(a.n)]):
            if zo < 0.15 and sp < 1.0:
                keep.append((f"rnd{seed}", g, zo, sp))
            else:
                rejected += 1
    keep.append(("stand", stand_gait(kin), 0.0, 0.0))
    K = len(keep)
    nmax = max(len(g["t"]) for _, g, _, _ in keep)
    J = len(G.JOINT_ORDER)
    lib = dict(names=np.array([k for k, *_ in keep]), length=np.zeros(K, np.int32),
               cycle_s=np.zeros(K), vel_xy=np.zeros((K, 2)), joint_pos=np.zeros((K, nmax, J)),
               root_height=np.zeros((K, nmax)), root_rp=np.zeros((K, nmax, 2)),
               contact=np.zeros((K, nmax, 2), bool), joint_names=np.array(G.JOINT_ORDER),
               fps=G.FPS, params=np.array([str(g["params"]) for _, g, *_ in keep]))
    for i, (name, g, zo, sp) in enumerate(keep):
        n = len(g["t"])
        lib["length"][i] = n
        lib["cycle_s"][i] = g["cycle_s"]
        lib["vel_xy"][i] = g["vel_xy"]
        lib["joint_pos"][i, :n] = g["joint_pos"]
        lib["root_height"][i, :n] = g["root_height"]
        lib["contact"][i, :n] = g["contact"]
        for k in range(n):
            w, x, y, z = g["root_quat"][k]
            lib["root_rp"][i, k] = [math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
                                    math.asin(max(-1, min(1, 2 * (w * y - z * x))))]
    np.savez(a.out, **lib)
    v = lib["vel_xy"][:, 0]
    print(f"{K} gangen opgeslagen in {a.out} ({rejected} afgekeurd); "
          f"snelheid {v.min():.3f} .. {v.max():.3f} m/s, cyclus {lib['cycle_s'].min():.2f} .. {lib['cycle_s'].max():.2f} s")


if __name__ == "__main__":
    main()
