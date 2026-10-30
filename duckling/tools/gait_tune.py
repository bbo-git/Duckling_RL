"""
gait_tune.py - zoekt loopparameters die de servo's aankunnen en in balans zijn, met behoud van
het karakter van een stijl. Begint bij een opgeslagen loopje (JSON) en past alleen de vrije
parameters aan binnen grenzen; de karakter-parameters liggen vast of hebben een smalle band.

    python gait_tune.py favorites/normaal_v1.json --stijl normaal --out favorites/normaal_v2.json
    python gait_tune.py favorites/parmantig_v1.json --stijl parmantig --out favorites/parmantig_v2.json

Score (lager = beter):
  * ZMP buiten de voet (dynamische balans)
  * zool zijwaarts gekanteld bij steun (geen enkel-roll: kanteling = op de rand staan)
  * gewrichtssnelheid boven 80% van de servo (9 rad/s)
  * steunvoet die schuift (IK haalt het doel niet: gewrichtslimiet)
  * botsingen tussen beenonderdelen
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gait_gen import GaitParams, DuckKinematics, JOINT_ORDER  # noqa: E402

LIM = {"hip": 9.0, "knee": 9.0, "ankle": 9.0, "neck": 9.0, "head": 8.1}
SPEED_FRAC = 0.8

# vrije parameters: (naam, ondergrens, bovengrens)
FREE = {
    "normaal": [
        ("height_m", 0.058, 0.068), ("cycle_s", 0.45, 0.70), ("stride_m", 0.030, 0.050),
        ("lift_m", 0.004, 0.012), ("width_m", 0.040, 0.070), ("sway", 0.4, 1.3),
        ("waddle", 0.0, 0.4), ("double_support", 0.05, 0.25), ("bob_m", 0.0, 0.005),
        ("dip_m", 0.0, 0.005), ("com_ahead_m", -0.004, 0.010), ("push_deg", 0.0, 30.0),
        ("swing_toe_deg", 0.0, 20.0), ("lift_hip_deg", 0.0, 15.0), ("lift_enkel_deg", 0.0, 15.0),
        ("lift_timing", -0.3, 0.3), ("pitch_deg", -2.0, 4.0),
    ],
    # parmantig: hoog marcheren (laat optillen met de heup), borst vooruit, kort en kordaat
    "parmantig": [
        ("height_m", 0.060, 0.068), ("cycle_s", 0.38, 0.52), ("stride_m", 0.025, 0.042),
        ("lift_m", 0.006, 0.014), ("width_m", 0.040, 0.070), ("sway", 0.4, 1.3),
        ("waddle", 0.0, 0.25), ("double_support", 0.05, 0.22), ("bob_m", 0.002, 0.006),
        ("dip_m", 0.0, 0.005), ("com_ahead_m", -0.004, 0.010), ("push_deg", 0.0, 30.0),
        ("swing_toe_deg", 10.0, 25.0), ("lift_hip_deg", 18.0, 30.0), ("lift_enkel_deg", 0.0, 15.0),
        ("lift_timing", 0.35, 0.6), ("pitch_deg", -6.0, -3.0), ("lift_toe_follow", 0.2, 1.0),
    ],
}


GRID_VARIANTS = []   # --rooster: (snelheid, bocht) waarop ook gescoord wordt


class Scorer:
    def __init__(self, scene):
        self.kin = DuckKinematics(scene)
        m = self.kin.m
        self.soles = [m.geom(f"{s}_foot_collision").id for s in ("left", "right")]
        self.sites = [m.site(f"{s}_foot").id for s in ("left", "right")]
        self.toes = [m.body(f"{b}_toe").id for b in ("L", "R")]
        # botsingsmodel: convex hulls van alle beenonderdelen
        spec = mujoco.MjSpec.from_file(scene)
        for g in spec.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH:
                g.contype, g.conaffinity = 1, 1
        self.mh = spec.compile(); self.dh = mujoco.MjData(self.mh)
        self.dh.qpos[2] = 0.0592; self.dh.qpos[3] = 1
        mujoco.mj_forward(self.mh, self.dh)
        self.base = {frozenset((c.geom1, c.geom2)) for c in self.dh.contact[:self.dh.ncon]}
        self.floor = self.mh.geom("floor").id
        self.qadr_h = np.array([self.mh.jnt_qposadr[self.mh.joint(n).id] for n in JOINT_ORDER])
        self.head_bodies = {self.mh.body(b).id for b in ("neck", "head_pitch", "head_roll", "head", "jaw")}

    def metrics(self, p):
        k = self.kin; m, d = k.m, k.d
        g = k.generate(p)
        n = len(g["t"]); fps = float(g["fps"])
        q = g["joint_pos"]
        # 2026-10-29: snelheden per SERVO (differentiëlen: 1,5 x (pitch +- roll)), niet per gewricht
        import duckling_servo as DS
        sp = DS.servo_speeds(q, fps, JOINT_ORDER)
        over = sum(max(0.0, v - SPEED_FRAC * lim) for v, lim in sp.values())
        w = DS.worst_by_group(sp)
        top = {"hip_pitch": w["heupservo"][0], "knee": w["enkel"][0], "ankle": w["teen"][0],
               "hip_yaw": w["heup_yaw"][0], "head": w["kop"][0]}
        # balans (ZMP), zoolkanteling, voetslip
        coms, polys, roll, pitch, slip = np.zeros((n, 3)), [], [], [], []
        prev = None
        for i in range(n):
            k._set(g["root_pos"][i], g["root_quat"][i], q[i])
            mujoco.mj_comPos(m, d)
            coms[i] = d.subtree_com[k.trunk]
            pts = []
            cur = []
            for s, gid in enumerate(self.soles):
                cur.append(d.site_xpos[self.sites[s]].copy())
                if g["contact"][i, s]:
                    mid = m.geom_dataid[gid]; va = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
                    w = m.mesh_vert[va:va + nv] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
                    pts.append(w[w[:, 2] < w[:, 2].min() + 0.0015][:, :2])
                    R = d.xmat[self.toes[s]].reshape(3, 3)
                    roll.append(abs(math.degrees(math.asin(np.clip(R[2, 1], -1, 1)))))
            if prev is not None:
                for s in range(2):
                    if g["contact"][i, s] and g["contact"][i - 1, s] and p.push_deg == 0:
                        slip.append(np.linalg.norm((cur[s] - prev[s])[:2]))
            prev = cur
            polys.append(np.vstack(pts) if pts else None)
        dt = 1 / fps
        trend = np.outer(np.arange(n) * dt, g["vel_xy"])
        per = coms[:, :2] - trend
        acc = (np.roll(per, -1, 0) - 2 * per + np.roll(per, 1, 0)) / dt ** 2
        zmp = coms[:, :2] - coms[:, 2].mean() / 9.81 * acc
        from scipy.spatial import ConvexHull
        zm = []
        for i, P in enumerate(polys):
            if P is None:
                continue
            try:
                eq = ConvexHull(P).equations
                zm.append(-(eq[:, :2] @ zmp[i] + eq[:, 2]).max())
            except Exception:
                zm.append(-np.min(np.linalg.norm(P - zmp[i], axis=1)))
        zm = np.array(zm) * 1000
        # botsingen (benen; kop telt niet mee)
        pen = 0.0
        for i in range(0, n, 2):
            self.dh.qpos[:3] = g["root_pos"][i]; self.dh.qpos[3:7] = g["root_quat"][i]
            self.dh.qpos[self.qadr_h] = q[i]
            mujoco.mj_forward(self.mh, self.dh)
            for c in self.dh.contact[:self.dh.ncon]:
                pr = frozenset((c.geom1, c.geom2))
                if self.floor in pr or pr in self.base:
                    continue
                b1, b2 = self.mh.geom_bodyid[c.geom1], self.mh.geom_bodyid[c.geom2]
                if b1 in self.head_bodies or b2 in self.head_bodies:
                    continue
                pen = min(pen, c.dist)
        # IK: voet haalt doel niet -> gewricht op limiet
        lim_hit = 0
        for j, nm in enumerate(JOINT_ORDER):
            if nm.startswith(("left", "right")):
                lo, hi = m.jnt_range[m.joint(nm).id]
                lim_hit += int(((q[:, j] < lo + 1e-3) | (q[:, j] > hi - 1e-3)).sum())
        return dict(zmp_out=float(np.mean(zm < -3)), zmp_worst=float(max(0, -zm.min())),
                    roll_mean=float(np.mean(roll)) if roll else 0, roll_max=float(np.max(roll)) if roll else 0,
                    speed_over=float(over), top=top, collide_mm=float(-pen * 1000), lim_hit=lim_hit,
                    speed=float(g["vel_xy"][0]))

    ROLL_W = 1.5

    @classmethod
    def score(cls, r):
        return (10 * r["zmp_out"] + 0.3 * r["zmp_worst"] + cls.ROLL_W * max(0, r["roll_mean"] - 3)
                + 0.3 * cls.ROLL_W * max(0, r["roll_max"] - 8) + 3 * r["speed_over"] + 5 * r["collide_mm"]
                + 1.0 * r["lim_hit"])


def tune(scorer, p0, free, iters, seed=0, log=print):
    rng = np.random.default_rng(seed)
    names = [f[0] for f in free]
    lo = np.array([f[1] for f in free]); hi = np.array([f[2] for f in free])
    x = np.clip(np.array([getattr(p0, nm) for nm in names], float), lo, hi)

    def P(xx):
        p = GaitParams(**asdict(p0))
        for nm, val in zip(names, xx):
            setattr(p, nm, float(val))
        return p

    def evaluate(p):
        if not GRID_VARIANTS:
            r = scorer.metrics(p); return scorer.score(r), r
        import walk_speed as W
        st = W.Style(p)
        tot, rr = 0.0, None
        for sp, turn in GRID_VARIANTS:            # rooster-gangen: de regels per snelheid erop
            q = st.params(sp); q.turn_dps = turn
            r = scorer.metrics(q); tot += scorer.score(r)
            if sp == 1.0 and turn == 0.0:
                rr = r
            rr = rr or r
            for k in ("knee", "ankle", "hip_pitch"):
                rr["top"][k] = max(rr["top"][k], r["top"][k])
        return tot / len(GRID_VARIANTS), rr

    def evaluate_keep(p):   # karakter houden: niet trager worden dan 95% van de start
        s, r = evaluate(p)
        return s + 400 * max(0.0, v_min[0] - r["speed"]), r

    v_min = [0.0]
    best, best_r = evaluate(P(x))
    v_min[0] = 0.95 * best_r["speed"]
    log(f"start: score {best:.2f}  {fmt(best_r)}")
    sigma = 0.15
    t0 = time.time()
    for it in range(iters):
        cand = np.clip(x + sigma * (hi - lo) * rng.standard_normal(len(x)) * (rng.random(len(x)) < 0.4), lo, hi)
        s, r = evaluate_keep(P(cand))
        if s < best:
            x, best, best_r = cand, s, r
            sigma = min(0.3, sigma * 1.2)
            log(f"{it:4d} score {best:.2f}  {fmt(best_r)}  ({time.time() - t0:.0f} s)")
        else:
            sigma = max(0.02, sigma * 0.98)
    return P(x), best_r


def fmt(r):
    t = r["top"]
    return (f"ZMP buiten {100 * r['zmp_out']:.0f}% (max {r['zmp_worst']:.0f} mm), zool-roll gem {r['roll_mean']:.1f} max {r['roll_max']:.0f} gr, "
            f"heupservo {t['hip_pitch']:.1f} enkel {t['knee']:.1f} teen {t['ankle']:.1f} rad/s, botsing {r['collide_mm']:.1f} mm, limiet {r['lim_hit']}, "
            f"{r['speed'] * 100:.1f} cm/s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("params")
    ap.add_argument("--stijl", choices=list(FREE), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--scene", default="scene_walk.xml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rooster", action="store_true",
                    help="ook langzaam en in bochten scoren (de regels van walk_speed.py)")
    ap.add_argument("--roll", type=float, default=1.5, help="gewicht zool-kanteling (hoger = vlakkere zolen)")
    ap.add_argument("--grens", nargs="*", default=[],
                    help="eigen grenzen, bv. width_m=0.045:0.054 (voeten smaller) of lift_hip_deg=20:30")
    a = ap.parse_args()
    p0 = GaitParams(**{k: float(v) for k, v in json.load(open(a.params)).items()})
    sc = Scorer(a.scene)
    Scorer.ROLL_W = a.roll
    if a.rooster:
        GRID_VARIANTS[:] = [(0.2, 0.0), (0.6, 0.0), (1.0, 0.0), (1.0, 45.0), (0.0, 90.0)]
    free = list(FREE[a.stijl])
    for g in a.grens:
        k, rng = g.split("=")
        lo, hi = (float(x) for x in rng.split(":"))
        free = [f for f in free if f[0] != k] + [(k, lo, hi)]
    p, r = tune(sc, p0, free, a.iters, a.seed, log=lambda s: print(s, flush=True))
    with open(a.out, "w") as f:
        json.dump(asdict(p), f, indent=1)
    print("opgeslagen:", a.out)
    print("eind:", fmt(r))


if __name__ == "__main__":
    main()
