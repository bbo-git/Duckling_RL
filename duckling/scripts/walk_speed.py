"""
walk_speed.py - your characteristic walk, steerable: speed, turning and sidestepping.

Your walk is the 100% speed. Every other speed is derived from it with a few rules:
slow = relaxed and cheerful (head up, happy bob, looks around), fast = hurried (quicker
steps, leaning forward, more head motion, waddle and doorveren). On top of that every
speed exists turning (-90..90 deg/s, turning on the spot included) and sidestepping
(-10..10 mm per cycle; with turning together that is an orbit walk).

All of it is one grid of gaits (speed x turn x side) generated once. At runtime the
reference is blended between the neighbouring gaits, so the command is continuous; going
to speed 0 blends into standing still. The head has its own layer on top (look_yaw /
look_pitch), e.g. for following a face with the camera.

    python walk_speed.py --view                       # sliders: snelheid_pct, draai_gr_s, zijstap_mm
    python walk_speed.py --check                      # balance / motors per gait
    python walk_speed.py --print 40                   # parameters at 40% (to make an anchor)
    python walk_speed.py --anchor 40=favorites/langzaam.npz --view
    python walk_speed.py --grid gait_grid.npz         # for training and playback
    python walk_speed.py --load gait_grid.npz         # play an existing grid (no physics)
"""
import argparse
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict, fields

import mujoco
import numpy as np

import gait_gen as G

STYLE = "favorites/jairo_walk_v3.npz"
SPEEDS = np.round(np.concatenate([[-0.5, -0.35, -0.2, 0.0], np.linspace(0.2, 1.0, 9), [1.15, 1.3]]), 3)
# negative = walking backwards (2026-10-13): your walk at |speed|, relaxed, steps reversed
TURNS = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])          # deg/s, + = left
SIDES = np.array([-0.010, 0.0, 0.010])                     # m per cycle, + = left
HOME_H = 0.0556
STANCE_WIDTH = 0.0                                         # m between the feet (0 = automatic)
FOLLOW = [True]
# what the knee servos can follow (factors on your walk); KNEE_MAX = warning level in --check
SERVO_FIT = dict(cycle=1.05, cadence=0.05, lift=0.8, dip=1.0, bob=0.3)   # dip: your own value (v3); bob kept low
KNEE_MAX = 8.0                                             # rad/s, ~60% of the servo no-load speed


def fast_model(m):
    """Shadows and reflections off (much higher fps on laptops); --mooi keeps them.
    Done on the model before the viewer opens, so it works with every MuJoCo version."""
    import sys
    if "--mooi" in sys.argv:
        return
    m.vis.quality.shadowsize = 0          # no shadow map
    m.mat_reflectance[:] = 0              # no floor reflection


def fast_render(v):
    """Same via the viewer's scene flags, where this MuJoCo version allows it."""
    import sys
    if "--mooi" in sys.argv:
        return
    try:
        with v.lock():
            v.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            v.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    except Exception:
        pass


def load_params(path):
    p = json.loads(str(np.load(path)["params"]))
    p["width_m"] = 0.0                                       # always automatic
    return G.GaitParams(**{f.name: float(p[f.name]) for f in fields(G.GaitParams) if f.name in p})


def rule_params(base: G.GaitParams, s: float) -> G.GaitParams:
    """Character over speed (100% = your walk, unchanged):
    slow  = relaxed and cheerful: upright, head up, looks around a lot, a happy bob
    fast  = hurried: quicker cadence, leans forward, head forward, nods/sways/springs more"""
    v0 = base.stride_m / base.cycle_s
    p = G.GaitParams(**asdict(base))
    sv = s                                    # signed: negative = backwards
    s = abs(s)                                # the character follows |speed|
    slow = max(0.0, 1.0 - s)                  # 0.8 at 20%, 0 from 100%
    fast = max(0.0, s - 1.0)                  # 0.3 at 130%
    # SERVO LIMIT: the knee servo has only ~25% torque left at 10 rad/s (no-load 13 rad/s), so
    # the knee must stay around <= 7 rad/s or the policy cannot track it (2026-10-13: the
    # original walk needed 12-17 rad/s and every policy trained on it became a jittery
    # wide-legged "tapdancer"). Hence: cycle +5%, hardly any cadence increase when fast,
    # 80% foot lift that stops growing above 100%, less dip and bob (see SERVO_FIT).
    p.cycle_s = base.cycle_s * SERVO_FIT["cycle"] * (1 + 0.05 * slow - SERVO_FIT["cadence"] * fast)
    # more than +5% and the swing leg runs out of hip roll (20 deg), so slow = mostly shorter steps
    p.stride_m = sv * v0 * p.cycle_s          # backwards: the same steps, reversed
    p.lift_m = base.lift_m * SERVO_FIT["lift"] * (0.6 + 0.4 * min(s, 1.0))
    # body: whole-gait amounts grow with speed ("faster = more")
    body = 0.3 + 0.7 * s
    p.waddle = base.waddle * body
    p.dip_m = base.dip_m * body * SERVO_FIT["dip"]
    p.bob_m = base.bob_m * (0.8 + 0.2 * s) * SERVO_FIT["bob"]     # happy bounce stays when slow
    # head: rhythmic motion grows faster than the body
    head = 0.15 + 0.85 * s ** 1.5                                     # 20%: 0.23, 130%: 1.41
    for k in ("nod_deg", "head_sway_deg", "head_follow_deg"):
        setattr(p, k, getattr(base, k) * head)
    p.head_look_deg = base.head_look_deg * max(0.3, 1 - 2.0 * fast)  # rhythmic; the calm looking
                                                                      # around is look_around() below
    p.head_tilt_deg = base.head_tilt_deg - 6.0 * slow + 12.0 * fast  # head up when slow, forward when hurried
    p.neck_lean_deg = base.neck_lean_deg - 3.0 * slow + 12.0 * fast
    p.pitch_deg = base.pitch_deg + 25.0 * fast                        # hurried: trunk leans forward
    if sv < 0:                      # backwards: careful - lower feet (stepping back costs
        p.lift_m *= 0.75            # more knee speed; a longer cycle made it worse)
    p.width_m = STANCE_WIDTH        # narrower than the vertical-leg width; the rounded
    return p                          # soles take the few degrees of sole tilt


class Style:
    """Rules + optional hand-made anchors (corrections interpolated over speed)."""

    def __init__(self, base, anchors=None):
        self.base = base
        self.anchors = {1.0: {}}                              # s -> {param: correction}
        for s, p in (anchors or {}).items():
            r = rule_params(base, s)
            self.anchors[s] = {f.name: getattr(p, f.name) - getattr(r, f.name)
                               for f in fields(G.GaitParams) if f.name != "width_m"}

    def params(self, s):
        p = rule_params(self.base, s)
        ss = sorted(self.anchors)
        if len(ss) == 1:
            return p
        i = np.searchsorted(ss, s)
        if i == 0:
            corr = self.anchors[ss[0]]; lo = hi = corr; w = 0.0
        elif i == len(ss):
            corr = self.anchors[ss[-1]]; lo = hi = corr; w = 0.0
        else:
            a, b = ss[i - 1], ss[i]
            lo, hi, w = self.anchors[a], self.anchors[b], (s - a) / (b - a)
        for f in fields(G.GaitParams):
            if f.name == "width_m":
                continue
            c = (1 - w) * lo.get(f.name, 0.0) + w * hi.get(f.name, 0.0)
            setattr(p, f.name, getattr(p, f.name) + c)
        return p


def look_around(t, s, max_deg=25.0):
    """Relaxed looking around, independent of the step rhythm: a slow wandering head yaw
    (periods 4-11 s) that fades out with speed. 20%: ~20 deg, 100%: ~4 deg, 130%: 0."""
    amount = max_deg * float(np.clip(1.15 - abs(s), 0.0, 0.8))
    w = (0.6 * math.sin(2 * math.pi * 0.09 * t + 0.4) + 0.3 * math.sin(2 * math.pi * 0.23 * t + 2.1)
         + 0.1 * math.sin(2 * math.pi * 0.17 * t + 4.0))
    return math.radians(amount * w)


def _build_one(args):
    scene, base, anchors, s, turn, side = args
    global _kin
    if _kin is None:
        _kin = G.DuckKinematics(scene)
    p = Style(base, anchors).params(s)
    p.turn_dps, p.side_m = float(turn), float(side)
    return (s, turn, side), _kin.generate(p, com_passes=3)


_kin = None


class GaitGrid:
    """Gaits on a speed x turn x side grid; reference blended between neighbours.
    Command: set(speed fraction of your walk, turn deg/s, side m/cycle)."""

    KEYS = ("joint_pos", "root_height", "root_local", "root_quat_local", "contact")

    def __init__(self, data):
        d = {k: data[k] for k in data.files} if hasattr(data, "files") else data
        self.d = d
        self.S, self.W, self.Y = d["speeds"], d["turns"], d["sides"]
        self.v100 = float(d["v100"])
        self.head_idx = {n: list(d["joint_names"]).index(n) for n in ("head_yaw", "head_pitch")}
        self.look_yaw = self.look_pitch = 0.0       # head layer (rad), set by the caller
        self.set(1.0, 0.0, 0.0)

    @staticmethod
    def _axis(ax, x):
        x = float(np.clip(x, ax[0], ax[-1]))
        i = int(np.clip(np.searchsorted(ax, x) - 1, 0, len(ax) - 2))
        return i, (x - ax[i]) / (ax[i + 1] - ax[i]), x

    def set(self, s, turn=0.0, side=0.0):
        (i, wi, s), (j, wj, turn), (k, wk, side) = (self._axis(self.S, s), self._axis(self.W, turn),
                                                    self._axis(self.Y, side))
        self.cmd = (s, turn, side)
        self.corners = []
        for a, wa in ((i, 1 - wi), (i + 1, wi)):
            for b, wb in ((j, 1 - wj), (j + 1, wj)):
                for c, wc in ((k, 1 - wk), (k + 1, wk)):
                    if wa * wb * wc > 1e-6:
                        self.corners.append(((a, b, c), wa * wb * wc))
        # stepping fades into standing still near a zero command
        self.walk = float(np.clip(max(abs(s) / 0.15, abs(turn) / 30.0, abs(side) / 0.006), 0.0, 1.0))
        bl = lambda key: sum(w * self.d[key][idx] for idx, w in self.corners)
        self.cycle = float(bl("cycle_s"))
        self.vel = self.walk * bl("vel")                     # (vx, vy) in the trunk frame
        self.yaw_rate = self.walk * float(bl("yaw_rate"))
        self.vx = float(self.vel[0])

    def _at(self, key, idx, phase):
        n = int(self.d["length"][idx]); f = (phase % 1.0) * n; i0 = int(f) % n; w = f - math.floor(f)
        x = self.d[key][idx]
        return (1 - w) * x[i0] + w * x[(i0 + 1) % n]

    def at(self, key, phase):
        x = sum(w * self._at(key, idx, phase) for idx, w in self.corners)
        stand = self.d["stand_" + key]
        return self.walk * x + (1 - self.walk) * stand

    def ref_q(self, phase):
        q = self.at("joint_pos", phase).copy()
        iy, ip = self.head_idx["head_yaw"], self.head_idx["head_pitch"]
        q[iy] = np.clip(q[iy] + self.look_yaw, -math.radians(55), math.radians(55))      # as in training
        q[ip] = np.clip(q[ip] + self.look_pitch, -math.radians(27), math.radians(27))
        return q

    def ref_h(self, phase):
        return float(self.at("root_height", phase))

    def ref_contact(self, phase):
        return self.at("contact", phase) > 0.5

    def tick(self, t):
        """Default head layer: relaxed looking around (fades out with speed)."""
        self.look_yaw = look_around(t, self.cmd[0])

    @staticmethod
    def build(scene, style, anchors=None, workers=None, log=print):
        jobs = [(scene, style.base, anchors or {}, float(s), float(w), float(y))
                for s in SPEEDS for w in TURNS for y in SIDES]
        out = {}
        t0 = time.time()
        workers = workers or min(mp.cpu_count(), 16)
        log(f"  {len(jobs)} gangen op {workers} processen ...")
        with mp.get_context("spawn").Pool(workers) as pool:
            for key, g in pool.imap_unordered(_build_one, jobs):
                out[key] = g
                if len(out) % 20 == 0:
                    log(f"  {len(out)}/{len(jobs)}  ({time.time() - t0:.0f} s)")
        log(f"  {len(out)} gangen in {time.time() - t0:.0f} s")
        S, W, Y = len(SPEEDS), len(TURNS), len(SIDES)
        nmax = max(len(g["t"]) for g in out.values()); J = len(G.JOINT_ORDER)
        d = dict(speeds=SPEEDS.astype(float), turns=TURNS, sides=SIDES,
                 length=np.zeros((S, W, Y), np.int32), cycle_s=np.zeros((S, W, Y)),
                 vel=np.zeros((S, W, Y, 2)), yaw_rate=np.zeros((S, W, Y)),
                 joint_pos=np.zeros((S, W, Y, nmax, J)), root_height=np.zeros((S, W, Y, nmax)),
                 root_local=np.zeros((S, W, Y, nmax, 3)), root_quat_local=np.zeros((S, W, Y, nmax, 4)),
                 contact=np.zeros((S, W, Y, nmax, 2)), joint_names=np.array(G.JOINT_ORDER),
                 style=json.dumps(asdict(style.base)), fps=G.FPS)
        for a, s in enumerate(SPEEDS):
            for b, w in enumerate(TURNS):
                for c, y in enumerate(SIDES):
                    g = out[(float(s), float(w), float(y))]; n = len(g["t"])
                    d["length"][a, b, c] = n; d["cycle_s"][a, b, c] = g["cycle_s"]
                    d["vel"][a, b, c] = g["vel_xy"]; d["yaw_rate"][a, b, c] = g["yaw_rate"]
                    for key in GaitGrid.KEYS:
                        d[key][a, b, c, :n] = g[key]
        i100 = int(np.argmin(np.abs(SPEEDS - 1.0)))
        d["v100"] = d["vel"][i100, list(TURNS).index(0.0), list(SIDES).index(0.0), 0]
        kin = G.DuckKinematics(scene)
        g0 = out[(0.0, 0.0, 0.0)]
        d["stand_joint_pos"] = kin.home.copy(); d["stand_root_height"] = np.array(HOME_H)
        d["stand_root_local"] = np.array([g0["root_local"][:, 0].mean(), 0.0, HOME_H])
        d["stand_root_quat_local"] = np.array([1.0, 0, 0, 0]); d["stand_contact"] = np.ones(2)
        return GaitGrid(d), out


def make_style(a):
    base = load_params(a.style)
    anchors = {float(kv.split("=")[0]) / 100: load_params(kv.split("=")[1]) for kv in a.anchor}
    return Style(base, anchors), anchors


def view(scene, grid):
    import mujoco.viewer
    spec = mujoco.MjSpec.from_file(scene)
    sliders = (("snelheid_pct", -60, 130, 100), ("draai_gr_s", -90, 90, 0), ("zijstap_mm", -10, 10, 0))
    for name, lo, hi, _ in sliders:
        body = spec.worldbody.add_body(name="_b_" + name, pos=[0, 0, -1])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0,
                      rgba=[0, 0, 0, 0], mass=1e-6)
        body.add_joint(name="_" + name, type=mujoco.mjtJoint.mjJNT_SLIDE, range=[lo, hi])
        act = spec.add_actuator(name=name, target="_" + name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrlrange = [lo, hi]; act.ctrllimited = 1
    m = spec.compile(); d = mujoco.MjData(m)
    ids = [m.actuator(n).id for n, *_ in sliders]
    for aid, (_, _, _, v0) in zip(ids, sliders):
        d.ctrl[aid] = v0
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in G.JOINT_ORDER])
    print("sliders onderaan het Control-paneel: snelheid_pct (100 = jouw loopje), draai_gr_s, zijstap_mm."
          "  G = camera.")

    def key(k):
        if k == ord("G"):
            FOLLOW[0] = not FOLLOW[0]

    cur = np.array([100.0, 0.0, 0.0]); rate = np.array([60.0, 120.0, 20.0]) / G.FPS
    phase, pos, yaw, t = 0.0, np.zeros(2), 0.0, 0.0
    fast_model(m)
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
        fast_render(v)
        while v.is_running():
            target = np.array([d.ctrl[i] for i in ids])
            cur += np.clip(target - cur, -rate, rate)
            grid.set(cur[0] / 100, cur[1], cur[2] / 1000)
            grid.tick(t)
            c, s = math.cos(yaw), math.sin(yaw)
            rl = grid.at("root_local", phase)
            d.qpos[:2] = pos + np.array([c * rl[0] - s * rl[1], s * rl[0] + c * rl[1]])
            d.qpos[2] = rl[2]
            ql = grid.at("root_quat_local", phase); ql = ql / np.linalg.norm(ql)
            d.qpos[3:7] = G._rotz_quat(yaw, ql)
            d.qpos[qadr] = grid.ref_q(phase)
            mujoco.mj_forward(m, d)
            if FOLLOW[0]:
                v.cam.lookat[:] = d.qpos[:3]
            v.sync()
            time.sleep(1 / G.FPS)
            dt = 1 / G.FPS; t += dt
            pos += np.array([c * grid.vel[0] - s * grid.vel[1], s * grid.vel[0] + c * grid.vel[1]]) * dt
            yaw += grid.yaw_rate * dt
            phase = (phase + dt / grid.cycle) % 1.0


def check(scene, gaits):
    import gait_lib as L
    kin = G.DuckKinematics(scene)
    J = G.JOINT_ORDER
    kc = [J.index("left_knee"), J.index("right_knee")]
    hy = [J.index("left_hip_yaw"), J.index("right_hip_yaw")]
    rows = []
    for (s, w, y), g in sorted(gaits.items()):
        zo, _ = L.check(kin, g)
        q = g["joint_pos"]; vv = np.abs(np.diff(np.vstack([q, q[:1]]), axis=0)) * G.FPS
        rows.append((s, w, y, zo, vv[:, kc].max(), np.degrees(np.abs(q[:, hy]).max()), g["vel_xy"][0]))
    r = np.array(rows)
    print("\n  snelheid  m/s    ZMP-buiten (slechtste over bocht/zijstap)  knie max   heup-yaw max")
    for s in SPEEDS:
        m = r[:, 0] == s
        print(f"  {s * 100:5.0f}%  {r[m & (r[:, 1] == 0) & (r[:, 2] == 0), 6][0]:.3f}"
              f"   {r[m, 3].max() * 100:4.0f}%{'':31s}{r[m, 4].max():5.1f}     {r[m, 5].max():5.1f}°")
    fast = r[r[:, 4] > KNEE_MAX]
    if len(fast):
        print(f"  let op: {len(fast)} gangen met knie > {KNEE_MAX} rad/s (servo houdt dan te weinig kracht over), "
              f"ergste {fast[:, 4].max():.1f} rad/s")
    bad = r[r[:, 3] >= 0.15]
    for s, w, y, zo, *_ in bad:
        print(f"  let op: {s * 100:.0f}% draai {w:.0f}°/s zij {y * 1000:.0f} mm: ZMP {zo * 100:.0f}% buiten")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mooi", action="store_true", help="schaduwen en reflecties aan (trager)")
    ap.add_argument("--scene", default="scene_walk.xml")
    ap.add_argument("--style", default=STYLE)
    ap.add_argument("--anchor", nargs="*", default=[], help="pct=bestand.npz, bv. 40=favorites/langzaam.npz")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print", type=float, default=None, metavar="PCT")
    ap.add_argument("--grid", default=None, help="schrijf het gangenrooster (training + afspelen)")
    ap.add_argument("--workers", type=int, default=None, help="aantal processen (standaard max 16)")
    ap.add_argument("--load", default=None, help="speel een bestaand gait_grid.npz af (geen nieuwe berekening)")
    a = ap.parse_args()
    if a.load:
        print(f"afspelen: {a.load} (precies de referentie die de training krijgt)")
        view(a.scene, GaitGrid(np.load(a.load)))
        return
    style, anchors = make_style(a)
    if a.print is not None:
        print(json.dumps(asdict(style.params(a.print / 100)), indent=1))
        return
    print(f"stijl: {a.style}" + (f"  ankers: {a.anchor}" if a.anchor else ""))
    print(f"rooster: {len(SPEEDS)} snelheden x {len(TURNS)} bochten x {len(SIDES)} zijstappen")
    grid, gaits = GaitGrid.build(a.scene, style, anchors, workers=a.workers)
    if a.grid:
        np.savez_compressed(a.grid, **grid.d); print("opgeslagen:", a.grid)
    if a.check:
        check(a.scene, gaits)
    if a.view:
        view(a.scene, grid)


if __name__ == "__main__":
    main()
