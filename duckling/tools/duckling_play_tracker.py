"""
duckling_play_tracker.py - play a Duckling policy (ONNX) in plain MuJoCo.

Steerable walk (policy from ./duck.sh train, task Mjlab-Steer-Flat-Duckling, 96 inputs):
    python duckling_play_tracker.py duckling_steer_latest.onnx --grid gait_grid.npz
    keys: arrow up/down = speed -/+ 10%, arrow left/right = turn +/- 30 deg/s,
          [ / ] = sidestep left/right, space = stop, G = camera, backspace = reset
    python duckling_play_tracker.py steer.onnx --grid gait_grid.npz --headless 10 --cmd 100 0 0
    python duckling_play_tracker.py steer.onnx --grid gait_grid.npz --headless 20 --script
        (scripted test: walk, turn, stop, start, orbit; prints how well it follows)

Old universal tracker (94 inputs) with generator presets:
    python duckling_play_tracker.py tracker.onnx --preset waggelen
    keys: 1..5 presets, 0 = stand still

Needs gait_gen.py, walk_speed.py, scene_walk.xml (+ robot_walk.xml, meshes/) here.
"""
import argparse
import json
import math
import time
from dataclasses import asdict

import mujoco
import numpy as np
import onnxruntime as ort

import gait_gen as G

FOLLOW = [True]   # camera follows the robot; G in the viewer toggles
TIMESTEP, DECIMATION = 0.002, 10
DT = TIMESTEP * DECIMATION
LOOKAHEAD = (1, 3, 6)
HOME_TRUNK_HEIGHT = 0.0556
SPAWN_Z = 0.059
V100 = 0.091          # m/s of your walk at 100% (overwritten from the grid)


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


class Gait:
    """A generated cycle sampled by phase (same interpolation as in training)."""

    def __init__(self, g):
        self.q = g["joint_pos"]; self.h = g["root_height"]; self.n = len(self.q)
        self.cycle = float(g["cycle_s"]); self.vx = float(g["vel_xy"][0])
        self.vel = np.array([self.vx, 0.0]); self.yaw_rate = 0.0

    def at(self, x, phase):
        f = (phase % 1.0) * self.n
        i0 = int(f) % self.n
        w = f - math.floor(f)
        return (1 - w) * x[i0] + w * x[(i0 + 1) % self.n]

    def ref_q(self, phase):
        return self.at(self.q, phase)

    def ref_h(self, phase):
        return self.at(self.h, phase)


def stand(kin):
    n = 25
    return Gait(dict(joint_pos=np.tile(kin.home, (n, 1)), root_height=np.full(n, HOME_TRUNK_HEIGHT),
                     cycle_s=n * 0.02, vel_xy=np.zeros(2)))


SLIDERS = (("snelheid_pct", 0.0, 130.0), ("draai_gr_s", -90.0, 90.0), ("zijstap_mm", -10.0, 10.0))


def model_with_sliders(scene):
    """The robot plus three hidden slider joints whose actuators show up at the bottom of
    the viewer's Control panel. They exert no force (gear 0) and float (gravcomp)."""
    spec = mujoco.MjSpec.from_file(scene)
    for name, lo, hi in SLIDERS:
        body = spec.worldbody.add_body(name="_b_" + name, pos=[0, 0, -1], gravcomp=1.0)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0,
                      rgba=[0, 0, 0, 0], mass=1e-3)
        body.add_joint(name="_" + name, type=mujoco.mjtJoint.mjJNT_SLIDE, range=[lo, hi], damping=1.0)
        act = spec.add_actuator(name=name, target="_" + name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrlrange = [lo, hi]; act.ctrllimited = 1; act.gear[0] = 0.0
    return spec.compile()


class Duck:
    def __init__(self, scene, policy, sliders=False):
        self.m = model_with_sliders(scene) if sliders else mujoco.MjModel.from_xml_path(scene)
        self.m.opt.timestep = TIMESTEP
        self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.m.opt.iterations, self.m.opt.ls_iterations = 10, 20
        self.d = mujoco.MjData(self.m)
        self.nj = len(G.JOINT_ORDER)                      # robot actuators come first
        self.jids = [self.m.actuator_trnid[i, 0] for i in range(self.nj)]
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.jids]
        assert self.names == G.JOINT_ORDER, "joint order differs from the generator"
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jids])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in self.jids])
        self.home = np.zeros(self.nj)
        for i, n in enumerate(self.names):
            for k, v in G.HOME_DEG.items():
                if n.endswith("_" + k):
                    self.home[i] = math.radians(v)
        self.sess = ort.InferenceSession(policy, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.n_in = self.sess.get_inputs()[0].shape[-1]
        assert self.n_in in (94, 96), f"onbekende policy (invoer {self.n_in}; tracker = 94, steer = 96)"
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[2] = SPAWN_Z
        self.d.qpos[3:7] = [1, 0, 0, 0]
        self.d.qpos[self.qadr] = self.home
        self.d.ctrl[:self.nj] = self.home
        mujoco.mj_forward(self.m, self.d)
        self.last_action = np.zeros(self.nj)
        self.prev_jvel = np.zeros(self.nj)
        self.phase = 0.0

    def obs(self, gait):
        R = np.zeros(9); mujoco.mju_quat2Mat(R, self.d.qpos[3:7]); R = R.reshape(3, 3)
        parts = [self.d.qvel[3:6].copy(), R.T @ np.array([0, 0, -1.0]),
                 self.d.qpos[self.qadr] - self.home, self.prev_jvel, self.last_action]
        self.prev_jvel = self.d.qvel[self.vadr].copy()
        for a in LOOKAHEAD:
            parts.append(gait.ref_q(self.phase + a * DT / gait.cycle) - self.home)
        ph = 2 * math.pi * self.phase
        if self.n_in == 96:
            parts.append([gait.vel[0], gait.vel[1], gait.yaw_rate])
        else:
            parts.append([gait.vx])
        parts += [[(gait.ref_h(self.phase) - HOME_TRUNK_HEIGHT) * 10.0], [math.sin(ph), math.cos(ph)]]
        return np.concatenate(parts).astype(np.float32)

    def step(self, gait):
        a = self.sess.run(None, {self.inp: self.obs(gait)[None]})[0][0]
        self.last_action = a.copy()
        self.d.ctrl[:self.nj] = self.home + a
        for _ in range(DECIMATION):
            mujoco.mj_step(self.m, self.d)
        self.phase = (self.phase + DT / gait.cycle) % 1.0

    @property
    def fallen(self):
        return self.d.xmat[1][8] < math.cos(math.radians(70))

    @property
    def yaw(self):
        return 2 * math.atan2(self.d.qpos[6], self.d.qpos[3])

    def body_vel(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        v = self.d.qvel[:2]
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]]), float(self.d.qvel[5])


class Steer:
    """Command (speed %, turn deg/s, side mm/cycle) with the same ramps as in training."""
    RATE = np.array([60.0, 120.0, 20.0])          # per second

    def __init__(self, grid, cmd=(100.0, 0.0, 0.0)):
        self.grid = grid
        self.cur = np.array(cmd, float); self.target = self.cur.copy(); self.t = 0.0
        self.apply()

    def apply(self):
        self.cur += np.clip(self.target - self.cur, -self.RATE * DT, self.RATE * DT)
        self.grid.set(self.cur[0] / 100, self.cur[1], self.cur[2] / 1000)
        self.grid.tick(self.t); self.t += DT
        return self.grid


SCRIPT = [(0, (100, 0, 0), "rechtdoor 100%"), (4, (100, 45, 0), "bocht links 45/s"),
          (8, (0, 0, 0), "stoppen"), (10, (50, 0, 0), "starten 50%"), (13, (30, -60, 0), "bocht rechts"),
          (16, (0, 60, 0), "draaien op de plek"), (19, (0, 30, -10), "orbit"), (23, (130, 0, 0), "130%")]


def run_script(duck, grid, total):
    """Scripted test. Besides speed/turn it measures the quality of the walk, so the
    numbers show what the viewer shows: stance width (feet apart, sideways), foot lift
    during swing, joint error vs the reference and how jittery the actions are."""
    st = Steer(grid, SCRIPT[0][1]); k_seg = 0
    seg_start = (0.0, duck.d.qpos[:2].copy(), duck.yaw); rows = []
    sites = [duck.m.site("left_foot").id, duck.m.site("right_foot").id]
    q = dict(width=[], lift=[], err=[], jit=[])
    seg_q = []
    last_a = np.zeros(duck.nj)

    def close_segment():
        seg_q.append({k: (np.mean(v) if v else float("nan")) for k, v in q.items()})
        for v in q.values():
            v.clear()

    for k in range(int(total / DT)):
        t = k * DT
        if k_seg + 1 < len(SCRIPT) and t >= SCRIPT[k_seg + 1][0]:
            rows.append(summary(duck, seg_start, t, SCRIPT[k_seg])); close_segment()
            k_seg += 1; st.target = np.array(SCRIPT[k_seg][1], float)
            seg_start = (t, duck.d.qpos[:2].copy(), duck.yaw)
        g = st.apply()
        ref = g.ref_q(duck.phase)
        duck.step(g)
        if duck.fallen:
            print(f"gevallen na {t:.2f} s ({SCRIPT[k_seg][2]})"); break
        c, s_ = math.cos(duck.yaw), math.sin(duck.yaw)
        p = [duck.d.site_xpos[i] for i in sites]
        dy = -s_ * (p[0][0] - p[1][0]) + c * (p[0][1] - p[1][1])        # sideways, trunk frame
        q["width"].append(dy * 1000)
        q["lift"].append(max(p[0][2], p[1][2]) * 1000)
        q["err"].append(np.degrees(np.sqrt(np.mean((duck.d.qpos[duck.qadr] - ref) ** 2))))
        q["jit"].append(np.degrees(np.abs(duck.last_action - last_a).mean()))
        last_a = duck.last_action.copy()
    else:
        rows.append(summary(duck, seg_start, total, SCRIPT[k_seg]))
    close_segment()
    print(f"\n  {'fase':22s} gevraagd (mm/s, gr/s)   gemeten        "
          f"breedte  voet-op  gewricht-fout  trillen")
    for (name, req, got), sq in zip(rows, seg_q):
        print(f"  {name:22s} {req[0]:6.0f} {req[1]:6.0f}       {got[0]:6.0f} {got[1]:6.0f}"
              f"      {sq['width']:4.0f}mm  {sq['lift']:4.1f}mm      {sq['err']:4.1f}°        {sq['jit']:4.1f}°")
    print("  breedte = voeten zijwaarts uit elkaar (referentie ~60-65 mm); voet-op = hoogste voet"
          " (gem.); gewricht-fout = RMS t.o.v. referentie; trillen = actieverandering per stap")


def summary(duck, start, t, seg):
    t0, p0, y0 = start
    dt = max(t - t0, 1e-6)
    dp = duck.d.qpos[:2] - p0
    c, s = math.cos(y0), math.sin(y0)
    fwd = (c * dp[0] + s * dp[1]) / dt
    dyaw = math.degrees(math.atan2(math.sin(duck.yaw - y0), math.cos(duck.yaw - y0))) / dt
    pct, turn, side = seg[1]
    return seg[-1], (pct / 100 * V100 * 1000, turn), (fwd * 1000, dyaw)


def make_gaits(a):
    gaits = {str(i + 1): (n, G.PRESETS[n]) for i, n in
             enumerate(["normaal", "drentelen", "parmantig", "sluipen", "waggelen"])}
    p = G.GaitParams(**asdict(G.PRESETS[a.preset]))
    for kv in a.set:
        k, v = kv.split("="); setattr(p, k, float(v))
    return gaits, p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mooi", action="store_true", help="schaduwen en reflecties aan (trager)")
    ap.add_argument("onnx")
    ap.add_argument("--scene", default="scene_walk.xml")
    ap.add_argument("--grid", default=None, help="gait_grid.npz (walk_speed.py --grid) voor de stuurbare loop")
    ap.add_argument("--cmd", nargs=3, type=float, default=[100.0, 0.0, 0.0],
                    metavar=("PCT", "DRAAI", "ZIJ_MM"), help="beginopdracht met --grid")
    ap.add_argument("--script", action="store_true", help="met --headless: vast testprogramma")
    ap.add_argument("--preset", default="normaal", choices=list(G.PRESETS))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--headless", type=float, default=0.0)
    a = ap.parse_args()
    kin = G.DuckKinematics(a.scene)
    duck = Duck(a.scene, a.onnx, sliders=bool(a.grid) and not a.headless)
    steer = None
    if a.grid:
        from walk_speed import GaitGrid
        assert duck.n_in == 96, "deze policy kent geen sturing (94 invoer): gebruik --preset"
        global V100
        steer = Steer(GaitGrid(np.load(a.grid)), a.cmd)
        V100 = steer.grid.v100
        gait = steer.grid
    else:
        presets, p = make_gaits(a)
        print("gang genereren ...")
        gait = Gait(kin.generate(p))
    if a.headless:
        if steer and a.script:
            run_script(duck, steer.grid, a.headless); return
        x0, y0 = duck.d.qpos[:2].copy(), duck.yaw
        for k in range(int(a.headless / DT)):
            g = steer.apply() if steer else gait
            duck.step(g)
            if duck.fallen:
                print(f"gevallen na {k * DT:.2f} s"); return
        _, req, got = summary(duck, (0.0, x0, y0), a.headless,
                              ("", tuple(a.cmd) if steer else (gait.vx / V100 * 100, 0, 0), ""))
        print(f"{a.headless:.0f} s: gevraagd {req[0]:.0f} mm/s, {req[1]:.0f} gr/s   "
              f"gemeten {got[0]:.0f} mm/s, {got[1]:.0f} gr/s")
        return
    import mujoco.viewer
    cache = {}
    state = dict(gait=gait, speed=1.0, reset=False)

    def key(k):
        c = chr(k) if 32 <= k < 127 else ""
        if steer:
            t = steer.target
            if k == 265: t[0] = min(t[0] + 10, 130)
            elif k == 264: t[0] = max(t[0] - 10, 0)
            elif k == 263: t[1] = min(t[1] + 30, 90)
            elif k == 262: t[1] = max(t[1] - 30, -90)
            elif c == "[": t[2] = min(t[2] + 5, 10)
            elif c == "]": t[2] = max(t[2] - 5, -10)
            elif c == " ": t[:] = 0
            if k in (262, 263, 264, 265) or c in "[] ":
                print(f"opdracht: snelheid {t[0]:.0f}%  draai {t[1]:.0f} gr/s  zij {t[2]:.0f} mm")
        elif c in presets:
            name, pp = presets[c]
            if name not in cache:
                print(f"gang '{name}' genereren ..."); cache[name] = Gait(kin.generate(pp))
            state["gait"] = cache[name]; print("gang:", name)
        elif c == "0":
            state["gait"] = stand(kin); print("stilstaan")
        if c == "G":
            FOLLOW[0] = not FOLLOW[0]; print("camera volgt:", FOLLOW[0])
        elif k == 259:
            state["reset"] = True
        elif c == "=":
            state["speed"] = min(state["speed"] * 2, 8.0)
        elif c == "-":
            state["speed"] = max(state["speed"] / 2, 0.125)

    sl = [duck.m.actuator(n).id for n, *_ in SLIDERS] if steer else []

    def to_sliders():
        for i, aid in enumerate(sl):
            duck.d.ctrl[aid] = steer.target[i]
        return [float(duck.d.ctrl[aid]) for aid in sl]

    shown = to_sliders()
    if steer:
        print("sliders onderaan het Control-paneel: snelheid_pct, draai_gr_s, zijstap_mm (of pijltjes/[ ]/spatie)")
    fast_model(duck.m)
    with mujoco.viewer.launch_passive(duck.m, duck.d, key_callback=key) as v:
        fast_render(v)
        t_next = time.time()
        while v.is_running():
            if sl:
                now = [float(duck.d.ctrl[aid]) for aid in sl]
                if now != shown:                                   # slider moved in the viewer
                    steer.target[:] = now
                elif list(steer.target) != shown:                  # changed with the keys
                    now = to_sliders()
                shown = now
            if state["reset"] or duck.fallen:
                duck.reset(); state["reset"] = False
                shown = to_sliders()
            duck.step(steer.apply() if steer else state["gait"])
            if FOLLOW[0]:
                v.cam.lookat[:] = duck.d.qpos[:3]
            v.sync()
            t_next += DT / state["speed"]
            time.sleep(max(0.0, t_next - time.time()))
            if time.time() - t_next > 0.5:
                t_next = time.time()


if __name__ == "__main__":
    main()
