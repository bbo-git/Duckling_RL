"""
duckling_play_onnx.py - test a trained Duckling walking policy (ONNX) in plain MuJoCo.

    pip install mujoco onnxruntime numpy
    python duckling_play_onnx.py                      # nieuwste .onnx in deze map
    python duckling_play_onnx.py duckling_walk.onnx scene_walk.xml
    python duckling_play_onnx.py duckling_walk.onnx scene_walk.xml --headless 10 --vx 0.15

Builds the 61-D observation exactly as in training (Mjlab-Velocity-Flat-Duckling):
    [0:3]   angular velocity of trunk_base, body frame
    [3:6]   projected gravity, body frame
    [6:20]  joint positions relative to HOME (see HOME_POS_DEG)
    [20:34] joint velocities, one control step old
    [34:48] previous action
    [48:51] velocity command  vx, vy, wz
    [51:55] head command      neck_pitch, head_pitch, head_yaw, head_roll (deltas from HOME)
    [55:61] body command      zeros
    [61:63] gait clock        sin, cos of the walk-cycle phase (only for policies trained
                              with the Blender walk cycle; detected from the ONNX input size)
Action (14) = joint position targets relative to HOME, sent at 50 Hz.
NOTE: policies trained before the HOME stance was introduced expect HOME = 0;
run those with --cad-home.

Keys in the viewer (keyboard focus on the MuJoCo window):
    arrow up / down     forward / backward speed
    arrow left / right  turn left / right
    , / .               sideways left / right
    I / K               neck pitch      J / L   head yaw
    Z                   all commands to zero
    backspace           reset the robot
    - / =               playback slower / faster (also --speed 2)
"""

import argparse
import time

import mujoco

FOLLOW = [True]   # camera follows the robot; G in the viewer toggles
import numpy as np
import onnxruntime as ort

# --- must match training -----------------------------------------------------
TIMESTEP = 0.002
DECIMATION = 10                      # 50 Hz policy
SPAWN_Z = 0.059
# HOME stance (deg), must match duckling_constants.py. Observations are joint angles
# relative to HOME, actions are offsets from HOME.
HOME_POS_DEG = {"hip_pitch": -11.8, "knee": -8.6, "ankle": -20.3}
TORSO = "trunk_base"
LIMITS = dict(vx=0.12, vy=0.02, wz=0.6)       # = training command ranges
CYCLE_S = 1.0                        # walk-cycle length (s), walk_cycle.npz "cycle_s"
HEAD_LIMITS = (0.25, 0.25, 0.50, 0.15)  # neck_pitch, head_pitch, head_yaw, head_roll (= training HEAD_CAPS)


NJ = 14
REST_MODE = [True]       # hold HOME when told to stand still (see run_viewer)
REST_FADE_S = 0.5
# Sliders at the bottom of the viewer's Control panel: name, min, max
SLIDERS = (("vooruit_cm_s", -4.5, 12.0), ("draai_gr_s", -34.0, 34.0), ("zijwaarts_cm_s", -2.0, 2.0),
           ("kop_draai_gr", -28.0, 28.0), ("kop_knik_gr", -14.0, 14.0))


def model_with_sliders(scene):
    """The robot plus hidden slider joints whose actuators show up at the bottom of the
    viewer's Control panel. They exert no force (gear 0) and float (gravcomp)."""
    spec = mujoco.MjSpec.from_file(scene)
    for name, lo, hi in SLIDERS:
        body = spec.worldbody.add_body(name="_b_" + name, pos=[0, 0, -1], gravcomp=1.0)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0,
                      rgba=[0, 0, 0, 0], mass=1e-3)
        body.add_joint(name="_" + name, type=mujoco.mjtJoint.mjJNT_SLIDE, range=[lo, hi], damping=1.0)
        act = spec.add_actuator(name=name, target="_" + name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrlrange = [lo, hi]; act.ctrllimited = 1; act.gear[0] = 0.0
    return spec.compile()


class GridClock:
    """Cadence of your walk (gait_grid.npz) at a twist command, exactly as the training
    blends it: speed = vx / v100, turn = wz in deg/s, side 0."""

    def __init__(self, path):
        d = np.load(path)
        self.S, self.W, self.Y = d["speeds"], d["turns"], d["sides"]
        self.cyc = d["cycle_s"][:, :, int(np.argmin(np.abs(self.Y)))]
        self.v100 = float(d["v100"])

    @staticmethod
    def _axis(ax, x):
        x = float(np.clip(x, ax[0], ax[-1]))
        i = int(np.clip(np.searchsorted(ax, x) - 1, 0, len(ax) - 2))
        return i, (x - ax[i]) / (ax[i + 1] - ax[i])

    def walk(self, twist):
        """0 = standing, 1 = walking (as in training): the gait clock fades out with it."""
        s = twist[0] / self.v100
        return float(np.clip(max(abs(s) / 0.15, abs(np.degrees(twist[2])) / 30.0,
                                 abs(twist[1]) * 0.45 / 0.006), 0.0, 1.0))

    def cycle(self, twist):
        (i, wi), (j, wj) = self._axis(self.S, twist[0] / self.v100), self._axis(self.W, np.degrees(twist[2]))
        c = self.cyc
        return ((1 - wi) * ((1 - wj) * c[i, j] + wj * c[i, j + 1])
                + wi * ((1 - wj) * c[i + 1, j] + wj * c[i + 1, j + 1]))


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


class GLFW:  # GLFW key codes, as passed to the viewer's key_callback
    KEY_Z, KEY_COMMA, KEY_PERIOD, KEY_MINUS, KEY_EQUAL = 90, 44, 46, 45, 61
    KEY_I, KEY_J, KEY_K, KEY_L = 73, 74, 75, 76
    KEY_BACKSPACE, KEY_RIGHT, KEY_LEFT, KEY_DOWN, KEY_UP = 259, 262, 263, 264, 265


class Policy:
    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        n_in = self.sess.get_inputs()[0].shape[-1]
        n_out = self.sess.get_outputs()[0].shape[-1]
        assert n_in in (61, 63) and n_out == 14, f"unexpected policy shape {n_in}->{n_out}"
        self.n_in = n_in

    def __call__(self, obs):
        return self.sess.run(None, {self.inp: obs[None].astype(np.float32)})[0][0]


class Duckling:
    gait_clock = False
    def __init__(self, scene, sliders=False, grid=None):
        self.m = model_with_sliders(scene) if sliders else mujoco.MjModel.from_xml_path(scene)
        self.grid = grid
        self.phase = 0.0
        self.m.opt.timestep = TIMESTEP
        self.m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.m.opt.iterations = 10
        self.m.opt.ls_iterations = 20
        self.d = mujoco.MjData(self.m)
        # actuated joints in actuator order (= training order)
        self.nj = NJ                                   # robot actuators come first (sliders after)
        self.jids = [self.m.actuator_trnid[i, 0] for i in range(self.nj)]
        self.qadr = np.array([self.m.jnt_qposadr[j] for j in self.jids])
        self.vadr = np.array([self.m.jnt_dofadr[j] for j in self.jids])
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.jids]
        head_order = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
        assert all(h in self.names for h in head_order), "head joints missing"
        self.torso = self.m.body(TORSO).id
        self.home = np.zeros(self.nj)
        for i, n in enumerate(self.names):
            for key, deg in HOME_POS_DEG.items():
                if n.endswith("_" + key):
                    self.home[i] = np.radians(deg)
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
        self.steps = 0                                # control steps since reset (gait clock)
        self.phase = 0.0

    def obs(self, twist, head):
        q = self.d.qpos[3:7]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, q)
        R = rot.reshape(3, 3)
        ang_vel_b = self.d.qvel[3:6].copy()          # free joint: already body frame
        grav_b = R.T @ np.array([0.0, 0.0, -1.0])
        jpos = self.d.qpos[self.qadr] - self.home
        jvel = self.prev_jvel                         # one control step old
        self.prev_jvel = self.d.qvel[self.vadr].copy()
        if self.grid is not None and not (np.hypot(twist[0], twist[1]) > 0.02 or abs(twist[2]) > 0.1):
            head = np.zeros(4)              # as in training: no look commands while standing still
        parts = [ang_vel_b, grav_b, jpos, jvel, self.last_action, twist, head, np.zeros(6)]
        if self.gait_clock:
            ph = 2 * np.pi * self.phase
            w = self.grid.walk(twist) if self.grid is not None else 1.0   # clock fades out when standing
            parts.append([w * np.sin(ph), w * np.cos(ph)])
        return np.concatenate(parts)

    def step(self, action, twist=None):
        # gait clock as in training: fixed Blender cycle, or the cadence of your walk
        # (gait grid) at the current command
        cyc = self.grid.cycle(twist) if (self.grid is not None and twist is not None) else CYCLE_S
        self.phase = (self.phase + TIMESTEP * DECIMATION / cyc) % 1.0
        self.last_action = action.copy()
        self.d.ctrl[:self.nj] = self.home + action           # targets relative to HOME
        for _ in range(DECIMATION):
            mujoco.mj_step(self.m, self.d)
        self.steps += 1

    @property
    def height(self):
        return self.d.qpos[2]

    @property
    def fallen(self):
        return self.d.xmat[self.torso][8] < np.cos(np.radians(70))


def _foot_contacts(robot):
    """(left, right) sole touching the floor, from MuJoCo's contact list."""
    m, d = robot.m, robot.d
    feet = [m.geom(f"{s}_foot_collision").id for s in ("left", "right")]
    out = [False, False]
    for c in d.contact[:d.ncon]:
        for i, g in enumerate(feet):
            if g in (c.geom1, c.geom2):
                out[i] = True
    return out


def _reference_gait(path="walk_cycle.npz"):
    """Cadence, step length, trunk height and yaw/roll ranges of the Blender cycle."""
    import os
    if not os.path.exists(path):
        return None
    c = np.load(path)
    n = [str(x) for x in c["joint_names"]]
    q = c["joint_pos"]
    rng = lambda j: np.degrees(q[:, n.index(j)].max() - q[:, n.index(j)].min())
    return dict(cadence=2.0 / float(c["cycle_s"]),
                step=float(c["vel_xy"][0]) * float(c["cycle_s"]) / 2.0,
                height=float(np.mean(c["root_height"])) if "root_height" in c else None,
                yaw=max(rng("left_hip_yaw"), rng("right_hip_yaw")),
                roll=max(rng("left_hip_roll"), rng("right_hip_roll")))


def run_headless(robot, pol, seconds, twist, head):
    steps = int(seconds / (TIMESTEP * DECIMATION))
    dt = TIMESTEP * DECIMATION
    x0 = robot.d.qpos[:2].copy()
    jids = robot.jids
    lo = np.array([robot.m.jnt_range[j][0] for j in jids])
    hi = np.array([robot.m.jnt_range[j][1] for j in jids])
    q_min = np.full(len(jids), np.inf)
    q_max = np.full(len(jids), -np.inf)
    fell = None
    touchdowns, heights, prev = [], [], _foot_contacts(robot)
    feet_xy = {0: [], 1: []}
    for k in range(steps):
        robot.step(pol(robot.obs(twist, head)), twist)
        q = robot.d.qpos[robot.qadr]
        q_min, q_max = np.minimum(q_min, q), np.maximum(q_max, q)
        cur = _foot_contacts(robot)
        for i in (0, 1):
            if cur[i] and not prev[i] and k * dt > 1.0:         # skip the first second
                touchdowns.append(k * dt)
                side = "left" if i == 0 else "right"
                feet_xy[i].append(robot.d.site(f"{side}_foot").xpos[:2].copy())
        prev = cur
        if k * dt > 1.0:
            heights.append(robot.height)
        if robot.fallen:
            fell = k * dt
            break
    dx = robot.d.qpos[:2] - x0
    if fell is not None:
        print(f"gevallen na {fell:.2f} s")
    else:
        print(f"{seconds:.1f} s blijven staan; hoogte romp {robot.height * 1000:.1f} mm, "
              f"verplaatsing x {dx[0] * 1000:.0f} mm, y {dx[1] * 1000:.0f} mm "
              f"({np.linalg.norm(dx) / seconds:.3f} m/s)")

    # gait metrics next to the Blender cycle
    t_meas = max(seconds - 1.0, 1e-6) if fell is None else max(fell - 1.0, 1e-6)
    cad = len(touchdowns) / t_meas
    strides = [np.linalg.norm(np.diff(np.array(v), axis=0), axis=1) for v in feet_xy.values() if len(v) > 1]
    step = np.mean(np.concatenate(strides)) / 2 if strides else float("nan")
    yaw = max(np.degrees(q_max[i] - q_min[i]) for i, n in enumerate(robot.names) if n.endswith("hip_yaw"))
    roll = max(np.degrees(q_max[i] - q_min[i]) for i, n in enumerate(robot.names) if n.endswith("hip_roll"))
    ref = _reference_gait()
    rows = [("stappen per seconde", cad, ref and ref["cadence"], "{:.2f}"),
            ("staplengte (mm)", step * 1000, ref and ref["step"] * 1000, "{:.0f}"),
            ("romphoogte gem. (mm)", np.mean(heights) * 1000 if heights else float("nan"),
             ref and ref["height"] and ref["height"] * 1000, "{:.1f}"),
            ("heup-yaw bereik (graden)", yaw, ref and ref["yaw"], "{:.0f}"),
            ("heup-roll bereik (graden)", roll, ref and ref["roll"], "{:.0f}")]
    print("looppatroon (na de eerste seconde)" + ("   policy   |  Blender-cyclus" if ref else ""))
    for label, v, r, fmt in rows:
        print(f"  {label:26s} {fmt.format(v):>8s}" + (f"   |  {fmt.format(r):>8s}" if r is not None else ""))
    print("  (vergelijk bij --vx ongeveer de snelheid van de cyclus, ~0.09)")

    print("bereik per gewricht (graden; * = binnen 10% van de limiet):")
    for n, a_, b_, l_, h_ in zip(robot.names, q_min, q_max, lo, hi):
        span = h_ - l_
        near = (a_ - l_) < 0.1 * span or (h_ - b_) < 0.1 * span
        print(f"  {n:16s} {np.degrees(a_):7.1f} .. {np.degrees(b_):6.1f}   "
              f"(limiet {np.degrees(l_):.0f} .. {np.degrees(h_):.0f}){'  *' if near else ''}")


def run_viewer(robot, pol, speed=1.0, vx=0.0, wz=0.0):
    import mujoco.viewer
    cmd = dict(vx=vx, vy=0.0, wz=wz)
    play = dict(speed=speed)
    head = np.zeros(4)
    flags = dict(reset=False)

    def key(k):
        g = GLFW
        step = {"vx": 0.02, "vy": 0.01, "wz": 0.1}
        if k == g.KEY_UP: cmd["vx"] += step["vx"]
        elif k == g.KEY_DOWN: cmd["vx"] -= step["vx"]
        elif k == g.KEY_LEFT: cmd["wz"] += step["wz"]
        elif k == g.KEY_RIGHT: cmd["wz"] -= step["wz"]
        elif k == g.KEY_COMMA: cmd["vy"] += step["vy"]
        elif k == g.KEY_PERIOD: cmd["vy"] -= step["vy"]
        elif k == g.KEY_I: head[0] -= 0.1
        elif k == g.KEY_K: head[0] += 0.1
        elif k == g.KEY_J: head[2] += 0.1
        elif k == g.KEY_L: head[2] -= 0.1
        elif k == g.KEY_Z:
            cmd.update(vx=0.0, vy=0.0, wz=0.0)
            head[:] = 0
        elif k == g.KEY_BACKSPACE: flags["reset"] = True
        elif k == ord("G"):
            FOLLOW[0] = not FOLLOW[0]; print("camera volgt:", FOLLOW[0])
        elif k == g.KEY_EQUAL: play["speed"] = min(play["speed"] * 2, 16.0)
        elif k == g.KEY_MINUS: play["speed"] = max(play["speed"] / 2, 0.125)
        for n, lim in LIMITS.items():
            cmd[n] = float(np.clip(cmd[n], -lim, lim))
        np.clip(head, -np.array(HEAD_LIMITS), np.array(HEAD_LIMITS), out=head)
        print(f"vx {cmd['vx']:+.2f}  vy {cmd['vy']:+.2f}  wz {cmd['wz']:+.2f}  "
              f"kop {np.round(head, 2)}  snelheid {play['speed']:g}x")

    dt = TIMESTEP * DECIMATION
    names = [n for n, *_ in SLIDERS]
    sl = [robot.m.actuator(n).id for n in names] if robot.m.nu > robot.nj else []

    def to_sliders():
        vals = [cmd["vx"] * 100, np.degrees(cmd["wz"]), cmd["vy"] * 100, np.degrees(head[2]), np.degrees(head[0])]
        for aid, val, (_, lo, hi) in zip(sl, vals, SLIDERS):
            robot.d.ctrl[aid] = float(np.clip(val, lo, hi))
        return [float(robot.d.ctrl[a]) for a in sl]

    def from_sliders(vals):
        cmd["vx"], cmd["wz"], cmd["vy"] = vals[0] / 100, np.radians(vals[1]), vals[2] / 100
        head[2], head[0] = np.radians(vals[3]), np.radians(vals[4])

    rest = dict(b=1.0)
    shown = to_sliders()
    if sl:
        print("schuifjes onderaan het Control-paneel: " + ", ".join(names))
    fast_model(robot.m)
    with mujoco.viewer.launch_passive(robot.m, robot.d, key_callback=key) as v:
        fast_render(v)
        t_next = time.time()
        while v.is_running():
            if sl:
                now = [float(robot.d.ctrl[a]) for a in sl]
                if now != shown:                       # a slider was moved in the viewer
                    from_sliders(now)
                shown = to_sliders()                   # keys changed the command -> sliders follow
            # Bij snelheid > 1 meerdere policy-stappen per getekend beeld.
            n = max(1, int(round(play["speed"])))
            for _ in range(n):
                if flags["reset"] or robot.fallen:
                    robot.reset()
                    flags["reset"] = False
                    shown = to_sliders()
                twist = np.array([cmd["vx"], cmd["vy"], cmd["wz"]])
                a = pol(robot.obs(twist, head))
                # rest mode: with a zero command the policy fades out and the robot just
                # holds HOME (statically stable: centre of mass 17 mm inside the feet).
                # It fades back in as soon as a command comes. --geen-rust turns it off.
                standing = not (np.hypot(twist[0], twist[1]) > 0.02 or abs(twist[2]) > 0.1)
                rest["b"] = float(np.clip(rest["b"] + (-dt if standing else dt) / REST_FADE_S, 0.0, 1.0)) \
                    if REST_MODE[0] else 1.0
                robot.step(rest["b"] * a, twist)
            if FOLLOW[0]:
                v.cam.lookat[:] = robot.d.qpos[:3]
            v.sync()
            t_next += n * dt / play["speed"]
            time.sleep(max(0.0, t_next - time.time()))
            if time.time() - t_next > 0.5:   # loopt achter: niet inhalen
                t_next = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mooi", action="store_true", help="schaduwen en reflecties aan (trager)")
    ap.add_argument("onnx", nargs="?", default=None,
                    help="policy; leeg = nieuwste .onnx in deze map")
    ap.add_argument("scene", nargs="?", default="scene_walk.xml")
    ap.add_argument("--headless", type=float, default=0.0, help="seconden zonder viewer")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--speed", type=float, default=1.0, help="afspeelsnelheid (2 = dubbel)")
    ap.add_argument("--geen-rust", action="store_true", help="policy ook bij stilstand (geen rust-modus)")
    ap.add_argument("--grid", default=None,
                    help="gait_grid.npz: voor policies getraind met jouw loopje (klok in jouw tempo)")
    ap.add_argument("--cad-home", action="store_true",
                    help="oude policies (getraind met HOME = CAD-nulstand)")
    a = ap.parse_args()
    if a.onnx is None:
        import glob, os
        found = sorted(glob.glob("*.onnx"), key=os.path.getmtime)
        if not found:
            ap.error("geen .onnx in deze map")
        a.onnx = found[-1]
        print("policy:", a.onnx)
    if a.cad_home:
        HOME_POS_DEG.clear()
        globals()["SPAWN_Z"] = 0.0625
    pol = Policy(a.onnx)
    REST_MODE[0] = not a.geen_rust
    grid = GridClock(a.grid) if a.grid else None
    robot = Duckling(a.scene, sliders=not a.headless, grid=grid)
    robot.gait_clock = pol.n_in == 63
    if robot.gait_clock:
        print("loopklok:", f"jouw loopje ({a.grid})" if grid else
              f"Blender-cyclus ({CYCLE_S:g} s) - getraind met jouw loopje? gebruik --grid gait_grid.npz")
    print("gewrichten:", ", ".join(robot.names))
    if a.headless:
        run_headless(robot, pol, a.headless, np.array([a.vx, 0.0, a.wz]), np.zeros(4))
    else:
        print(f"commando: vx {a.vx:+.2f}  wz {a.wz:+.2f}  (pijltjes in het MuJoCo-venster passen het aan)")
        run_viewer(robot, pol, a.speed, a.vx, a.wz)


if __name__ == "__main__":
    main()
