"""
duckling_follow.py - test a target-following Duckling policy (2026-10-15) in plain MuJoCo.

The policy sees the target like a camera would (as in training): [bearing, distance - 0.5,
visible] with noise, 60 deg field of view, 80 ms delay and dropouts. The twist command
(P-controller on the true target) only drives the gait clock and the rest mode, as in training.

    python duckling_follow.py policy.onnx --grid grids/gait_grid_v6e.npz            # viewer
    python duckling_follow.py policy.onnx --grid grids/gait_grid_v6e.npz --meet     # scenarios

Viewer: the person is steered with the sliders "persoon_cm_s" and "persoon_draai_gr_s" at the
bottom of the Control panel (smooth). Keys (focus on the 3D view): arrows move the same sliders, Z = target stops, J / L = target jumps 90 deg left/right
of the duck, K = target walks towards / past the duck, X = person waits for the duck on/off,
backspace = reset (keeps the person's speed), G = camera follows the duck (default off).
Keys only work while the MuJoCo window has focus (click in the 3D view, not on a panel). The target is a
"person" (two red legs); orange = waiting for the duck (it fell > 80 cm behind).
"""
import argparse, math, time
import numpy as np
import mujoco
import duckling_play_onnx as P

FOLLOW_DIST, FOV_HALF = 0.5, math.radians(30)
BEARING_NOISE, DIST_NOISE, DROPOUT, DELAY = math.radians(2), 0.15, 0.05, 4
K_DIST, K_BEARING = 1.0, 1.5
WAIT_FAR, WAIT_NEAR, VMAX = 0.8, 0.6, 0.14
BACK_DIST, BACK_MAX_V, BACK_MIN_V, BACK_RELEASE, BACK_FACE = 0.36, 0.06, 0.035, 0.45, 0.8
REST_IN, REST_OUT, REST_IN_B, REST_OUT_B = 0.08, 0.14, 0.25, 0.45
ACC_V, ACC_W = 0.15, 1.5
WALK_MAX_V, RUN_DIST, RUN_RELEASE, RUN_V = 0.075, 0.62, 0.54, 0.13
ACCEL, TURN_ACCEL = 0.15, 2.0   # person: m/s^2 and rad/s^2 towards the slider setpoint (smooth)
STALL_S, STALL_MOVE, STALL_TURN = 1.0, 0.015, 0.10   # no progress for 1 s (< 1.5 cm, < 6 deg) ...
STALL_ERR, STALL_BERR = 0.15, 0.6                   # ... while near (15 cm, 35 deg) and the person stands -> REST
HEAD_BY_CONTROLLER = [False]  # H toggles: head joints = look command, policy head output ignored (helps volg17, hurts volg16)
PATIENT = [True]   # the person waits for the duck (X in the viewer toggles)
rng = np.random.default_rng(0)


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


class Target:
    def __init__(self, robot, bearing=0.0, v=0.0, w=0.0):
        self.r = robot
        self.held = np.array([0.0, FOLLOW_DIST])
        self.place(bearing)
        self.yaw = self.heading() ; self.v, self.w = v, w
        self.buf = []; self.waiting = False

    def heading(self):
        return yaw_of(self.r.d.qpos[3:7])

    def place(self, bearing):
        h = self.heading() + bearing
        self.p = self.r.d.qpos[:2] + FOLLOW_DIST * np.array([math.cos(h), math.sin(h)])
        self.held = np.array([bearing, FOLLOW_DIST])   # as in training: it saw where it went

    def toward(self, bearing=0.2, v=0.06):
        """Start 1 m away and walk towards / past the duck."""
        self.place(bearing); self.p = self.r.d.qpos[:2] + 2 * (self.p - self.r.d.qpos[:2])
        to = self.r.d.qpos[:2] - self.p; self.yaw = math.atan2(to[1], to[0]) + 0.2; self.v, self.w = v, 0.0

    def v_now(self):
        return 0.0 if self.waiting else self.v

    def step(self, dt=0.02):
        d, _ = self.relative(); rel = self.p - self.r.d.qpos[:2]
        away = math.cos(self.yaw) * rel[0] + math.sin(self.yaw) * rel[1] > 0
        self.waiting = PATIENT[0] and ((self.waiting and d > WAIT_NEAR) or d > WAIT_FAR) and away   # a person waits
        if not self.waiting:
            self.yaw += self.w * dt
            self.p = self.p + self.v * dt * np.array([math.cos(self.yaw), math.sin(self.yaw)])

    def relative(self):
        rel = self.p - self.r.d.qpos[:2]; h = self.heading(); c, s = math.cos(h), math.sin(h)
        dx, dy = c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1]
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def twist(self):
        """Same controller as training (gait clock + rest mode): rest hysteresis, back-off
        hysteresis and a rate-limited (smooth) command. Call once per control step."""
        d, b = self.relative(); h = self.heading()
        moving = self.v_now() > 1e-4
        face = float(np.clip(1 - abs(b), 0, 1))
        vx = max((K_DIST * (d - FOLLOW_DIST) + self.v_now() * math.cos(self.yaw - h)) * face, 0.0)
        wz = K_BEARING * b
        err, berr = abs(d - FOLLOW_DIST), abs(b)
        # stall watchdog (2026-10-15): stepping on the spot without getting anywhere -> rest
        hist = getattr(self, "hist", []); hist.append((self.r.d.qpos[:2].copy(), h)); self.hist = hist[-int(STALL_S / 0.02) - 1:]
        q0, h0 = self.hist[0]
        stalled = (len(self.hist) > STALL_S / 0.02 and np.linalg.norm(self.r.d.qpos[:2] - q0) < STALL_MOVE
                   and abs((h - h0 + math.pi) % (2 * math.pi) - math.pi) < STALL_TURN and err < STALL_ERR and berr < STALL_BERR)
        self.resting = (not moving) and ((err < REST_IN and berr < REST_IN_B) or stalled
                                         or (getattr(self, "resting", False) and err < REST_OUT and berr < REST_OUT_B))
        if self.resting: vx, wz = 0.0, 0.0
        self.backing = d < BACK_DIST or (getattr(self, "backing", False) and d < BACK_RELEASE)
        # walk / run zones (as in training, volg17): walk capped at WALK_MAX_V, run = fixed RUN_V
        self.running = (not self.backing) and berr < 0.5 and (d > RUN_DIST or (getattr(self, "running", False) and d > RUN_RELEASE))
        vx = 0.0 if self.resting else (RUN_V if self.running else min(vx, WALK_MAX_V))
        if self.backing:
            vx = -float(np.clip(K_DIST * (BACK_RELEASE - d), BACK_MIN_V, BACK_MAX_V)) * (berr < BACK_FACE)
        raw_v, raw_w = float(np.clip(vx, -BACK_MAX_V, VMAX)), float(np.clip(wz, -0.6, 0.6))
        cv, cw = getattr(self, "cmd_v", raw_v), getattr(self, "cmd_w", raw_w)
        cv += float(np.clip(raw_v - cv, -ACC_V * 0.02, ACC_V * 0.02))
        cw += float(np.clip(raw_w - cw, -ACC_W * 0.02, ACC_W * 0.02))
        if self.resting and abs(cv) < 0.01: cv = 0.0
        if self.resting and abs(cw) < 0.05: cw = 0.0
        self.cmd_v, self.cmd_w = cv, cw
        return np.array([cv, 0.0, cw])

    def obs(self):
        d, b = self.relative()
        bn, dn = b + BEARING_NOISE * rng.standard_normal(), d * (1 + DIST_NOISE * rng.standard_normal())
        vis = abs(b) < FOV_HALF and rng.random() > DROPOUT
        if vis: self.held = np.array([bn, dn])
        self.buf.append(np.array([self.held[0], float(np.clip(self.held[1] - FOLLOW_DIST, -0.5, 1.0)), float(vis)]))
        self.buf = self.buf[-(DELAY + 1):]
        return self.buf[0]


PERSON_GO, PERSON_STOP = 0.015, 0.008   # m/s: the person "walks" above 1.5 cm/s, "stands" again below 0.8


def person_speed(t, sp):
    """Speed of the person from the mouse/arrow motion, with a dead band + hysteresis.
    2026-10-15: 0.3 mm of mouse jitter read as 0.4 cm/s = 'walking' -> rest never engaged."""
    t.walking = sp > PERSON_GO or (getattr(t, "walking", False) and sp > PERSON_STOP)
    return min(sp, VMAX) if t.walking else 0.0


def head_fix(robot, a):
    """Head driven by the controller: the policy's head output is replaced by the look
    command (here: straight ahead). volg17 held its head 20-35 deg left and 20 deg rolled,
    which cost it the turn to the right."""
    if HEAD_BY_CONTROLLER[0]:
        if not hasattr(robot, "_head_ids"):
            robot._head_ids = [robot.names.index(j) for j in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
        a = a.copy(); a[robot._head_ids] = 0.0
    return a


def policy_obs(robot, tgt, twist):
    o = robot.obs(twist, np.zeros(4))          # builds everything incl. gait clock from the twist
    o[48:51] = tgt.obs()                       # ... but the policy sees the target, not the twist
    return o


def run(pol, grid, bearing=0.0, v=0.0, w=0.0, seconds=8.0, phase=0.0, toward=False):
    r = P.Duckling("duckling/src/mjlab_microduck/robot/duckling/scene_walk.xml", grid=grid); r.gait_clock = True; r.phase = phase
    t = Target(r, bearing, v, w); D, B, face_t = [], [], None
    if toward: t.toward()
    p0 = r.d.qpos[:2].copy()
    for k in range(int(seconds / 0.02)):
        tw = t.twist(); r.step(head_fix(r, pol(policy_obs(r, t, tw))), tw); t.step()
        d, b = t.relative()
        if face_t is None and abs(b) < math.radians(10): face_t = k * 0.02
        if k * 0.02 > 2.0: D.append(abs(d - FOLLOW_DIST)); B.append(abs(b))
        if r.fallen: return None
    return dict(d=np.mean(D) * 1000, b=math.degrees(np.mean(B)), face=face_t,
                walked=np.linalg.norm(r.d.qpos[:2] - p0) / seconds * 1000)


def meet(pol, grid):
    sc = [("doel staat stil voor hem", 0.0, 0.0, 0.0), ("doel loopt 4 cm/s", 0.0, 0.04, 0.0),
          ("doel loopt 8 cm/s", 0.0, 0.08, 0.0), ("doel loopt 13 cm/s", 0.0, 0.13, 0.0), ("doel bocht L (8 cm/s)", 0.0, 0.08, 0.4),
          ("doel bocht R (8 cm/s)", 0.0, 0.08, -0.4), ("doel springt 90° L", 1.57, 0.0, 0.0),
          ("doel springt 90° R", -1.57, 0.0, 0.0), ("doel achter hem", 3.0, 0.0, 0.0),
          ("doel loopt langs hem", None, 0.0, 0.0)]
    print("na de eerste 2 s, gemiddeld over 4 startfases:")
    for name, b, v, w in sc:
        R = [run(pol, grid, b or 0.0, v, w, phase=ph, toward=b is None, seconds=12 if b is None else 8)
             for ph in (0, .25, .5, .75)]
        fell = sum(x is None for x in R); R = [x for x in R if x]
        if not R: print(f"  {name:26s} VALT"); continue
        f = [x["face"] for x in R if x["face"] is not None]
        print(f"  {name:26s} afstandsfout {np.mean([x['d'] for x in R]):4.0f} mm  richtingsfout {np.mean([x['b'] for x in R]):4.0f}°"
              f"  gedraaid naar doel na {np.mean(f) if f else float('nan'):4.1f} s  snelheid {np.mean([x['walked'] for x in R]):4.0f} mm/s"
              + (f"  VALT {fell}/4" if fell else ""))


def _scene_with_person(scene):
    """The robot scene plus a mocap 'person' (two legs + nose, no collisions) you can grab
    with the mouse: double-click it, then Ctrl + right-drag moves it over the floor."""
    spec = mujoco.MjSpec.from_file(scene)
    b = spec.worldbody.add_body(name="persoon", mocap=True, pos=[FOLLOW_DIST, 0, 0])
    for y in (0.05, -0.05):
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.02, 0.15, 0], pos=[0, y, 0.15],
                   contype=0, conaffinity=0, rgba=[0.9, 0.1, 0.1, 1])
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.015, 0, 0], pos=[0.06, 0, 0.30],
               contype=0, conaffinity=0, rgba=[0.9, 0.1, 0.1, 1])
    return spec


def viewer(pol, grid):
    """The person is moved DIRECTLY: grab it with the mouse (double-click, then Ctrl + right
    drag) or step it with the arrow keys (5 cm, in the view's ground plane: up = away from
    you). Its speed and walking direction follow from how you move it."""
    import mujoco.viewer
    r = P.Duckling("duckling/src/mjlab_microduck/robot/duckling/scene_walk.xml", grid=grid, model=_scene_with_person("duckling/src/mjlab_microduck/robot/duckling/scene_walk.xml").compile()); r.gait_clock = True
    mid = r.m.body("persoon").mocapid[0]
    t = Target(r); flags = dict(reset=False); step = 0.05
    r.d.mocap_pos[mid][:2] = t.p

    def view_dirs(v):
        az = math.radians(v.cam.azimuth)
        fwd = np.array([math.cos(az), math.sin(az)]); left = np.array([-fwd[1], fwd[0]])
        return fwd, left

    def key(k):
        g = P.GLFW
        fwd, left = view_dirs(viewer_ref[0])
        dp = {g.KEY_UP: fwd, g.KEY_DOWN: -fwd, g.KEY_LEFT: left, g.KEY_RIGHT: -left}.get(k)
        if dp is not None: r.d.mocap_pos[mid][:2] += step * dp
        elif k == g.KEY_J: t.place(1.57); r.d.mocap_pos[mid][:2] = t.p
        elif k == g.KEY_L: t.place(-1.57); r.d.mocap_pos[mid][:2] = t.p
        elif k == g.KEY_BACKSPACE: flags["reset"] = True
        elif k == ord("H"): HEAD_BY_CONTROLLER[0] = not HEAD_BY_CONTROLLER[0]; print("kop door controller:", HEAD_BY_CONTROLLER[0])
        elif k == ord("G"): P.FOLLOW[0] = not P.FOLLOW[0]; print("camera volgt:", P.FOLLOW[0])

    viewer_ref = [None]
    P.fast_model(r.m)
    with mujoco.viewer.launch_passive(r.m, r.d, key_callback=key) as v:
        viewer_ref[0] = v
        P.fast_render(v); rest = 1.0; t_next = time.time()
        P.FOLLOW[0] = False
        with v.lock():
            v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            v.cam.trackbodyid = -1
            v.cam.lookat[:] = [0.15, 0.0, 0.05]; v.cam.distance = 1.2; v.cam.azimuth = 135; v.cam.elevation = -30
        print("camera staat vast (G = laten volgen).")
        print("persoon: dubbelklik erop, dan Ctrl + rechtermuis slepen - of pijltjes (5 cm per tik).")
        vel = np.zeros(2); shown = None
        while v.is_running():
            if flags["reset"] or r.fallen:
                if r.fallen: print("gevallen - reset")
                r.reset(); t = Target(r); r.d.mocap_pos[mid][:2] = t.p; vel[:] = 0; flags["reset"] = False
            # the person IS where you put it; speed and heading follow from the motion
            new = r.d.mocap_pos[mid][:2].copy()
            vel = 0.8 * vel + 0.2 * (new - t.p) / 0.02
            t.p = new
            sp = float(np.linalg.norm(vel))
            t.v = person_speed(t, sp)
            if sp > 0.01:
                t.yaw = math.atan2(vel[1], vel[0])
            t.w = 0.0
            yq = np.zeros(4); mujoco.mju_axisAngle2Quat(yq, [0, 0, 1], t.yaw); r.d.mocap_quat[mid] = yq
            tw = t.twist(); a = head_fix(r, pol(policy_obs(r, t, tw)))
            standing = not (abs(tw[0]) > 0.02 or abs(tw[2]) > 0.1)
            rest = float(np.clip(rest + (-0.02 if standing else 0.02) / P.REST_FADE_S, 0, 1))
            r.step(rest * a, tw)
            d, b = t.relative()
            state = ("te dichtbij - eend stapt terug" if t.backing else "") + ("eend RENT" if t.running else "") + ("rust - netwerk uit" if t.resting else "")
            if state != shown:
                if state: print("persoon:", state)
                shown = state
            if P.FOLLOW[0]: v.cam.lookat[:] = r.d.qpos[:3]
            v.sync(); t_next += 0.02; time.sleep(max(0.0, t_next - time.time()))
            if time.time() - t_next > 0.5: t_next = time.time()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx"); ap.add_argument("--grid", required=True); ap.add_argument("--meet", action="store_true")
    ap.add_argument("--mooi", action="store_true")
    a = ap.parse_args()
    pol = P.Policy(a.onnx); grid = P.GridClock(a.grid)
    meet(pol, grid) if a.meet else viewer(pol, grid)
