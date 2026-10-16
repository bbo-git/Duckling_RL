"""
gait_gen.py - parametric walk generator for the Duckling (no animation, no training).

A gait is a handful of numbers (cycle time, stride, foot lift, trunk height, bob, stance
width, sway, waddle). From those the generator plans the feet and the trunk, then
solves the leg joint angles with inverse kinematics on the MuJoCo model, so every gait
is built from the robot's real kinematics:

  * feet: stance foot stays put on the floor (flat sole), swing foot follows a smooth arc
  * trunk: moves forward at the walking speed and sways sideways over the stance foot
    following the linear inverted pendulum (sway = 1.0 is the physically balanced
    amount for that cadence), with an optional bob and waddle (trunk roll)

Output: the same walk_cycle.npz the training uses (joint angles, trunk pose, contact
schedule, trunk height, speed), plus the balance and motor-speed check.

    python gait_gen.py --view                        # play it, change it live with keys
    python gait_gen.py --preset sluipen --view
    python gait_gen.py --set cycle_s=0.6 stride_m=0.05 --save walk_cycle.npz
    python gait_gen.py --list                        # presets

Sliders: open the "Control" panel on the right of the viewer; the robot's 14 motors come
first (ignore those), then the gait sliders (stap_mm, doorzakken_mm, kop_knik_gr, ...).
Let go of a slider and the gait is regenerated (~1 s).

Keys in the viewer (focus on the MuJoCo window; each key regenerates the gait):
    1 / 2  cycle time -/+        3 / 4  stride -/+          5 / 6  foot lift -/+
    7 / 8  trunk height -/+      9 / 0  sway -/+            B / N  bob -/+
    W / E  stance width -/+ (0 = automatic)   R / T  waddle -/+   D / F  double support -/+
    H / J  head nod -/+          K / L  head counter-nod -/+
    O / I  head sway -/+         U / Y  head look-around -/+
    S      save walk_cycle.npz   P      print the parameters
"""
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, fields

import mujoco

FOLLOW = [True]   # camera follows the robot; G in the viewer toggles
import numpy as np

JOINT_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "neck_pitch", "head_pitch", "head_roll", "head_yaw",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HOME_DEG = {"hip_pitch": -11.8, "knee": -8.6, "ankle": -20.3}
FPS = 50.0
G = 9.81


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


@dataclass
class GaitParams:
    cycle_s: float = 0.4        # one full cycle (two steps)
    stride_m: float = 0.03      # distance per cycle = forward speed * cycle_s
    lift_m: float = 0.010       # foot clearance at mid-swing
    height_m: float = 0.0545    # trunk_base height above the floor (HOME = 0.0556)
    bob_m: float = 0.002        # trunk up/down per step
    width_m: float = 0.0        # sideways distance between the feet; 0 = automatic: legs
                                # vertical at mid-stance (flat soles, no ankle roll needed)
    sway: float = 1.0           # sideways trunk sway, 1.0 = inverted-pendulum balance
    waddle: float = 0.0         # extra trunk roll toward the stance side (character)
    double_support: float = 0.10  # fraction of the cycle with both feet down, per step
    pitch_deg: float = 0.0      # trunk lean forward (+)
    turn_dps: float = 0.0       # turning rate, degrees per second (+ = left)
    side_m: float = 0.0         # sideways distance per cycle (+ = left): sidestep / orbit
    # head (defaults fitted on the Blender walk cycle): the neck nods twice per cycle and
    # the head pitches against it (a bird keeping its gaze steady), plus a slow sway/look
    nod_deg: float = 14.0       # neck pitch amplitude
    nod_counter: float = 1.0    # head pitch against the nod (0 = head rides along)
    neck_lean_deg: float = 4.6  # mean neck pitch
    head_tilt_deg: float = -12.0  # mean head pitch
    head_sway_deg: float = 6.0  # head roll, once per cycle
    head_look_deg: float = 5.0  # head yaw, once per cycle
    # follow-through ("doorveren"): after every foot landing the trunk sinks and springs back
    dip_m: float = 0.002        # how far the trunk sinks after a landing
    dip_hz: float = 6.0         # spring frequency of the rebound (lower = softer, slower)
    dip_damping: float = 0.45   # 0.2 = bouncy (overshoots), 1.0 = no overshoot
    head_follow_deg: float = 6.0  # head pitch dragging behind the dip (overlapping action)


PRESETS = {
    # Width 0 = automatic (legs vertical at mid-stance, flat soles). The slower the cycle,
    # the wider the stance and the bigger the sideways sway this robot needs.
    "normaal": GaitParams(cycle_s=0.40, stride_m=0.030, lift_m=0.010),
    "drentelen": GaitParams(cycle_s=0.32, stride_m=0.020, lift_m=0.007, bob_m=0.001),
    "parmantig": GaitParams(cycle_s=0.36, stride_m=0.030, lift_m=0.012, height_m=0.0565,
                            bob_m=0.003, pitch_deg=-3.0, nod_deg=18.0, head_tilt_deg=-16.0),
    "sluipen": GaitParams(cycle_s=0.50, stride_m=0.020, lift_m=0.014, height_m=0.050,
                          bob_m=0.0, double_support=0.15, pitch_deg=4.0, nod_deg=5.0,
                          head_look_deg=12.0, head_tilt_deg=-5.0),
    "waggelen": GaitParams(cycle_s=0.50, stride_m=0.025, lift_m=0.008, bob_m=0.002, waddle=0.3,
                           head_sway_deg=12.0),
}


def _dip_response(tau, hz, zeta):
    """Damped-spring response to a landing at tau = 0, scaled so the deepest point = -1."""
    w = 2 * math.pi * max(hz, 0.5)
    zeta = min(max(zeta, 0.05), 0.99)
    wd = w * math.sqrt(1 - zeta**2)
    tt = np.linspace(0, 3.0 / max(hz, 0.5), 400)
    peak = np.max(np.exp(-zeta * w * tt) * np.sin(wd * tt))
    tau = np.asarray(tau, float)
    y = np.where(tau >= 0, np.exp(-zeta * w * tau) * np.sin(wd * tau), 0.0)
    return -y / peak


def dip_curve(t, T, swing, hz, zeta):
    """Sum of the responses to both landings (left at phase swing, right at 0.5 + swing),
    over a few earlier cycles so the curve is periodic."""
    out = np.zeros_like(t, dtype=float)
    for land in (swing, 0.5 + swing):
        for c in range(-3, 1):
            out += _dip_response(t - (land + c) * T, hz, zeta)
    return out


def _rot2(a):
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def _rotz_quat(yaw, quat):
    """World quaternion = rotation about z by yaw, then the local (roll/pitch) quat."""
    qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, quat)
    return out


def _smooth(u):
    """0 -> 1 with zero velocity at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# Sliders in the MuJoCo viewer ("Control" panel): (parameter, label, lo, hi, scale to SI)
SLIDERS = [
    ("cycle_s", "cyclus_s", 0.28, 0.70, 1.0),
    ("stride_m", "stap_mm", -20.0, 50.0, 0.001),
    ("lift_m", "voethoogte_mm", 4.0, 20.0, 0.001),
    ("height_m", "romphoogte_mm", 48.0, 58.0, 0.001),
    ("bob_m", "op_neer_mm", 0.0, 6.0, 0.001),
    ("dip_m", "doorzakken_mm", 0.0, 8.0, 0.001),
    ("dip_hz", "veer_Hz", 2.0, 12.0, 1.0),
    ("dip_damping", "veer_demping", 0.15, 0.95, 1.0),
    ("waddle", "waggel", 0.0, 1.0, 1.0),
    ("pitch_deg", "romp_voorover_gr", -8.0, 10.0, 1.0),
    ("nod_deg", "kop_knik_gr", 0.0, 25.0, 1.0),
    ("nod_counter", "kop_tegenknik", 0.0, 1.5, 1.0),
    ("head_follow_deg", "kop_naslepen_gr", 0.0, 20.0, 1.0),
    ("head_sway_deg", "kop_zwaai_gr", 0.0, 18.0, 1.0),
    ("head_look_deg", "kop_rondkijken_gr", 0.0, 25.0, 1.0),
    ("head_tilt_deg", "kop_kanteling_gr", -25.0, 10.0, 1.0),
]


def _model_with_sliders(scene):
    """The robot plus a hidden body with one slider joint + actuator per parameter, so the
    viewer's Control panel shows them as sliders. Only used for kinematic playback."""
    spec = mujoco.MjSpec.from_file(scene)
    for _, label, lo, hi, _ in SLIDERS:
        body = spec.worldbody.add_body(name="_body_" + label, pos=[0, 0, -1])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0,
                      conaffinity=0, rgba=[0, 0, 0, 0], mass=1e-6)
        body.add_joint(name="_" + label, type=mujoco.mjtJoint.mjJNT_SLIDE, range=[lo, hi])
        a = spec.add_actuator(name=label, target="_" + label, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.ctrlrange = [lo, hi]
        a.ctrllimited = 1
    return spec.compile()


class DuckKinematics:
    def __init__(self, scene, sliders=False):
        self.m = _model_with_sliders(scene) if sliders else mujoco.MjModel.from_xml_path(scene)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.names = JOINT_ORDER
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINT_ORDER])
        self.home = np.zeros(len(JOINT_ORDER))
        for i, n in enumerate(JOINT_ORDER):
            for k, v in HOME_DEG.items():
                if n.endswith("_" + k):
                    self.home[i] = math.radians(v)
        self.legs = {}
        for side, b in (("left", "L"), ("right", "R")):
            cols = [JOINT_ORDER.index(f"{side}_{j}") for j in
                    ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
            jids = [m.joint(f"{side}_{j}").id for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
            self.legs[side] = dict(cols=cols, dof=np.array([m.jnt_dofadr[j] for j in jids]),
                                   lo=np.array([m.jnt_range[j][0] for j in jids]),
                                   hi=np.array([m.jnt_range[j][1] for j in jids]),
                                   site=m.site(f"{side}_foot").id, toe=m.body(f"{b}_toe").id,
                                   sole=m.geom(f"{side}_foot_collision").id)
        self.trunk = m.body("trunk_base").id
        self.hip_y = float(m.body("L_hip_yaw").pos[1])          # hip offset from the trunk centre
        # HOME reference: foot sites in the trunk frame, sole bottom below the site, CoM height
        self._set(np.array([0, 0, 0.0556]), np.array([1.0, 0, 0, 0]), self.home)
        self.site_above_floor = {}
        for side, L in self.legs.items():
            g = L["sole"]
            mid = m.geom_dataid[g]; va = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
            w = m.mesh_vert[va:va + nv] @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]
            self.site_above_floor[side] = self.d.site_xpos[L["site"]][2] - w[:, 2].min()
            L["home_site"] = self.d.site_xpos[L["site"]].copy()
        mujoco.mj_comPos(m, self.d)
        self.com_height = self.d.subtree_com[self.trunk][2] - (0.0556 - 0.0556)  # with trunk at 55.6 mm
        self.com_offset = self.d.subtree_com[self.trunk][:2] - np.zeros(2)

    def _set(self, pos, quat, q):
        self.d.qpos[:3] = pos
        self.d.qpos[3:7] = quat
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)

    def solve_leg(self, side, q, target_pos, target_rot, iters=40):
        """Damped least squares: sole site to target_pos, toe body to target_rot (flat)."""
        m, d, L = self.m, self.d, self.legs[side]
        jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
        w_rot = np.array([0.02, 0.03, 0.05])    # m per rad: roll, pitch, yaw (roll softest)
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_pos = target_pos - d.site_xpos[L["site"]]
            R = d.xmat[L["toe"]].reshape(3, 3)
            e_rot = _rotvec(target_rot @ R.T)
            mujoco.mj_jacSite(m, d, jacp, None, L["site"])
            mujoco.mj_jacBody(m, d, None, jacr, L["toe"])
            J = np.vstack([jacp[:, L["dof"]], jacr[:, L["dof"]] * w_rot[:, None]])
            e = np.concatenate([e_pos, e_rot * w_rot])
            if np.linalg.norm(e) < 2e-5:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
            q[L["cols"]] = np.clip(q[L["cols"]] + dq, L["lo"], L["hi"])
        return q

    def _feet(self, p, t, T, v, swing):
        """Foot targets and contact flags for one frame. Each foot is planned in the moving
        frame (frame_at): a stance foot is put down at its mid-stance frame and stays there;
        a swing foot travels from the previous to the next mid-stance frame. Returns world
        positions (sole sites) and world yaw per foot."""
        ph = (t / T) % 1.0
        half = p.width_m / 2
        out, yaw, c = {}, {}, []
        for side, start in (("left", 0.0), ("right", 0.5)):
            lp = (ph - start) % 1.0                          # 0 = lift-off of this foot
            t_lo = t - lp * T
            mid = (swing + 1.0) / 2.0                        # stance centred on the trunk
            tau_next = t_lo + mid * T
            if lp < swing:
                s = _smooth(lp / swing)
                tau = tau_next - T + s * T
                fz = p.lift_m * math.sin(math.pi * lp / swing) ** 1.5
                c.append(False)
            else:
                tau, fz = tau_next, 0.0
                c.append(True)
            local = np.array([self.legs[side]["home_site"][0], half if side == "left" else -half])
            fp, fyaw = self.frame_at(p, tau, T)
            out[side] = np.array([*(fp + _rot2(fyaw) @ local), fz + self.site_above_floor[side]])
            yaw[side] = fyaw
        return out, c, yaw

    @staticmethod
    def frame_at(p, t, T):
        """Pose (xy, yaw) at time t of the frame that moves with the gait: forward
        stride/T, sideways side/T (in the frame), turning turn_dps."""
        vx, vy = p.stride_m / T, p.side_m / T
        w = math.radians(p.turn_dps)
        if abs(w) < 1e-6:
            return np.array([vx * t, vy * t]), 0.0
        sn, cs = math.sin(w * t), math.cos(w * t)
        return np.array([vx * sn / w - vy * (1 - cs) / w, vx * (1 - cs) / w + vy * sn / w]), w * t

    def _head(self, p, ph, q, dip=0.0):
        """Head joints as a function of the gait phase (0 = left foot lifts off). `dip` is
        the delayed trunk dip (-1 = deepest): the head drops forward after the trunk."""
        r = math.radians
        J = {n: i for i, n in enumerate(JOINT_ORDER)}
        nod = math.cos(4 * math.pi * (ph - 0.05))
        q[J["neck_pitch"]] = r(p.neck_lean_deg + p.nod_deg * nod)
        q[J["head_pitch"]] = r(p.head_tilt_deg - p.nod_counter * p.nod_deg * math.cos(4 * math.pi * (ph - 0.10))
                               - p.head_follow_deg * dip)
        q[J["head_roll"]] = r(p.head_sway_deg * math.cos(2 * math.pi * (ph - 0.98)))
        q[J["head_yaw"]] = r(p.head_look_deg * math.cos(2 * math.pi * (ph - 0.77)))
        for n in ("neck_pitch", "head_pitch", "head_roll", "head_yaw"):
            j = self.m.joint(n).id
            q[J[n]] = float(np.clip(q[J[n]], *(0.9 * self.m.jnt_range[j])))
        return q

    def _solve_frame(self, pos, roll, pitch, q, feet, yaw=0.0, foot_yaw=None):
        local = np.zeros(4)
        mujoco.mju_euler2Quat(local, np.array([roll, pitch, 0.0]), "xyz")
        quat = _rotz_quat(yaw, local)
        self._set(pos, quat, q)
        for side in ("left", "right"):
            a = (foot_yaw or {}).get(side, 0.0)
            Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            q = self.solve_leg(side, q, feet[side], Rz)
        return q, quat, local

    def generate(self, p: GaitParams, com_passes=4):
        n = int(round(p.cycle_s * FPS))
        T = n / FPS
        t = np.arange(n) / FPS
        v = p.stride_m / T
        swing = max(0.5 - p.double_support, 0.15)            # swing fraction per foot
        w = math.sqrt(G / max(self.com_height, 0.03))
        Ts = T / 2
        if p.width_m <= 0:
            # Without ankle roll a sole is flat only when its leg is vertical (seen from the
            # front). Feet under the hips while the trunk sways A toward the stance foot:
            # LIPM sway A = p (1 - 1/cosh(w Ts/2)) with p = A + hip offset  ->  A = h (cosh - 1)
            hip = abs(self.hip_y)
            A = p.sway * hip * (math.cosh(w * Ts / 2) - 1)
            p = GaitParams(**{**asdict(p), "width_m": 2 * (hip + A)})
            self.last_width = p.width_m
        half = p.width_m / 2
        feet, contact, fyaw = zip(*[self._feet(p, tk, T, v, swing) for tk in t])
        contact = np.array(contact)
        frames = [self.frame_at(p, tk, T) for tk in t]       # moving frame per frame
        fpos = np.array([f[0] for f in frames]); fyw = np.array([f[1] for f in frames])
        # Everything below is planned in the moving frame ("local"): x forward, y left.
        ph = (t / T) % 1.0
        u = (ph % 0.5) / 0.5
        sgn = np.where(ph < 0.5, -1.0, 1.0)                  # left swings first: lean right
        shape = 1 - np.cosh(w * Ts * (u - 0.5)) / math.cosh(w * Ts / 2)
        com_y_des = p.sway * sgn * half * shape
        x = np.zeros(n)
        y = com_y_des.copy()
        self._dip = dip_curve(t, T, swing, p.dip_hz, p.dip_damping)   # -1 .. ~+0.3
        z = p.height_m + p.bob_m * np.cos(4 * math.pi * ph) + p.dip_m * self._dip
        roll = np.zeros(n)
        pitch = math.radians(p.pitch_deg)

        def to_local(k, xy):
            return _rot2(-fyw[k]) @ (np.asarray(xy) - fpos[k])

        def to_world(k, xy):
            return fpos[k] + _rot2(fyw[k]) @ np.asarray(xy)

        foot_x = np.array([np.mean([to_local(k, f[s][:2])[0] for s, cc in zip(("left", "right"), c) if cc] or
                                   [np.mean([to_local(k, f[s][:2])[0] for s in ("left", "right")])])
                           for k, (f, c) in enumerate(zip(feet, contact))])
        q = self.home.copy()
        for it in range(com_passes):
            roll = -p.waddle * np.arctan2(y, z)
            out_q, out_quat, out_local, com = [], [], [], []
            for warm, k in [(True, kk) for kk in range(n - 5, n)] + [(False, kk) for kk in range(n)]:
                pos = np.array([*to_world(k, [x[k], y[k]]), z[k]])
                q, quat, lq = self._solve_frame(pos, roll[k], pitch, q, feet[k], fyw[k], fyaw[k])
                if warm:
                    continue
                q = self._head(p, t[k] / T, q, self._dip[(k - 3) % n])   # 60 ms behind
                self._set(pos, quat, q)                          # CoM includes the head motion
                mujoco.mj_comPos(self.m, self.d)
                out_q.append(q.copy()); out_quat.append(quat.copy()); out_local.append(lq.copy())
                com.append(np.array([*to_local(k, self.d.subtree_com[self.trunk][:2]),
                                     self.d.subtree_com[self.trunk][2]]))
            com = np.array(com)
            y += com_y_des - com[:, 1]
            # forward: CoM centred over the feet (the trunk counter-moves the head nod)
            x += np.mean(foot_x) - com[:, 0]
        root_pos = np.stack([*np.array([to_world(k, [x[k], y[k]]) for k in range(n)]).T, z], 1)
        vel = np.array([v, p.side_m / T])
        return dict(t=t, joint_pos=np.array(out_q), root_pos=root_pos, root_quat=np.array(out_quat),
                    root_local=np.stack([x, y, z], 1), root_quat_local=np.array(out_local),
                    frame_pos=fpos, frame_yaw=fyw, yaw_rate=math.radians(p.turn_dps),
                    contact=contact, root_height=z.copy(), vel_xy=vel, cycle_s=T,
                    fps=FPS, joint_names=np.array(JOINT_ORDER), params=json.dumps(asdict(p)),
                    width_m=p.width_m)


def _rotvec(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(9))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def report(kin, g, scene):
    """Balance and motor-speed check (same as convert_anim.py) plus foot accuracy."""
    import os, tempfile
    from convert_anim import feasibility
    f = os.path.join(tempfile.gettempdir(), "_gait_check.npz")
    np.savez(f, **g)
    p = json.loads(str(g["params"]))
    print(f"\ncyclus {g['cycle_s']:.2f} s, snelheid {g['vel_xy'][0]:.3f} m/s, "
          f"stap {p['stride_m'] / 2 * 1000:.0f} mm, {2 / g['cycle_s']:.1f} stappen/s, "
          f"voeten {float(g['width_m']) * 1000:.0f} mm uit elkaar")
    feasibility(f, scene)


def generate_checked(kin, p):
    return kin.generate(p)


def view(kin, p, scene, save_path):
    import mujoco.viewer
    state = dict(p=p, g=generate_checked(kin, p), dirty=False)
    steps = {"cycle_s": 0.05, "stride_m": 0.005, "lift_m": 0.002, "height_m": 0.001,
             "sway": 0.1, "bob_m": 0.001, "width_m": 0.002, "waddle": 0.1, "double_support": 0.02,
             "nod_deg": 2.0, "head_sway_deg": 2.0, "head_look_deg": 2.0, "nod_counter": 0.1}
    keymap = {ord("1"): ("cycle_s", -1), ord("2"): ("cycle_s", 1), ord("3"): ("stride_m", -1),
              ord("4"): ("stride_m", 1), ord("5"): ("lift_m", -1), ord("6"): ("lift_m", 1),
              ord("7"): ("height_m", -1), ord("8"): ("height_m", 1), ord("9"): ("sway", -1),
              ord("0"): ("sway", 1), ord("B"): ("bob_m", -1), ord("N"): ("bob_m", 1),
              ord("W"): ("width_m", -1), ord("E"): ("width_m", 1), ord("R"): ("waddle", -1),
              ord("T"): ("waddle", 1), ord("D"): ("double_support", -1), ord("F"): ("double_support", 1),
              ord("H"): ("nod_deg", -1), ord("J"): ("nod_deg", 1), ord("K"): ("nod_counter", -1),
              ord("L"): ("nod_counter", 1), ord("O"): ("head_sway_deg", -1), ord("I"): ("head_sway_deg", 1),
              ord("U"): ("head_look_deg", -1), ord("Y"): ("head_look_deg", 1)}

    def key(k):
        if k in keymap:
            name, sgn = keymap[k]
            cur = getattr(state["p"], name)
            setattr(state["p"], name, round(max(0.0, cur + sgn * steps[name]), 4))  # magnitudes
            state["dirty"] = True
            print(f"{name} = {getattr(state['p'], name)}")
        elif k == ord("G"):
            FOLLOW[0] = not FOLLOW[0]; print("camera volgt:", FOLLOW[0])
        elif k == ord("P"):
            print(json.dumps(asdict(state["p"]), indent=1))
        elif k == ord("S"):
            np.savez(save_path, **state["g"])
            print("opgeslagen:", save_path)
            report(kin, state["g"], scene)

    m, d = kin.m, kin.d
    slider_ids = {}
    for name, label, lo, hi, sc in SLIDERS:
        try:
            slider_ids[name] = (m.actuator(label).id, sc)
        except KeyError:
            pass

    def to_sliders():
        for name, (aid, sc) in slider_ids.items():
            d.ctrl[aid] = getattr(state["p"], name) / sc

    to_sliders()
    last = {"vals": None, "t": 0.0}
    fast_model(m)
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
        fast_render(v)
        loop, k = 0, 0
        while v.is_running():
            vals = tuple(round(float(d.ctrl[aid]), 4) for aid, _ in slider_ids.values())
            if last["vals"] is None:
                last["vals"] = vals
            elif vals != last["vals"]:
                last["vals"], last["t"] = vals, time.time()
                last["pending"] = True
            elif last.get("pending") and time.time() - last["t"] > 0.3:   # slider let go
                for (name, (aid, sc)) in slider_ids.items():
                    setattr(state["p"], name, round(float(d.ctrl[aid]) * sc, 5))
                state["dirty"] = True
                last["pending"] = False
            if state["dirty"]:
                to_sliders()
                state["g"] = generate_checked(kin, state["p"]); state["dirty"] = False
                loop, k = 0, 0
            g = state["g"]
            n = len(g["t"])
            stride = g["vel_xy"][0] * g["cycle_s"]
            d.qpos[:3] = g["root_pos"][k] + np.array([loop * stride, 0, 0])
            d.qpos[3:7] = g["root_quat"][k]
            d.qpos[kin.qadr] = g["joint_pos"][k]
            mujoco.mj_forward(m, d)
            if FOLLOW[0]:
                v.cam.lookat[:] = d.qpos[:3]
            v.sync()
            time.sleep(1 / FPS)
            k += 1
            if k == n:
                k, loop = 0, loop + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mooi", action="store_true", help="schaduwen en reflecties aan (trager)")
    ap.add_argument("--scene", default="scene_walk.xml")
    ap.add_argument("--preset", default="normaal", choices=list(PRESETS))
    ap.add_argument("--set", nargs="*", default=[], help="naam=waarde, bv. cycle_s=0.6")
    ap.add_argument("--params", help="JSON-bestand met parameters")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--save", default=None, help="schrijf walk_cycle.npz")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for k, v in PRESETS.items():
            print(f"{k:10s} {asdict(v)}")
        return
    p = GaitParams(**asdict(PRESETS[a.preset]))
    if a.params:
        for k, v in json.load(open(a.params)).items():
            setattr(p, k, float(v))
    for kv in a.set:
        k, v = kv.split("=")
        assert k in {f.name for f in fields(GaitParams)}, f"onbekende parameter {k}"
        setattr(p, k, float(v))
    kin = DuckKinematics(a.scene, sliders=a.view)
    if a.view:
        view(kin, p, a.scene, a.save or "walk_cycle.npz")
        return
    g = generate_checked(kin, p)
    if a.save:
        np.savez(a.save, **g)
        print("opgeslagen:", a.save)
    report(kin, g, a.scene)


if __name__ == "__main__":
    main()
