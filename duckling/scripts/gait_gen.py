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
import numpy as np

JOINT_ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
               "neck_pitch", "head_pitch", "head_roll", "head_yaw",
               "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
HOME_DEG = {"hip_pitch": -11.8, "knee": -8.6, "ankle": -20.3}
FPS = 50.0
G = 9.81


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
    # head (defaults fitted on the Blender walk cycle): the neck nods twice per cycle and
    # the head pitches against it (a bird keeping its gaze steady), plus a slow sway/look
    nod_deg: float = 14.0       # neck pitch amplitude
    nod_counter: float = 1.0    # head pitch against the nod (0 = head rides along)
    neck_lean_deg: float = 4.6  # mean neck pitch
    head_tilt_deg: float = -12.0  # mean head pitch
    head_sway_deg: float = 6.0  # head roll, once per cycle
    head_look_deg: float = 5.0  # head yaw, once per cycle


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


def _smooth(u):
    """0 -> 1 with zero velocity at both ends."""
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


class DuckKinematics:
    def __init__(self, scene):
        self.m = mujoco.MjModel.from_xml_path(scene)
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
        """World targets of both sole sites and the contact flags for one frame."""
        ph = (t / T) % 1.0
        half = p.width_m / 2
        out, c = {}, []
        for side, start in (("left", 0.0), ("right", 0.5)):
            lp = (ph - start) % 1.0                          # 0 = lift-off of this foot
            t_lo = t - lp * T
            mid = (swing + 1.0) / 2.0                        # stance centred on the trunk
            x_next = v * (t_lo + mid * T) + self.legs[side]["home_site"][0]
            if lp < swing:
                s = _smooth(lp / swing)
                fx = x_next - p.stride_m + s * p.stride_m
                fz = p.lift_m * math.sin(math.pi * lp / swing) ** 1.5
                c.append(False)
            else:
                fx, fz = x_next, 0.0
                c.append(True)
            fy = half if side == "left" else -half
            out[side] = np.array([fx, fy, fz + self.site_above_floor[side]])
        return out, c

    def _head(self, p, ph, q):
        """Head joints as a function of the gait phase (0 = left foot lifts off)."""
        r = math.radians
        J = {n: i for i, n in enumerate(JOINT_ORDER)}
        nod = math.cos(4 * math.pi * (ph - 0.05))
        q[J["neck_pitch"]] = r(p.neck_lean_deg + p.nod_deg * nod)
        q[J["head_pitch"]] = r(p.head_tilt_deg - p.nod_counter * p.nod_deg * math.cos(4 * math.pi * (ph - 0.10)))
        q[J["head_roll"]] = r(p.head_sway_deg * math.cos(2 * math.pi * (ph - 0.98)))
        q[J["head_yaw"]] = r(p.head_look_deg * math.cos(2 * math.pi * (ph - 0.77)))
        for n in ("neck_pitch", "head_pitch", "head_roll", "head_yaw"):
            j = self.m.joint(n).id
            q[J[n]] = float(np.clip(q[J[n]], *(0.9 * self.m.jnt_range[j])))
        return q

    def _solve_frame(self, pos, roll, pitch, q, feet):
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([roll, pitch, 0.0]), "xyz")
        self._set(pos, quat, q)
        for side in ("left", "right"):
            q = self.solve_leg(side, q, feet[side], np.eye(3))
        return q, quat

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
        feet, contact = zip(*[self._feet(p, tk, T, v, swing) for tk in t])
        contact = np.array(contact)
        # Desired sideways CoM path: linear inverted pendulum with the ZMP on the stance foot.
        ph = (t / T) % 1.0
        u = (ph % 0.5) / 0.5
        sgn = np.where(ph < 0.5, -1.0, 1.0)                  # left swings first: lean right
        shape = 1 - np.cosh(w * Ts * (u - 0.5)) / math.cosh(w * Ts / 2)
        com_y_des = p.sway * sgn * half * shape
        # Trunk plan, corrected until the whole-body CoM follows com_y_des (the head and
        # the swing leg move the CoM too) and sits over the feet front-to-back.
        x = v * t
        y = com_y_des.copy()
        z = p.height_m + p.bob_m * np.cos(4 * math.pi * ph)
        roll = np.zeros(n)
        pitch = math.radians(p.pitch_deg)
        foot_x = np.array([np.mean([f[s][0] for s, cc in zip(("left", "right"), c) if cc] or
                                   [np.mean([f["left"][0], f["right"][0]])])
                           for f, c in zip(feet, contact)])
        q = self.home.copy()
        for it in range(com_passes):
            # waddle: trunk roll toward the stance side (character; 0 = trunk level)
            roll = -p.waddle * np.arctan2(y, z)
            out_q, out_quat, com = [], [], []
            for warm, k in [(True, kk) for kk in range(n - 5, n)] + [(False, kk) for kk in range(n)]:
                pos = np.array([x[k], y[k], z[k]])               # warm-up frames: periodic IK
                q, quat = self._solve_frame(pos, roll[k], pitch, q, feet[k])
                if warm:
                    continue
                q = self._head(p, t[k] / T, q)
                self._set(pos, quat, q)                          # CoM includes the head motion
                mujoco.mj_comPos(self.m, self.d)
                out_q.append(q.copy()); out_quat.append(quat.copy())
                com.append(self.d.subtree_com[self.trunk].copy())
            com = np.array(com)
            y += com_y_des - com[:, 1]
            # forward: CoM moves at constant speed, centred over the feet (the trunk
            # counter-moves the head nod, as a bird does)
            com_x_des = v * t + np.mean(foot_x - v * t)
            x += com_x_des - com[:, 0]
        root_pos = np.stack([x, y, z], 1)
        return dict(t=t, joint_pos=np.array(out_q), root_pos=root_pos, root_quat=np.array(out_quat),
                    contact=contact, root_height=z.copy(), vel_xy=np.array([v, 0.0]), cycle_s=T,
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
        elif k == ord("P"):
            print(json.dumps(asdict(state["p"]), indent=1))
        elif k == ord("S"):
            np.savez(save_path, **state["g"])
            print("opgeslagen:", save_path)
            report(kin, state["g"], scene)

    m, d = kin.m, kin.d
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
        loop, k = 0, 0
        while v.is_running():
            if state["dirty"]:
                state["g"] = generate_checked(kin, state["p"]); state["dirty"] = False
                loop, k = 0, 0
            g = state["g"]
            n = len(g["t"])
            stride = g["vel_xy"][0] * g["cycle_s"]
            d.qpos[:3] = g["root_pos"][k] + np.array([loop * stride, 0, 0])
            d.qpos[3:7] = g["root_quat"][k]
            d.qpos[kin.qadr] = g["joint_pos"][k]
            mujoco.mj_forward(m, d)
            v.cam.lookat[:] = d.qpos[:3]
            v.sync()
            time.sleep(1 / FPS)
            k += 1
            if k == n:
                k, loop = 0, loop + 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    kin = DuckKinematics(a.scene)
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
