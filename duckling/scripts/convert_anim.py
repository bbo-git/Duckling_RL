"""Blender dump (duck_anim_dump.json) -> duckling joint trajectory (CAD-zero angles, rad).

Rig: armature in CAD frame (mm), bones knee.X = whole hip (yaw->roll->pitch),
ankle.X = enkel-pitch (walk 'knee'), toe.X = teen-pitch (walk 'ankle'),
neck = neck_pitch, head = head pitch->roll->yaw.
Output npz: t, qpos (root xyz m + quat wxyz + 14 joints in action order), per-joint residuals.
"""
import json, sys, numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

ORDER = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
         "neck_pitch", "head_pitch", "head_roll", "head_yaw",
         "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
TORSO_ORIGIN_MM = np.array([0, 0, 60.0])

def load(path):
    d = json.load(open(path))
    rest = {k.split("/")[1]: np.array(v["matrix_local"]) for k, v in d["bones"].items()}
    frames = []
    for f in d["frames"]:
        A = np.array(f["objects"]["Armature"])
        D = {n: np.array(f["pose"]["Armature/" + n]["matrix"]) @ np.linalg.inv(rest[n]) for n in rest}
        frames.append((A, D))
    return d["fps"], frames

def angles(A, D):
    R = {n: m[:3, :3] for n, m in D.items()}
    q, res = {}, {}
    for s, side, sg in (("L", "left", 1), ("R", "right", -1)):
        e = Rot.from_matrix(R["root"].T @ R["knee." + s]).as_euler("ZXY")
        q[side + "_hip_yaw"], q[side + "_hip_roll"], q[side + "_hip_pitch"] = sg * e[0], sg * e[1], e[2]
        ra = Rot.from_matrix(R["knee." + s].T @ R["ankle." + s]).as_rotvec()
        rt = Rot.from_matrix(R["ankle." + s].T @ R["toe." + s]).as_rotvec()
        q[side + "_knee"], q[side + "_ankle"] = ra[1], -rt[1]
        res[side + "_knee"] = np.hypot(ra[0], ra[2]); res[side + "_ankle"] = np.hypot(rt[0], rt[2])
    rn = Rot.from_matrix(R["root"].T @ R["neck"]).as_rotvec()
    q["neck_pitch"] = rn[1]
    Rh = Rot.from_rotvec([0, rn[1], 0]).as_matrix().T @ (R["root"].T @ R["head"])
    e = Rot.from_matrix(Rh).as_euler("YXZ")          # head_pitch(+y) -> head_roll(-x) -> head_yaw(+z)
    q["head_pitch"], q["head_roll"], q["head_yaw"] = e[0], -e[1], e[2]
    res["neck_pitch"] = np.hypot(rn[0], rn[2])
    T = A @ D["root"]
    pos = (T @ np.r_[TORSO_ORIGIN_MM - A[:3, 3] + 0, 1])[:3] / 1000.0  # trunk origin, m
    # trunk origin in armature space = CAD point minus armature offset
    quat = Rot.from_matrix(A[:3, :3] @ R["root"]).as_quat()            # xyzw
    return np.array([q[n] for n in ORDER]), pos, quat, res

def main(path, out="walk_cycle.npz", fps_out=50.0):
    fps, frames = load(path)
    J, P, Q, RES = [], [], [], []
    for A, D in frames:
        j, p, qq, r = angles(A, D); J.append(j); P.append(p); Q.append(qq); RES.append(r)
    J, P, Q = np.array(J), np.array(P), np.array(Q)
    n = len(J); T = n / fps                                   # cycle length (frame n+1 == frame 1)
    # forward progression per cycle from a linear fit of the root path
    t_in = np.arange(n) / fps
    vel = np.array([np.polyfit(t_in, P[:, k], 1)[0] for k in range(2)])
    stride = vel * T
    Pc = P.copy(); Pc[:, :2] -= np.outer(t_in, vel)          # periodic part
    # periodic interpolation to fps_out
    t_out = np.arange(int(round(T * fps_out))) / fps_out
    def interp_per(Y):
        Yp = np.vstack([Y, Y[:1]]); tp = np.r_[t_in, T]
        return np.stack([np.interp(t_out, tp, Yp[:, k]) for k in range(Y.shape[1])], 1)
    Jo = interp_per(np.unwrap(J, axis=0))
    Po = interp_per(Pc); Po[:, :2] += np.outer(t_out, vel)
    Qp = np.vstack([Q, Q[:1]]); sl = Slerp(np.r_[t_in, T], Rot.from_quat(Qp))
    Qo = sl(t_out).as_quat()[:, [3, 0, 1, 2]]                # wxyz
    res = {k: max(r[k] for r in RES) for k in RES[0]}
    np.savez(out, t=t_out, joint_pos=Jo, root_pos=Po, root_quat=Qo, joint_names=np.array(ORDER),
             cycle_s=T, vel_xy=vel, fps=fps_out)
    print(f"cyclus {T:.3f} s, {len(t_out)} stappen @ {fps_out:g} Hz, snelheid x {vel[0]:.3f} y {vel[1]:.3f} m/s, staplengte {stride[0]*1000:.0f} mm")
    print("bereik per gewricht (graden):")
    for i, nme in enumerate(ORDER):
        print(f"  {nme:16s} {np.degrees(Jo[:, i].min()):7.1f} .. {np.degrees(Jo[:, i].max()):6.1f}"
              + (f"   (afwijking van de as max {np.degrees(res[nme]):.1f})" if nme in res else ""))
    print(f"romphoogte {Po[:, 2].min()*1000:.1f} .. {Po[:, 2].max()*1000:.1f} mm (t.o.v. CAD-vloer)")
    return out



def add_contacts(npz_path, scene_xml, lift_mm=1.5):
    """Add the reference foot-contact schedule (left, right) to walk_cycle.npz.
    A foot counts as 'in contact' while its lowest point is within lift_mm of the floor."""
    import mujoco
    a = dict(np.load(npz_path))
    m = mujoco.MjModel.from_xml_path(scene_xml); d = mujoco.MjData(m)
    adr = [m.jnt_qposadr[m.joint(str(n)).id] for n in a["joint_names"]]
    z = np.zeros((len(a["t"]), 2))
    low = np.zeros(len(a["t"]))                     # lowest sole vertex per frame
    soles = [m.geom(f"{s}_foot_collision").id for s in ("left", "right")]
    for k in range(len(a["t"])):
        d.qpos[:3] = a["root_pos"][k]; d.qpos[3:7] = a["root_quat"][k]; d.qpos[adr] = a["joint_pos"][k]
        mujoco.mj_kinematics(m, d)
        z[k] = [d.site("left_foot").xpos[2], d.site("right_foot").xpos[2]]
        lows = []
        for g in soles:
            mid = m.geom_dataid[g]; va = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
            w = m.mesh_vert[va:va + nv] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
            lows.append(w[:, 2].min())
        low[k] = min(lows)
        z[k] = lows
    # A foot is in contact when its lowest point is within lift_mm of this frame's floor.
    a["contact"] = (z - low[:, None]) * 1000 < lift_mm
    # Trunk height above the floor: the animation's floor is not exactly z = 0, so take the
    # lowest sole point of every frame as the floor (the stance foot stands on it).
    a["root_height"] = a["root_pos"][:, 2] - low
    print(f"romphoogte boven de vloer {a['root_height'].min()*1000:.1f} .. {a['root_height'].max()*1000:.1f} mm")
    np.savez(npz_path, **a)
    for i, s in enumerate(("links ", "rechts")):
        c = a["contact"][:, i]
        print(f"{s} {''.join('#' if x else '.' for x in c)}  lucht {(~c).sum() / float(a['fps']):.2f} s")


def feasibility(npz_path, scene_xml, noload=None):
    """Print two physical checks of the cycle:
    1. static balance: is the centre of mass above the feet that touch the floor?
       (at a slow cycle it must be; a fast cycle can get away with being outside briefly)
    2. joint speed versus the motors' no-load speed."""
    import mujoco
    from scipy.spatial import ConvexHull, Delaunay
    noload = noload or {"hip": 9.0, "knee": 9.0, "ankle": 9.0, "neck": 9.0, "head": 8.1}
    a = np.load(npz_path)
    m = mujoco.MjModel.from_xml_path(scene_xml); d = mujoco.MjData(m)
    names = [str(n) for n in a["joint_names"]]
    adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
    soles = [m.geom(f"{s}_foot_collision").id for s in ("left", "right")]
    trunk = m.body("trunk_base").id
    margins, single, tilts = [], [], []
    polys, coms = [], []
    for k in range(len(a["t"])):
        d.qpos[:3] = a["root_pos"][k]; d.qpos[3:7] = a["root_quat"][k]; d.qpos[adr] = a["joint_pos"][k]
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        com = d.subtree_com[trunk][:2]
        pts = []
        for i, g in enumerate(soles):
            if a["contact"][k, i]:
                mid = m.geom_dataid[g]; va = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
                w = m.mesh_vert[va:va + nv] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                pts.append(w[w[:, 2] < w[:, 2].min() + 0.0015][:, :2])
        if not pts:
            continue
        P = np.vstack(pts)
        polys.append((k, P)); coms.append((k, d.subtree_com[trunk].copy()))
        try:
            eq = ConvexHull(P).equations
            margins.append(-(eq[:, :2] @ com + eq[:, 2]).max())   # >0: inside
        except Exception:                                          # foot on an edge: a line
            margins.append(-np.min(np.linalg.norm(P - com, axis=1)))
        for i, g in enumerate(soles):                              # sole tilt while standing on it
            if a["contact"][k, i]:
                tilts.append(np.degrees(np.arccos(min(1.0, d.xmat[m.geom_bodyid[g]].reshape(3, 3)[2, 2]))))
        single.append(a["contact"][k].sum() == 1)
    margins = np.array(margins) * 1000; single = np.array(single)
    fps = float(a["fps"])
    out = margins < 0
    print("\n== balans")
    print(f"  zwaartepunt buiten het steunvlak: {out.mean()*100:.0f}% van de cyclus, "
          f"langste aaneengesloten stuk {max((len(s) for s in ''.join('x' if o else ' ' for o in out).split()), default=0) / fps:.2f} s, "
          f"slechtste {-margins.min():.0f} mm ernaast")
    print("  (statisch: alleen een maat voor langzaam lopen; bij lopen telt de ZMP-regel hieronder)")
    # Dynamic balance: zero-moment point (inverted pendulum) instead of the CoM itself.
    # A walking robot may have its CoM outside the foot as long as the ZMP stays inside.
    allc = np.zeros((len(a["t"]), 3))
    for k in range(len(a["t"])):
        d.qpos[:3] = a["root_pos"][k]; d.qpos[3:7] = a["root_quat"][k]; d.qpos[adr] = a["joint_pos"][k]
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        allc[k] = d.subtree_com[trunk]
    n = len(allc); dt = 1.0 / fps
    trend = np.outer(np.arange(n) * dt, a["vel_xy"])
    per = allc[:, :2] - trend
    acc = (np.roll(per, -1, 0) - 2 * per + np.roll(per, 1, 0)) / dt**2
    zc = allc[:, 2].mean()
    zmp = allc[:, :2] - zc / 9.81 * acc
    zm = []
    for k, P in polys:
        try:
            eq = ConvexHull(P).equations
            zm.append(-(eq[:, :2] @ zmp[k] + eq[:, 2]).max())
        except Exception:
            zm.append(-np.min(np.linalg.norm(P - zmp[k], axis=1)))
    zm = np.array(zm) * 1000
    frac = np.mean(zm < -3)
    verdict = "goed" if frac < 0.15 else ("matig, training moet veel corrigeren" if frac < 0.35 else "NIET in balans")
    print(f"  dynamisch (ZMP, wat telt bij lopen): >3 mm buiten de voet {frac*100:.0f}% van de cyclus, "
          f"slechtste {max(0, -zm.min()):.0f} mm  -> {verdict}")
    if tilts:
        print(f"  zool-kanteling bij steun: gemiddeld {np.mean(tilts):.0f} graden, max {np.max(tilts):.0f}"
              " (geen enkel-roll: alles boven ~5 graden = staan op de rand van de voet)")
    print("\n== snelheid gewrichten t.o.v. de motoren (onbelast)")
    q = a["joint_pos"]; v = np.abs(np.diff(np.vstack([q, q[:1]]), axis=0)) * fps
    for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle", "neck_pitch", "head_pitch", "head_roll", "head_yaw"):
        lim = noload.get(j.split("_")[0], noload["knee"])
        cols = [names.index(f"{s}_{j}") for s in ("left", "right") if f"{s}_{j}" in names] or [names.index(j)]
        top = max(v[:, c].max() for c in cols)
        flag = "  TE SNEL" if top > 0.8 * lim else ""
        print(f"  {j:10s} max {top:5.1f} rad/s   (motor {lim:.1f}){flag}")


if __name__ == "__main__":
    # python convert_anim.py duck_anim_dump.json walk_cycle.npz [scene_walk.xml]
    args = sys.argv[1:]
    out = main(args[0], args[1] if len(args) > 1 else "walk_cycle.npz")
    scene = args[2] if len(args) > 2 else "scene_walk.xml"
    import os
    if os.path.exists(scene):
        add_contacts(out, scene)
        feasibility(out, scene)
    else:
        print(f"let op: {scene} niet gevonden, contactschema niet toegevoegd")
