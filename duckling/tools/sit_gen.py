r"""
sit_gen.py - hurken/zitten, rondkijken en opstaan voor de Duckling (vaste beweging, geen training).

Beweging: STAAN (HOME) -> zakken -> HURKEN (rondkijken) -> opstaan -> STAAN.
Optioneel (zitten_0_1 = 1): vanuit hurken achterover tot de staart de grond raakt, en terug.
De voeten blijven plat op hun plek; alleen de romp beweegt (hoogte, kanteling, voor/achter).
Beenhoeken uit inverse kinematica op het echte model, net als gait_gen.py.

De controle speelt de beweging af in de FYSICA-simulatie met de echte servo-limieten
(60 mNm, differentieel 2x, kop 19 mNm) en meldt per fase:
  * koppel per servo als % van wat de servo kan (>=100% = haalt het niet)
  * of hij omvalt, en of hij aan het eind weer staat
  * hoeveel gewicht de romp draagt tijdens zitten (rust voor de benen)
  * zwaartepunt boven de voeten tijdens zakken/opstaan
  * botsingen tussen onderdelen (convex hulls, dus aan de voorzichtige kant)
  * gewrichtssnelheden (servo ~9 rad/s)

Vanuit robot\duckling:
    python ..\..\..\..\tools\sit_gen.py --view          # schuifjes, S = opslaan + controle
    python ..\..\..\..\tools\sit_gen.py --set crouch_z_m=0.044 up_s=2.5

Toetsen in de viewer: F = fysica aan/uit (aan = eerlijk: valt hij om?), P = parameters, S = opslaan.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gait_gen import (JOINT_ORDER, DuckKinematics, fast_model, fast_render)  # noqa: E402

FPS = 50.0
HOME_Z = 0.0592
SERVO_RAD_S = {"hip": 9.0, "knee": 9.0, "ankle": 9.0, "neck": 9.0, "head": 8.1}


@dataclass
class SitParams:
    # HURKEN: zwaartepunt blijft boven de voeten, alles langzaam en statisch
    crouch_z_m: float = 0.044       # trunk_base-hoogte gehurkt (diepst zonder botsing ~42 mm)
    crouch_pitch_deg: float = 10.0  # romp voorover gehurkt
    # ZITTEN OP DE STAART (optioneel): vanuit hurken achterover tot de staart de grond raakt
    sit: float = 0.0                # 1 = doorzakken naar zitten op de staart, 0 = alleen hurken
    sit_z_m: float = 0.036
    sit_pitch_deg: float = -50.0    # - = achterover
    foot_dx_m: float = 0.025        # voeten zoveel voor de romp in zit
    down_s: float = 1.5             # staan -> hurken
    back_s: float = 1.0             # hurken -> zitten (laatste stuk = gecontroleerd achterover vallen)
    hold_s: float = 4.0             # zitten / rondkijken
    fwd_s: float = 0.4              # zitten -> hurken (snel: zwaartepunt moet over de voeten)
    up_s: float = 2.5               # hurken -> staan (1.5 s: enkel 79%, 2.5 s: 60%)
    look_deg: float = 30.0          # rondkijken: kop-yaw amplitude
    look_s: float = 3.0
    rest_s: float = 1.0


SLIDERS = [
    ("crouch_z_m", "hurk_hoogte_mm", 38.0, 59.0, 0.001),
    ("crouch_pitch_deg", "hurk_voorover_gr", -20.0, 40.0, 1.0),
    ("sit", "zitten_0_1", 0.0, 1.0, 1.0),
    ("sit_z_m", "zit_hoogte_mm", 25.0, 50.0, 0.001),
    ("sit_pitch_deg", "zit_achterover_gr", -70.0, 0.0, 1.0),
    ("foot_dx_m", "zit_voet_voor_mm", -10.0, 40.0, 0.001),
    ("down_s", "zak_s", 0.5, 5.0, 1.0),
    ("back_s", "achterover_s", 0.2, 3.0, 1.0),
    ("fwd_s", "voorover_s", 0.15, 2.0, 1.0),
    ("up_s", "opstaan_s", 0.5, 5.0, 1.0),
    ("look_deg", "kijk_gr", 0.0, 80.0, 1.0),
    ("look_s", "kijk_s", 1.0, 8.0, 1.0),
]


def min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def _sliders_spec(scene):
    spec = mujoco.MjSpec.from_file(scene)
    for _, label, lo, hi, _ in SLIDERS:
        b = spec.worldbody.add_body(name="_body_" + label, pos=[0, 0, -1])
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0,
                   rgba=[0, 0, 0, 0], mass=1e-6)
        b.add_joint(name="_" + label, type=mujoco.mjtJoint.mjJNT_SLIDE, range=[lo, hi])
        a = spec.add_actuator(name=label, target="_" + label, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.ctrlrange = [lo, hi]
        a.ctrllimited = 1
    return spec


def physics_spec(scene, sliders=False):
    """Fysica-model: romp, kop en benen krijgen botsing met de vloer (convex hulls van de
    zichtbare meshes), zodat de romp echt op de grond kan steunen."""
    spec = _sliders_spec(scene) if sliders else mujoco.MjSpec.from_file(scene)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.contype == 0 and g.conaffinity == 0:
            g.contype, g.conaffinity = 8, 0        # alleen met de vloer (conaffinity 1|8)
    for g in spec.geoms:
        if g.name == "floor":
            g.conaffinity = 1 | 8
    return spec


class SitPlanner:
    def __init__(self, scene):
        self.kin = DuckKinematics(scene)
        self.J = {n: i for i, n in enumerate(JOINT_ORDER)}

    def pose(self, x, z, pitch, q, feet):
        """IK: romp op (x, 0, z) met kanteling pitch, voeten plat op hun vaste plek."""
        k = self.kin
        quat = np.zeros(4)
        mujoco.mju_euler2Quat(quat, np.array([0.0, pitch, 0.0]), "xyz")
        k._set(np.array([x, 0.0, z]), quat, q)
        err = 0.0
        for side in ("left", "right"):
            q = k.solve_leg(side, q, feet[side], np.eye(3), iters=80)
            err = max(err, float(np.linalg.norm(feet[side] - k.d.site_xpos[k.legs[side]["site"]])))
        return q, quat, err

    def generate(self, p: SitParams):
        k, J = self.kin, self.J
        feet = {s: k.legs[s]["home_site"].copy() for s in ("left", "right")}   # vaste voetplek
        r = math.radians
        HOME = (0.0, HOME_Z, 0.0)
        CROUCH = (0.0, p.crouch_z_m, r(p.crouch_pitch_deg))
        SIT = (-p.foot_dx_m, p.sit_z_m, r(p.sit_pitch_deg))
        sit = p.sit >= 0.5
        low = SIT if sit else CROUCH
        segs = [("staan", p.rest_s, HOME, HOME), ("zakken", p.down_s, HOME, CROUCH)]
        if sit:
            segs.append(("achterover", p.back_s, CROUCH, SIT))
        segs.append(("zitten", p.hold_s, low, low))
        if sit:
            segs.append(("voorover", p.fwd_s, SIT, CROUCH))
        segs += [("opstaan", p.up_s, CROUCH, HOME), ("staan", p.rest_s, HOME, HOME)]
        X, Z, A, phase, hold_t = [], [], [], [], []
        for name, dur, a0, a1 in segs:
            nn = max(1, int(round(dur * FPS)))
            for i in range(nn):
                s = float(min_jerk(i / nn)) if a0 != a1 else 0.0
                X.append(a0[0] + s * (a1[0] - a0[0])); Z.append(a0[1] + s * (a1[1] - a0[1]))
                A.append(a0[2] + s * (a1[2] - a0[2])); phase.append(name)
                hold_t.append(i / FPS if name == "zitten" else -1.0)
        n = len(X)
        t = np.arange(n) / FPS
        q = k.home.copy()
        out_q, out_pos, out_quat, errs = [], [], [], []
        for i in range(n):
            q, quat, err = self.pose(X[i], Z[i], A[i], q, feet)
            qo = q.copy()
            # kop: tijdens zitten rustig rondkijken (yaw), kop waterpas t.o.v. de wereld
            ht = hold_t[i]
            w = float(np.clip(min(ht, p.hold_s - ht) / 0.6, 0, 1)) if ht >= 0 else 0.0
            qo[J["head_yaw"]] = r(p.look_deg) * w * math.sin(2 * math.pi * ht / max(p.look_s, 0.5))
            qo[J["neck_pitch"]] = float(np.clip(-A[i] * 0.5, -0.5, 0.5))
            qo[J["head_pitch"]] = float(np.clip(A[i] * 0.5, -0.5, 0.5))   # head_pitch-as is gespiegeld
            out_q.append(qo); out_pos.append(np.array([X[i], 0.0, Z[i]])); out_quat.append(quat.copy())
            errs.append(err)
        return dict(t=t, joint_pos=np.array(out_q), root_pos=np.array(out_pos),
                    root_quat=np.array(out_quat), phase=np.array(phase), ik_err=np.array(errs),
                    joint_names=np.array(JOINT_ORDER), fps=FPS, params=json.dumps(asdict(p)))


def simulate(spec_or_model, g, scene_qadr=None, settle_s=0.5):
    """Speelt de beweging af als servo-doelen in de fysica. Geeft de gemeten data terug."""
    m = spec_or_model if isinstance(spec_or_model, mujoco.MjModel) else spec_or_model.compile()
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[m.joint(nm).id] for nm in JOINT_ORDER])
    act = np.array([m.actuator(nm).id for nm in JOINT_ORDER])
    d.qpos[:3] = g["root_pos"][0]; d.qpos[3:7] = g["root_quat"][0]; d.qpos[qadr] = g["joint_pos"][0]
    d.qpos[2] += 0.0005
    d.ctrl[act] = g["joint_pos"][0]
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
    sub = max(1, int(round(1 / FPS / m.opt.timestep)))
    n = len(g["t"])
    frc = np.zeros((n, len(JOINT_ORDER))); pitch = np.zeros(n); roll = np.zeros(n)
    rootz = np.zeros(n); qv = np.zeros((n, len(JOINT_ORDER))); trunk_load = np.zeros(n)
    for i in range(n):
        d.ctrl[act] = g["joint_pos"][i]
        f = np.zeros(len(JOINT_ORDER))
        for _ in range(sub):
            mujoco.mj_step(m, d)
            f = np.maximum(f, np.abs(d.actuator_force[act]))
        frc[i] = f
        R = d.xmat[1].reshape(3, 3)
        pitch[i] = math.atan2(-R[2, 0], R[2, 2]); roll[i] = math.atan2(R[2, 1], R[2, 2])
        rootz[i] = d.qpos[2]; qv[i] = d.qpos[qadr]
        cf = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            if 1 in (m.geom_bodyid[con.geom1], m.geom_bodyid[con.geom2]):   # romp raakt de vloer
                mujoco.mj_contactForce(m, d, c, cf)
                trunk_load[i] += abs(cf[0])
    lim = np.array([m.actuator_forcerange[a][1] for a in act])
    weight = float(m.body_subtreemass[1] * 9.81)
    return dict(force=frc, limit=lim, pitch=pitch, roll=roll, rootz=rootz, qpos=qv, model=m, data=d,
                trunk_load=trunk_load / weight)


def report(planner, g, sim):
    k, J = planner.kin, planner.J
    p = json.loads(str(g["params"]))
    ph = g["phase"]
    print("\n=== zitten / opstaan ===")
    print(f"hurken {p['crouch_z_m'] * 1000:.0f} mm / {p['crouch_pitch_deg']:+.0f} gr"
          + (f", zitten {p['sit_z_m'] * 1000:.0f} mm / {p['sit_pitch_deg']:+.0f} gr, voeten {p['foot_dx_m'] * 1000:+.0f} mm"
             if p['sit'] >= 0.5 else " (alleen hurken)"))
    ik = g["ik_err"].max() * 1000
    print(f"IK: voeten {'op hun plek' if ik < 1 else f'NIET bereikt (tot {ik:.1f} mm ernaast): gewrichtslimiet'}")
    # koppel per fase
    use = sim["force"] / sim["limit"]
    fases = [f for f in ("zakken", "achterover", "zitten", "voorover", "opstaan") if (ph == f).any()]
    print("koppel (piek % van servo):  " + "".join(f"{f:>11s}" for f in fases))
    worst = 0
    for nm in JOINT_ORDER:
        if nm.startswith("right"):
            continue
        j = JOINT_ORDER.index(nm)
        row = [100 * use[ph == fs, j].max() for fs in fases]
        worst = max(worst, max(row))
        flag = "  <-- TE ZWAAR" if max(row) >= 99 else ("  (krap)" if max(row) > 75 else "")
        print(f"  {nm.replace('left_', ''):24s}" + "".join(f"{v:10.0f}%" for v in row) + flag)
    # omvallen
    dp = np.degrees(np.abs(sim["pitch"] - np.array([_pitch(q) for q in g["root_quat"]])))
    dr = np.degrees(np.abs(sim["roll"]))
    fell = dp.max() > 15 or dr.max() > 15
    print(f"romp wijkt af van plan: kanteling max {dp.max():.0f} gr, zijwaarts max {dr.max():.0f} gr"
          f"  -> {'VALT OM' if fell else 'blijft staan'}")
    hold = sim["trunk_load"][ph == "zitten"]
    if len(hold):
        sh = float(np.median(hold))
        print(f"zitten: romp draagt {100 * sh:.0f}% van het gewicht"
              + ("  (rust: benen kunnen slap)" if sh > 0.6 else
                 "  (romp raakt de grond niet of nauwelijks: benen dragen alles)" if sh < 0.1 else ""))
    end = g['root_pos'][-1][2] - sim['rootz'][-1]
    print(f"na opstaan: romp {sim['rootz'][-1] * 1000:.0f} mm (plan {g['root_pos'][-1][2] * 1000:.0f})"
          f"  -> {'STAAT WEER' if abs(end) < 0.004 and not fell else 'NIET OPGESTAAN'}")
    # statische balans: CoM boven de voeten (zolang de romp de grond niet raakt)
    m, d = k.m, k.d
    worst_out = 0.0
    for i in range(0, len(g["t"]), 2):
        k._set(g["root_pos"][i], g["root_quat"][i], g["joint_pos"][i])
        mujoco.mj_comPos(m, d)
        cx = d.subtree_com[k.trunk][0]
        xs = []
        for s in ("left", "right"):
            gid = k.legs[s]["sole"]
            mid = m.geom_dataid[gid]; va = m.mesh_vertadr[mid]; nv = m.mesh_vertnum[mid]
            w = m.mesh_vert[va:va + nv] @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
            xs += [w[:, 0].min(), w[:, 0].max()]
        lo, hi = min(xs), max(xs)
        if ph[i] in ("zakken", "opstaan"):   # achterover/voorover: staart steunt of hij zwaait over
            worst_out = max(worst_out, lo - cx, cx - hi)
    print(f"zwaartepunt t.o.v. voeten tijdens zakken/opstaan: "
          f"{'binnen' if worst_out <= 0 else f'tot {worst_out * 1000:.1f} mm ERBUITEN (romp/staart moet steunen)'}")
    # snelheden
    qd = np.abs(np.diff(g["joint_pos"], axis=0)) * FPS
    over = []
    for j, nm in enumerate(JOINT_ORDER):
        lim = SERVO_RAD_S[nm.split("_")[1] if nm.startswith(("left", "right")) else nm.split("_")[0]]
        if qd[:, j].max() > lim:
            over.append(f"{nm} {qd[:, j].max():.1f}")
    print("snelheden: " + ("binnen servo-snelheid" if not over else "TE SNEL: " + ", ".join(over)))
    # botsingen
    pen = self_collisions(k, g)
    print("botsingen tussen onderdelen: " + ("geen" if not pen else ", ".join(pen[:4])))
    print("conclusie: " + ("KAN HET" if worst < 99 and not fell and ik < 1 and not pen and abs(end) < 0.004
                           else "nog niet - zie hierboven"))


def _pitch(quat):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, quat); R = R.reshape(3, 3)
    return math.atan2(-R[2, 0], R[2, 2])


_HULL = {}


def self_collisions(kin, g):
    """Convex hulls van alle meshes; paren die al in HOME overlappen (ontwerp) tellen niet."""
    if "m" not in _HULL:
        spec = mujoco.MjSpec.from_file(_HULL_SCENE[0])
        for gg in spec.geoms:
            if gg.type == mujoco.mjtGeom.mjGEOM_MESH:
                gg.contype, gg.conaffinity = 1, 1
        mh = spec.compile(); dh = mujoco.MjData(mh)
        dh.qpos[2] = HOME_Z; dh.qpos[3] = 1; mujoco.mj_forward(mh, dh)
        base = {frozenset((c.geom1, c.geom2)) for c in dh.contact[:dh.ncon]}
        _HULL.update(m=mh, d=dh, base=base, floor=mh.geom("floor").id,
                     qadr=np.array([mh.jnt_qposadr[mh.joint(n).id] for n in JOINT_ORDER]))
    mh, dh = _HULL["m"], _HULL["d"]
    found = {}
    for i in range(0, len(g["t"]), 3):
        dh.qpos[:] = 0
        dh.qpos[:3] = g["root_pos"][i]; dh.qpos[3:7] = g["root_quat"][i]
        dh.qpos[_HULL["qadr"]] = g["joint_pos"][i]
        mujoco.mj_forward(mh, dh)
        for c in dh.contact[:dh.ncon]:
            pr = frozenset((c.geom1, c.geom2))
            if _HULL["floor"] in pr or pr in _HULL["base"] or c.dist > -0.0005:
                continue
            key = f"{mh.geom(c.geom1).name}/{mh.geom(c.geom2).name}"
            found[key] = min(found.get(key, 0), c.dist)
    return [f"{k} ({v * 1000:.1f} mm)" for k, v in sorted(found.items(), key=lambda kv: kv[1])]


_HULL_SCENE = ["scene_walk.xml"]


def view(planner, p, scene, save_path):
    import mujoco.viewer
    spec = physics_spec(scene, sliders=True)
    m = spec.compile(); d = mujoco.MjData(m)
    sim_m = physics_spec(scene).compile()
    qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in JOINT_ORDER])
    act = np.array([m.actuator(n).id for n in JOINT_ORDER])
    sl = {name: (m.actuator(label).id, sc) for name, label, lo, hi, sc in SLIDERS}
    st = dict(p=p, g=planner.generate(p), phys=False, dirty=False)

    def to_sliders():
        for name, (aid, sc) in sl.items():
            d.ctrl[aid] = getattr(st["p"], name) / sc

    def key(kc):
        if kc == ord("F"):
            st["phys"] = not st["phys"]; st["dirty"] = True
            print("fysica:", "aan (servo's volgen, kan omvallen)" if st["phys"] else "uit (alleen het plan)")
        elif kc == ord("P"):
            print(json.dumps(asdict(st["p"]), indent=1))
        elif kc == ord("S"):
            np.savez(save_path, **st["g"])
            with open(os.path.splitext(save_path)[0] + ".json", "w") as f:
                json.dump(asdict(st["p"]), f, indent=1)
            print("opgeslagen:", save_path, "(+ .json)")
            report(planner, st["g"], simulate(sim_m, st["g"]))

    to_sliders()
    fast_model(m)
    last = {"vals": None, "t": 0.0, "pending": False}
    with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
        fast_render(v)
        v.cam.distance = 0.35; v.cam.elevation = -10; v.cam.azimuth = 120
        i = 0
        while v.is_running():
            vals = tuple(round(float(d.ctrl[aid]), 4) for aid, _ in sl.values())
            if last["vals"] is None:
                last["vals"] = vals
            elif vals != last["vals"]:
                last.update(vals=vals, t=time.time(), pending=True)
            elif last["pending"] and time.time() - last["t"] > 0.3:
                for name, (aid, sc) in sl.items():
                    setattr(st["p"], name, round(float(d.ctrl[aid]) * sc, 5))
                last["pending"] = False; st["dirty"] = True
            if st["dirty"]:
                st["g"] = planner.generate(st["p"]); st["dirty"] = False; i = 0
                to_sliders()
            g = st["g"]; n = len(g["t"])
            if i == 0 or not st["phys"]:
                d.qpos[:3] = g["root_pos"][i]; d.qpos[3:7] = g["root_quat"][i]; d.qpos[qadr] = g["joint_pos"][i]
                d.qvel[:] = 0
            if st["phys"]:
                d.ctrl[act] = g["joint_pos"][i]
                t_end = d.time + 1 / FPS
                while d.time < t_end:
                    mujoco.mj_step(m, d)
            else:
                mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(1 / FPS)
            i = (i + 1) % n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="scene_walk.xml")
    ap.add_argument("--params", help="JSON met parameters")
    ap.add_argument("--set", nargs="*", default=[], help="naam=waarde, bv. sit_z_m=0.03")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--save", default=None)
    ap.add_argument("--mooi", action="store_true")
    a = ap.parse_args()
    _HULL_SCENE[0] = a.scene
    p = SitParams()
    if a.params:
        for kk, vv in json.load(open(a.params)).items():
            setattr(p, kk, float(vv))
    for kv in a.set:
        kk, vv = kv.split("=")
        assert hasattr(p, kk), f"onbekende parameter {kk}"
        setattr(p, kk, float(vv))
    planner = SitPlanner(a.scene)
    if a.view:
        view(planner, p, a.scene, a.save or "sit_motion.npz")
        return
    g = planner.generate(p)
    if a.save:
        np.savez(a.save, **g)
        print("opgeslagen:", a.save)
    report(planner, g, simulate(physics_spec(a.scene).compile(), g))


if __name__ == "__main__":
    main()
