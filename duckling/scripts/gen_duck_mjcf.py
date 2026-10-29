"""
gen_duck_mjcf.py - bouwt duck.xml (MuJoCo) uit de Fusion-export.

Gebruik:
    python gen_duck_mjcf.py duck_joints.csv duck_links.csv [uitvoer.xml]          -> duck.xml
    python gen_duck_mjcf.py duck_joints.csv duck_links.csv --walk [robot_walk.xml] -> model voor mjlab

Verwacht naast de XML een map meshes/ met per geprint onderdeel een STL,
genoemd naar de body in Fusion (bijv. meshes/onderbeen_links.stl),
geexporteerd in mm, in het globale assenstelsel, in de nulstand.

Aannames (pas aan in de blokken hieronder):
  * Fusion rekende met staal (7,85 g/cm3); geprinte delen worden omgerekend naar
    hars (1,15) of TPU (1,21), tenzij de kolom echte_massa_g in duck_links.csv is ingevuld.
  * Gekochte onderdelen (motoren, servo's, magneten, ...) zijn kleine primitieven
    met hun eigen massa; daar is geen STL voor nodig.
"""

import csv
import math
import sys
from pathlib import Path

# ------------------------------------------------------------------ instellingen
STEEL = 7.85
RESIN = 1.15
TPU = 1.21
TORSO_ORIGIN_MM = (0.0, 0.0, 60.0)  # oorsprong van de romp-body (ongeveer heuphoogte)
SPAWN_CLEARANCE_MM = 2.0
# Traagheid uit de mesh: "convex" (convex hull, werkt ook bij open of rommelige meshes),
# "exact" (vereist een gesloten, correct georienteerde mesh) of "shell" (massa op het oppervlak).
MESH_INERTIA = "convex"
# Holle schaaldelen: massa op het oppervlak (convex hull zou ze massief vullen).
SHELL_INERTIA_PARTS = {"head_shell", "body"}

# Massa's van gekochte onderdelen (g). Overschreven door echte_massa_g als die is ingevuld.
BOUGHT_MASS_G = {
    "drv1": 1.5, "drv2": 1.5, "drv3": 1.5,     # DRV8835-moduletjes zonder headers (schatting)
    "batterij": 8.5,                           # Tattu 1S 300 mAh (schatting ~8-9 g)
    "ESP3-CAM:1": 10.0,                        # ESP32-CAM met camera (schatting)
}
EXTRA_TORSO_PARTS = [  # (naam, massa_g, positie_mm) – nog niet in CAD
    ("controller_board", 5.0, (0.0, 0.0, 85.0)),   # Core2350B (schatting)
    ("imu_kabels", 6.0, (0.0, 0.0, 75.0)),         # IMU + bedrading (schatting)
]

# Welke onderdelen bewegen met welk gewricht. "torso" = romp.
# Onderdelen die hier niet staan worden genegeerd (met een waarschuwing).
LINKS = {
    "torso": ["body", "body_armature", "batterij", "servo_yaw_rechts", "servo_yaw_links", "servo_neck"],
    # ---- linkerbeen
    "L_hip_yaw": ["beugel_pitch_links", "servo_differentieelA_links", "servo_differentieelB_links",
                  "gear_motor1_links", "gear_motor2_links"],
    "L_hip_roll": ["blok_differentieel_links"],
    "L_hip_pitch": ["onderbeen_links", "magneet_pith_houder_links"],
    "L_ankle": ["enkel_links", "servo_enkel_links"],
    "L_toe": ["teen_links", "zool_links", "servo_teen_links"],
    # ---- rechterbeen
    "R_hip_yaw": ["beugel_pitch_rechts", "servo_differentieelA_rechts", "servo_differentieelB_rechts",
                  "gear_motor1_rechts", "gear_motor2_rechts"],
    "R_hip_roll": ["blok_differentieel_rechts"],
    "R_hip_pitch": ["onderbeen_rechts", "magneet_pith_houder_rechts"],
    "R_ankle": ["enkel_rechts", "servo_enkel_rechts"],
    "R_toe": ["teen_rechts", "zool_rechts", "servo_teen_rechts"],
    # ---- nek en kop
    "neck": ["neck"],
    "head_pitch": ["head_shell", "servo_jaw", "ESP3-CAM:1"],
    "head_roll": ["servo_pitch_head"],
    "head": ["beugel_pitch_head", "servo_differentieelA_head", "servo_differentieelB_head", "blok_differentieel_head", "gear_motor1_head", "gear_motor2_head"],
    "jaw": ["jaw"],
}

# Kinematische boom: (link, ouder, joint-naam uit duck_joints.csv)
TREE = [
    ("L_hip_yaw", "torso", "left_hip_yaw"),
    ("L_hip_roll", "L_hip_yaw", "left_hip_roll"),
    ("L_hip_pitch", "L_hip_roll", "left_hip_pitch"),
    ("L_ankle", "L_hip_pitch", "left_ankle_pitch"),
    ("L_toe", "L_ankle", "left_toe_pitch"),
    ("R_hip_yaw", "torso", "right_hip_yaw"),
    ("R_hip_roll", "R_hip_yaw", "right_hip_roll"),
    ("R_hip_pitch", "R_hip_roll", "right_hip_pitch"),
    ("R_ankle", "R_hip_pitch", "right_ankle_pitch"),
    ("R_toe", "R_ankle", "right_toe_pitch"),
    ("neck", "torso", "neck_pitch"),
    ("head_pitch", "neck", "head_pitch"),
    ("head_roll", "head_pitch", "head_roll"),
    ("head", "head_roll", "head_yaw"),
    ("jaw", "head", "jaw_pitch"),
]

# Bereik per gewricht (graden). Eerste schatting – stel bij na de motion study.
RANGE_DEG = {
    "left_hip_yaw": (-20, 20), "right_hip_yaw": (-20, 20),
    "left_hip_roll": (-20, 20), "right_hip_roll": (-20, 20),
    "left_hip_pitch": (-60, 60), "right_hip_pitch": (-60, 60),
    "left_ankle_pitch": (-50, 90), "right_ankle_pitch": (-50, 90),
    "left_toe_pitch": (-60, 90), "right_toe_pitch": (-60, 90),
    "neck_pitch": (-30, 30), "head_pitch": (-30, 30),
    "head_roll": (-20, 20), "head_yaw": (-90, 90), "jaw_pitch": (0, 30),
}

# Actuatoren. Differentielen worden aangestuurd via twee motoren op fixed tendons:
#   motor A = uitgang + drager, motor B = uitgang - drager  (miter 1:1)
DIFFERENTIALS = [
    # (naam, joint van het uitgaande wiel, joint van de drager)
    ("L_hip", "left_hip_pitch", "left_hip_roll"),
    ("R_hip", "right_hip_pitch", "right_hip_roll"),
    ("head", "head_yaw", "head_roll"),
]
CORELESS_DIRECT = []   # 2026-10-28: geen coreless-motoren meer, alles servo
SERVOS = ["left_hip_yaw", "right_hip_yaw",
          "left_ankle_pitch", "right_ankle_pitch", "left_toe_pitch", "right_toe_pitch",
          "neck_pitch", "head_pitch", "jaw_pitch"]
# Kleine servo (EMAX ES9251 II) voor alle kopgewrichten behalve de nek; de rest ES9051.
SMALL_SERVO_JOINTS = {"head_pitch", "head_roll", "head_yaw", "jaw_pitch"}


def actuator_class(jname):
    return "servo_s" if jname in SMALL_SERVO_JOINTS else "servo"

# Ruwe actuatorparameters (SI). TODO: vervangen door meetwaarden van de testbank.
CORELESS = dict(kp=0.6, kv=0.006, force=0.05, armature=1.0e-5, damping=2e-4, frictionloss=2e-3)
# Walk-model: differentieelgewrichten krijgen samen het koppel van twee motoren
# (tau_pitch = tauA + tauB, tau_roll = tauA - tauB).
DIFF_JOINT_FORCE_FACTOR = 2.0
# Servokoppel bij 4,8 V uit de specs, geschaald naar een 1S-accu (3,7 V nominaal: x0,77).
SUPPLY_FACTOR = 3.7 / 4.8
KGFCM = 0.0980665          # 1 kgf.cm in N.m
# EMAX ES9051: 0,8 kgf.cm @ 4,8 V, 4,1 g
SERVO = dict(kp=2.0, kv=0.02, force=round(0.8 * KGFCM * SUPPLY_FACTOR, 4),
             armature=2.0e-5, damping=5e-4, frictionloss=3e-3)
# EMAX ES9251 II: 0,25 kgf.cm @ 4,8 V, 2,5 g (kop)
SERVO_S = dict(kp=0.6, kv=0.006, force=round(0.25 * KGFCM * SUPPLY_FACTOR, 4),
               armature=8.0e-6, damping=2e-4, frictionloss=1.5e-3)
SERVO_MASS_G = {"servo": 4.1, "servo_s": 2.5}
# Onbelaste uitgaande toeren (rad/s). Een DC-motor aan een verzadigde H-brug levert
# tau = tau_stall * (u - w / w0), |u| <= 1: forcerange = stall-koppel en de tegen-EMK
# werkt als extra gewrichtsdemping tau_stall / w0. Boven w0 kan het gewricht dus niet.
NOLOAD_RAD_S = {
    "coreless": 300 * 2 * math.pi / 60,   # 300 rpm-versie (150 rpm: 15.7)
    "servo": math.radians(60) / 0.09 * SUPPLY_FACTOR,    # ES9051 0,09 s/60 gr @4,8 V, op 1S
    "servo_s": math.radians(60) / 0.10 * SUPPLY_FACTOR,  # ES9251 II: snelheid nog niet bekend, aanname
}
CLASS_PARAMS = {"servo": SERVO, "servo_s": SERVO_S, "coreless": CORELESS}


def joint_damping(cls, diff=False):
    base = CLASS_PARAMS[cls]
    force = base["force"]
    if diff:
        force *= DIFF_JOINT_FORCE_FACTOR
    return base["damping"] + force / NOLOAD_RAD_S[cls]

# Botsingen: alleen deze meshes botsen (convex hull). Voeten met de vloer, romp en kop bij vallen.
FOOT_COLLIDERS = ["zool_links", "zool_rechts", "teen_links", "teen_rechts"]
BODY_COLLIDERS = ["body", "head_shell", "jaw"]
# Botsingen gebruiken de convex hull van de mesh. Met een maximum aantal hoekpunten
# wordt die hull grof en dus snel; het zichtbare model blijft op volle resolutie.
COLLISION_HULL_VERTS = 32

# ---- Walk-model voor mjlab / microduck_rl (python gen_duck_mjcf.py ... --walk)
# Namen volgen de Microduck-conventie, zodat hun rewards en observaties werken.
WALK_TORSO_NAME = "trunk_base"
WALK_JOINT_NAMES = {
    "left_ankle_pitch": "left_knee", "right_ankle_pitch": "right_knee",
    "left_toe_pitch": "left_ankle", "right_toe_pitch": "right_ankle",
    "jaw_pitch": "passive_jaw",          # passief: niet in de policy
}
WALK_FOOT_SOLES = {"zool_links": "left", "zool_rechts": "right"}
# Links-rechts beenbotsing: deze delen botsen alleen met het ANDERE been (niet met de
# vloer, niet met het eigen been). Bitmasker: vloer 1, linkerbeen 2, rechterbeen 4.
WALK_LEG_COLLIDERS = {
    "onderbeen_links": ("left", "shin"), "enkel_links": ("left", "ankle"), "teen_links": ("left", "toe"),
    "onderbeen_rechts": ("right", "shin"), "enkel_rechts": ("right", "ankle"), "teen_rechts": ("right", "toe"),
}
WALK_COLLISION_BITS = {  # (contype, conaffinity)
    "left_foot": (3, 5), "right_foot": (5, 3),   # vloer + ander been
    "left": (2, 4), "right": (4, 2),             # alleen ander been
}
# Volgorde van de takken onder de romp (bepaalt de volgorde van gewrichten/acties):
WALK_BRANCH_ORDER = ["L_hip_yaw", "neck", "R_hip_yaw"]
IMU_POS_MM = (0.0, 0.0, 80.0)        # TODO: echte plek van de IMU op het board

PRINTED_PREFIXES = ("body", "beugel", "blok", "plaat", "gear_", "onderbeen", "enkel",
                    "teen", "zool", "neck", "head_", "jaw")
TINY_MESH_AS_SPHERE = ("gear_",)
BOUGHT_KEYWORDS = ("motor_", "servo", "magneet", "encoder", "lager", "drv", "batterij")  # kleine tandwielen als bolletje, geen STL nodig


# ------------------------------------------------------------------ inlezen
def is_printed(name):
    if name.startswith("gear_"):
        return True
    if any(k in name for k in BOUGHT_KEYWORDS):
        return False
    return name.startswith(PRINTED_PREFIXES)


def read_joints(path):
    joints = {}
    with open(path, encoding="utf-8-sig") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    for line in lines[1:]:
        p = [x.strip() for x in line.split(",")]
        name = p[0]
        x, y, z, ax, ay, az = (float(v) for v in p[-9:-3])
        n = math.sqrt(ax * ax + ay * ay + az * az)
        joints[name] = ((x, y, z), (ax / n, ay / n, az / n))
    return joints


def read_parts(path):
    parts = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            name = r["onderdeel_pad"].split("/")[-1].strip()
            m_fusion = float(r["massa_fusion_g"] or 0)
            com = tuple(float(r[k] or 0) for k in ("com_x_mm", "com_y_mm", "com_z_mm"))
            real = (r.get("echte_massa_g") or "").strip()
            if real:
                mass, src = float(real), "gewogen"
            elif name in BOUGHT_MASS_G:
                mass, src = BOUGHT_MASS_G[name], "schatting"
            elif name.startswith(("servo", "serva")):
                small = name.endswith("_head") or name == "servo_jaw"
                mass, src = SERVO_MASS_G["servo_s" if small else "servo"], "specs"
            elif is_printed(name):
                dens = TPU if name.startswith("zool") else RESIN
                mass, src = m_fusion * dens / STEEL, "omgerekend"
            else:
                mass, src = m_fusion, "fusion"
            parts[name] = dict(mass=mass, com=com, src=src,
                               printed=is_printed(name))
    return parts


# ------------------------------------------------------------------ helpers
def mm(v):
    return " ".join(f"{c / 1000.0:.6g}" for c in v)


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def f(v):
    return " ".join(f"{c:.6g}" for c in v)


# ------------------------------------------------------------------ bouwen
UNITS = {"mm": 0.001, "cm": 0.01, "m": 1.0, "inch": 0.0254}


def stl_mean(path):
    """Gemiddelde vertex van een binaire of ASCII STL, in de eenheid van het bestand."""
    import struct
    data = path.read_bytes()
    pts = []
    n = int.from_bytes(data[80:84], "little") if len(data) >= 84 else 0
    if len(data) >= 84 and 84 + 50 * n == len(data):
        for k in range(n):
            o = 84 + 50 * k + 12
            pts.extend(struct.unpack_from("<9f", data, o)[i:i + 3] for i in (0, 3, 6))
    else:
        t = data.decode(errors="ignore").split()
        pts = [tuple(float(v) for v in t[i + 1:i + 4]) for i, w in enumerate(t) if w == "vertex"]
    if not pts:
        return None
    return tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))


def detect_scale(path, com_mm):
    """Kiest de eenheid waarvoor het midden van de STL het best bij het zwaartepunt uit Fusion past."""
    if not path.exists():
        return "mm", UNITS["mm"]
    mean = stl_mean(path)
    if mean is None:
        return "mm", UNITS["mm"]
    com_m = [c / 1000.0 for c in com_mm]
    best = min(UNITS.items(),
               key=lambda kv: sum((m * kv[1] - c) ** 2 for m, c in zip(mean, com_m)))
    return best


def walk_scene(robot_file):
    return f"""<mujoco model="duckling scene">
  <include file="{robot_file}"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.26 0.28" rgb2="0.32 0.33 0.35" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="40 40"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 2 0.05" material="grid"/>
  </worldbody>
</mujoco>
"""


def build(joints, parts, meshdir=Path("meshes"), walk=False):
    warn = []
    jn = (lambda j: WALK_JOINT_NAMES.get(j, j)) if walk else (lambda j: j)
    torso_name = WALK_TORSO_NAME if walk else "torso"
    origin = {"torso": TORSO_ORIGIN_MM}
    for link, parent, jname in TREE:
        if jname not in joints:
            raise SystemExit(f"Joint {jname} ontbreekt in duck_joints.csv")
        origin[link] = joints[jname][0]

    used = {p for lst in LINKS.values() for p in lst}
    for p in parts:
        if p not in used:
            warn.append(f"onderdeel '{p}' hoort bij geen enkel gewricht en wordt genegeerd")

    meshes = []
    for lst in LINKS.values():
        for p in lst:
            if p not in parts:
                continue
            required = parts[p]["printed"] and not p.startswith(TINY_MESH_AS_SPHERE)
            optional = (meshdir / f"{p}.stl").exists()  # gekochte delen: mesh als die er is
            if required or optional:
                meshes.append(p)

    children = {}
    for link, parent, jname in TREE:
        children.setdefault(parent, []).append((link, jname))
    if walk:
        order = {n: i for i, n in enumerate(WALK_BRANCH_ORDER)}
        children["torso"].sort(key=lambda c: order.get(c[0], 99))

    def geoms(link, ind):
        out = []
        o = origin[link]
        for p in LINKS[link]:
            if p not in parts:
                warn.append(f"onderdeel '{p}' (link {link}) staat niet in duck_links.csv")
                continue
            d = parts[p]
            if p in meshes:
                vcls = "visual" if d["printed"] else "visual_dark"
                out.append(f'{ind}<geom name="{p}" class="{vcls}" type="mesh" mesh="{p}" '
                           f'pos="{mm(sub((0, 0, 0), o))}" mass="{d["mass"] / 1000:.6g}"/>')
                if walk:
                    if p in WALK_FOOT_SOLES:
                        side = WALK_FOOT_SOLES[p]
                        ct, ca = WALK_COLLISION_BITS[side + "_foot"]
                        out.append(f'{ind}<geom name="{side}_foot_collision" class="foot" type="mesh" '
                                   f'mesh="{p}" pos="{mm(sub((0, 0, 0), o))}" '
                                   f'contype="{ct}" conaffinity="{ca}"/>')
                        c = d["com"]
                        out.append(f'{ind}<site name="{side}_foot" group="3" size="0.002" '
                                   f'pos="{mm(sub((c[0], c[1], 1.0), o))}"/>')
                    if p in WALK_LEG_COLLIDERS:
                        side, tag = WALK_LEG_COLLIDERS[p]
                        ct, ca = WALK_COLLISION_BITS[side]
                        out.append(f'{ind}<geom name="{side}_{tag}_collision" class="foot" type="mesh" '
                                   f'mesh="{p}" pos="{mm(sub((0, 0, 0), o))}" condim="1" '
                                   f'contype="{ct}" conaffinity="{ca}"/>')
                    continue
                if p in FOOT_COLLIDERS:
                    out.append(f'{ind}<geom name="{p}_col" class="foot" type="mesh" mesh="{p}" '
                               f'pos="{mm(sub((0, 0, 0), o))}"/>')
                if p in BODY_COLLIDERS:
                    out.append(f'{ind}<geom name="{p}_col" class="shell" type="mesh" mesh="{p}" '
                               f'pos="{mm(sub((0, 0, 0), o))}"/>')
            else:
                size = 0.003 if d["mass"] > 1.0 else 0.0015
                out.append(f'{ind}<geom name="{p}" class="part" type="sphere" size="{size}" '
                           f'pos="{mm(sub(d["com"], o))}" mass="{d["mass"] / 1000:.6g}"/>')
        if link == "torso":
            for n, m, pos in EXTRA_TORSO_PARTS:
                out.append(f'{ind}<geom name="{n}" class="part" type="sphere" size="0.003" '
                           f'pos="{mm(sub(pos, o))}" mass="{m / 1000:.6g}"/>')
        return out

    def body(link, parent, jname, ind):
        lines = []
        o = origin[link]
        rel = sub(o, origin[parent])
        lines.append(f'{ind}<body name="{link}" pos="{mm(rel)}">')
        axis = joints[jname][1]
        cls = actuator_class(jname)
        if walk and jn(jname).startswith("passive_"):
            cls = "passive"
        lo, hi = RANGE_DEG[jname]
        damp = ""
        if cls != "passive":
            is_diff = any(jname in (o_, c_) for _, o_, c_ in DIFFERENTIALS)
            damp = f' damping="{joint_damping(cls, is_diff):.5g}"'
        lines.append(f'{ind}  <joint name="{jn(jname)}" class="{cls}" axis="{f(axis)}" '
                     f'range="{math.radians(lo):.4f} {math.radians(hi):.4f}"{damp}/>')
        lines += geoms(link, ind + "  ")
        for c, cj in children.get(link, []):
            lines += body(c, link, cj, ind + "  ")
        lines.append(f"{ind}</body>")
        return lines

    spawn = (TORSO_ORIGIN_MM[0], TORSO_ORIGIN_MM[1], TORSO_ORIGIN_MM[2] + SPAWN_CLEARANCE_MM)
    x = []
    x.append('<mujoco model="duckling">')
    x.append('  <compiler angle="radian" meshdir="meshes" autolimits="true"/>')
    x.append('  <option timestep="0.001" integrator="implicitfast"/>')
    x.append('  <default>')
    x.append(f'    <default class="coreless"><joint armature="{CORELESS["armature"]}" '
             f'damping="{CORELESS["damping"]}" frictionloss="{CORELESS["frictionloss"]}"/>'
             f'<position kp="{CORELESS["kp"]}" kv="{CORELESS["kv"]}" '
             f'forcerange="-{CORELESS["force"]} {CORELESS["force"]}"/></default>')
    x.append(f'    <default class="servo"><joint armature="{SERVO["armature"]}" '
             f'damping="{SERVO["damping"]}" frictionloss="{SERVO["frictionloss"]}"/>'
             f'<position kp="{SERVO["kp"]}" kv="{SERVO["kv"]}" '
             f'forcerange="-{SERVO["force"]} {SERVO["force"]}"/></default>')
    x.append(f'    <default class="servo_s"><joint armature="{SERVO_S["armature"]}" '
             f'damping="{SERVO_S["damping"]}" frictionloss="{SERVO_S["frictionloss"]}"/>'
             f'<position kp="{SERVO_S["kp"]}" kv="{SERVO_S["kv"]}" '
             f'forcerange="-{SERVO_S["force"]} {SERVO_S["force"]}"/></default>')
    x.append('    <default class="visual"><geom contype="0" conaffinity="0" group="2" rgba="0.82 0.82 0.80 1"/></default>')
    x.append('    <default class="visual_dark"><geom contype="0" conaffinity="0" group="2" rgba="0.12 0.12 0.13 1"/></default>')
    x.append('    <default class="part"><geom contype="0" conaffinity="0" group="2" rgba="0.12 0.12 0.13 1"/></default>')
    x.append('    <default class="foot"><geom contype="1" conaffinity="1" group="4" density="0" friction="0.9 0.005 0.0001" rgba="0.9 0.4 0.2 0.4"/></default>')
    x.append('    <default class="passive"><joint armature="1e-6" damping="1e-4" frictionloss="5e-4"/></default>')
    x.append('    <default class="shell"><geom contype="2" conaffinity="0" group="4" density="0" rgba="0.2 0.4 0.9 0.3"/></default>')
    x.append('  </default>')
    x.append('  <asset>')
    for p in meshes:
        unit, sc = detect_scale(meshdir / f"{p}.stl", parts[p]["com"])
        if unit != "mm":
            warn.append(f"{p}.stl lijkt in {unit} te staan; schaal daarop aangepast")
        hull = (f' maxhullvert="{COLLISION_HULL_VERTS}"'
                if p in FOOT_COLLIDERS or p in BODY_COLLIDERS else "")
        x.append(f'    <mesh name="{p}" file="{p}.stl" scale="{sc:g} {sc:g} {sc:g}" inertia="{"shell" if p in SHELL_INERTIA_PARTS else MESH_INERTIA}"{hull}/>')
    if not walk:
        x.append('    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.26 0.28" rgb2="0.32 0.33 0.35" width="256" height="256"/>')
        x.append('    <material name="grid" texture="grid" texrepeat="40 40"/>')
    x.append('  </asset>')
    x.append('  <worldbody>')
    if not walk:
        x.append('    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>')
        x.append('    <geom name="floor" type="plane" size="2 2 0.05" material="grid" contype="1" conaffinity="3"/>')
    x.append(f'    <body name="{torso_name}" pos="{mm(spawn)}">')
    x.append(f'      <freejoint name="{torso_name + "_freejoint" if walk else "root"}"/>')
    x.append(f'      <site name="imu" group="3" pos="{mm(sub(IMU_POS_MM, TORSO_ORIGIN_MM))}" size="0.002"/>')
    x += geoms("torso", "      ")
    for c, cj in children.get("torso", []):
        x += body(c, "torso", cj, "      ")
    x.append('    </body>')
    x.append('  </worldbody>')
    if walk:
        # Policy werkt op gewrichtsniveau; de firmware mengt pitch/roll naar motor A/B.
        x.append('  <actuator>')
        dfs = []

        def walk_tree(node):
            for c, cj in children.get(node, []):
                dfs.append((c, node, cj))
                walk_tree(c)
        walk_tree("torso")
        for link, parent, jname in dfs:   # zelfde volgorde als de gewrichten
            if jn(jname).startswith("passive_"):
                continue
            cls = actuator_class(jname)
            lo, hi = RANGE_DEG[jname]
            extra = ""
            if any(jname in (o, c) for _, o, c in DIFFERENTIALS):
                # Beide motoren werken samen op elk differentieelgewricht: tot 2x motorkoppel.
                fmax = CLASS_PARAMS[cls]["force"] * DIFF_JOINT_FORCE_FACTOR
                extra = f' forcerange="-{fmax:g} {fmax:g}"'
            x.append(f'    <position name="{jn(jname)}" class="{cls}" joint="{jn(jname)}" '
                     f'ctrlrange="{math.radians(lo):.4f} {math.radians(hi):.4f}"{extra}/>')
        x.append('  </actuator>')
        x.append('  <sensor>')
        x.append('    <framequat name="orientation" objtype="site" noise="0.001" objname="imu"/>')
        x.append('    <gyro name="angular-velocity" site="imu" noise="0.005"/>')
        x.append('    <gyro name="imu_ang_vel" site="imu"/>')
        x.append('    <velocimeter name="imu_lin_vel" site="imu"/>')
        x.append('    <accelerometer name="imu_accel" site="imu"/>')
        x.append(f'    <subtreeangmom name="root_angmom" body="{torso_name}"/>')
        x.append('  </sensor>')
        x.append('</mujoco>')
        return "\n".join(x) + "\n", meshes, warn
    x.append('  <tendon>')
    for name, out_j, car_j in DIFFERENTIALS:
        x.append(f'    <fixed name="{name}_A"><joint joint="{out_j}" coef="1"/><joint joint="{car_j}" coef="1"/></fixed>')
        x.append(f'    <fixed name="{name}_B"><joint joint="{out_j}" coef="1"/><joint joint="{car_j}" coef="-1"/></fixed>')
    x.append('  </tendon>')
    x.append('  <actuator>')
    for name, out_j, car_j in DIFFERENTIALS:
        for s in ("A", "B"):
            x.append(f'    <position name="{name}_motor_{s}" class="{actuator_class(out_j)}" tendon="{name}_{s}" ctrlrange="-2.5 2.5"/>')
    for j in CORELESS_DIRECT:
        lo, hi = RANGE_DEG[j]
        x.append(f'    <position name="{j}" class="coreless" joint="{j}" ctrlrange="{math.radians(lo):.4f} {math.radians(hi):.4f}"/>')
    for j in SERVOS:
        lo, hi = RANGE_DEG[j]
        x.append(f'    <position name="{j}" class="{actuator_class(j)}" joint="{j}" ctrlrange="{math.radians(lo):.4f} {math.radians(hi):.4f}"/>')
    x.append('  </actuator>')
    x.append('  <sensor>')
    x.append('    <gyro name="imu_gyro" site="imu"/>')
    x.append('    <accelerometer name="imu_acc" site="imu"/>')
    x.append('    <framequat name="imu_quat" objtype="site" objname="imu"/>')
    x.append('  </sensor>')
    nj = len(TREE)
    x.append('  <keyframe>')
    x.append(f'    <key name="home" qpos="{mm(spawn)} 1 0 0 0 {" ".join(["0"] * nj)}"/>')
    x.append('  </keyframe>')
    x.append('</mujoco>')
    return "\n".join(x) + "\n", meshes, warn


def is_ascii_stl(path):
    """ASCII-STL begint met 'solid' en heeft geen geldige binaire lengte."""
    data = path.read_bytes()
    if not data[:5].lower() == b"solid":
        return False
    if len(data) >= 84:
        n = int.from_bytes(data[80:84], "little")
        if 84 + 50 * n == len(data):
            return False  # binair bestand dat toevallig met 'solid' begint
    return True


def ascii_to_binary_stl(path):
    """Zet een ASCII-STL om naar binaire STL (in place, origineel als .ascii.stl bewaard)."""
    import struct
    text = path.read_text(errors="ignore").split()
    tris = []
    i = 0
    while i < len(text):
        if text[i] == "facet" and text[i + 1] == "normal":
            nrm = tuple(float(v) for v in text[i + 2:i + 5])
            verts = []
            j = i + 5
            while len(verts) < 3 and j < len(text):
                if text[j] == "vertex":
                    verts.append(tuple(float(v) for v in text[j + 1:j + 4]))
                    j += 4
                else:
                    j += 1
            tris.append((nrm, verts))
            i = j
        else:
            i += 1
    path.rename(path.with_suffix(".ascii.stl"))
    with open(path, "wb") as fh:
        fh.write(b"binary STL converted by gen_duck_mjcf".ljust(80, b" "))
        fh.write(struct.pack("<I", len(tris)))
        for nrm, verts in tris:
            fh.write(struct.pack("<3f", *nrm))
            for v in verts:
                fh.write(struct.pack("<3f", *v))
            fh.write(b"\0\0")
    return len(tris)


def convert_meshes(meshdir):
    if not meshdir.is_dir():
        return
    for p in sorted(meshdir.glob("*.stl")):
        if p.name.endswith(".ascii.stl"):
            continue
        if is_ascii_stl(p):
            n = ascii_to_binary_stl(p)
            print(f"{p.name}: ASCII omgezet naar binair ({n} driehoeken)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    walk = "--walk" in sys.argv
    if len(args) < 2:
        print(__doc__)
        sys.exit(1)
    joints = read_joints(args[0])
    parts = read_parts(args[1])
    default_out = "robot_walk.xml" if walk else "duck.xml"
    out = Path(args[2] if len(args) > 2 else default_out)
    convert_meshes(out.parent / "meshes")
    xml, meshes, warn = build(joints, parts, out.parent / "meshes", walk=walk)
    out.write_text(xml, encoding="utf-8")
    print(f"{out} geschreven ({len(meshes)} meshes nodig in {out.parent / 'meshes'})")
    if walk:
        scene = out.parent / ("scene_" + out.name.replace("robot_", ""))
        scene.write_text(walk_scene(out.name), encoding="utf-8")
        print(f"{scene} geschreven (om het walk-model los te bekijken)")
    missing = [m for m in meshes if not (out.parent / "meshes" / f"{m}.stl").exists()]
    if missing:
        print("Nog ontbrekende STL's:", ", ".join(missing))
    for w in sorted(set(warn)):
        print("Let op:", w)


if __name__ == "__main__":
    main()
