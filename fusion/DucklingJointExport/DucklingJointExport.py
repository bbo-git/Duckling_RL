"""
Fusion 360-script: exporteert alle joints en onderdelen van de duck naar CSV.

Gebruik:
  1. Fusion: Utilities > Add-Ins > Scripts and Add-Ins > Create (Python),
     vervang de inhoud van het nieuwe .py-bestand door dit bestand en klik Run.
  2. Zet de robot eerst in de nulstand (de pose die in MuJoCo qpos = 0 wordt).
  3. Je krijgt twee bestanden op je bureaublad:
       duck_joints.csv  - per joint: naam, type, onderdelen, oorsprong (mm), as, limieten
       duck_links.csv   - per onderdeel: massa volgens Fusion en zwaartepunt (mm)
  4. Stuur beide CSV's door.

Alleen joints en constructie-assen waarvan de naam begint met PREFIX (hieronder, nu "J2")
worden geexporteerd; oude joints en assen zonder die prefix worden overgeslagen. De prefix
(plus een eventuele _ of spatie erna) gaat van de naam af: "J2_left_hip_pitch" -> "left_hip_pitch".
Gewrichten zonder Fusion-joint: maak een constructie-as (Construct > Axis Through
Cylinder/Cone/Torus) op de as of het lagergat met een naam die met de prefix begint.

Alle coordinaten worden omgerekend naar het globale assenstelsel van de assembly, in mm.
"""

import adsk.core
import adsk.fusion
import traceback
import os
import csv
import math

CM_TO_MM = 10.0
PREFIX = "J2"          # alleen joints/assen met deze prefix (hoofdletterongevoelig)


def _has_prefix(name):
    return name.upper().startswith(PREFIX.upper())


def _strip(name):
    return name[len(PREFIX):].lstrip("_- ")


def _occ_name(occ):
    return occ.fullPathName if occ else "(root)"


def _contexts(root, comp):
    """Alle (occurrence, transform) waarin een component voorkomt."""
    if comp == root:
        return [(None, adsk.core.Matrix3D.create())]
    return [(o, o.transform2) for o in root.allOccurrencesByComponent(comp)]


def _geometry(j):
    """JointGeometry van een joint of as-built joint."""
    g = None
    if hasattr(j, "geometryOrOriginOne"):
        g = j.geometryOrOriginOne
    elif hasattr(j, "geometry"):
        g = j.geometry
    jo = adsk.fusion.JointOrigin.cast(g)
    if jo:
        g = jo.geometry
    return adsk.fusion.JointGeometry.cast(g)


def _rotation_info(j, geom):
    """Geeft (as-vector, min_deg, max_deg) voor een revolute joint, anders (None, '', '')."""
    motion = j.jointMotion
    rm = adsk.fusion.RevoluteJointMotion.cast(motion)
    if not rm:
        return None, "", ""
    axis = None
    try:
        axis = rm.rotationAxisVector
    except Exception:
        axis = None
    if axis is None and geom:
        # Terugval: Fusion's joint-Z is de primaire as van de joint-geometrie
        d = rm.rotationAxis
        if d == adsk.fusion.JointDirections.XAxisJointDirection:
            axis = geom.secondaryAxisVector
        elif d == adsk.fusion.JointDirections.YAxisJointDirection:
            axis = geom.thirdAxisVector
        else:
            axis = geom.primaryAxisVector
    lo = hi = ""
    lim = rm.rotationLimits
    if lim.isMinimumValueEnabled:
        lo = round(math.degrees(lim.minimumValue), 2)
    if lim.isMaximumValueEnabled:
        hi = round(math.degrees(lim.maximumValue), 2)
    return axis, lo, hi


def _type_name(j):
    try:
        t = j.jointMotion.jointType
        names = {
            adsk.fusion.JointTypes.RigidJointType: "rigid",
            adsk.fusion.JointTypes.RevoluteJointType: "revolute",
            adsk.fusion.JointTypes.SliderJointType: "slider",
            adsk.fusion.JointTypes.CylindricalJointType: "cylindrical",
            adsk.fusion.JointTypes.PinSlotJointType: "pinslot",
            adsk.fusion.JointTypes.PlanarJointType: "planar",
            adsk.fusion.JointTypes.BallJointType: "ball",
        }
        return names.get(t, str(t))
    except Exception:
        return "?"


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox("Open eerst het duck-ontwerp.")
            return
        root = design.rootComponent

        out_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.isdir(out_dir):
            out_dir = os.path.expanduser("~")
        joints_path = os.path.join(out_dir, "duck_joints.csv")
        links_path = os.path.join(out_dir, "duck_links.csv")

        # ---------- Joints ----------
        rows = []
        all_joints = [(j, "joint") for j in root.allJoints] + \
                     [(j, "as-built") for j in root.allAsBuiltJoints]
        skipped = 0
        for j, kind in all_joints:
            if not _has_prefix(j.name):
                skipped += 1
                continue
            geom = _geometry(j)
            axis, lo, hi = _rotation_info(j, geom)
            for occ, xf in _contexts(root, j.parentComponent):
                if geom:
                    p = geom.origin.copy()
                    p.transformBy(xf)
                    pos = [round(v * CM_TO_MM, 3) for v in (p.x, p.y, p.z)]
                else:
                    pos = ["", "", ""]
                if axis:
                    a = axis.copy()
                    a.transformBy(xf)
                    a.normalize()
                    ax = [round(v, 4) for v in (a.x, a.y, a.z)]
                else:
                    ax = ["", "", ""]
                rows.append([
                    _strip(j.name), kind, _type_name(j),
                    _occ_name(j.occurrenceOne), _occ_name(j.occurrenceTwo),
                    _occ_name(occ),
                    *pos, *ax, lo, hi,
                    "ja" if j.isSuppressed else "nee",
                ])

        # ---------- Constructie-assen met PREFIX (voor gewrichten zonder Fusion-joint) ----------
        comps = [root] + [c for c in design.allComponents if c != root]
        for comp in comps:
            for ca in comp.constructionAxes:
                if not _has_prefix(ca.name):
                    continue
                line = ca.geometry
                for occ, xf in _contexts(root, comp):
                    p = line.origin.copy()
                    p.transformBy(xf)
                    a = line.direction.copy()
                    a.transformBy(xf)
                    a.normalize()
                    rows.append([
                        _strip(ca.name), "constructie-as", "revolute", "", "", _occ_name(occ),
                        *[round(v * CM_TO_MM, 3) for v in (p.x, p.y, p.z)],
                        *[round(v, 4) for v in (a.x, a.y, a.z)],
                        "", "", "nee",
                    ])

        with open(joints_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "joint", "soort", "type", "onderdeel_1", "onderdeel_2", "context",
                "x_mm", "y_mm", "z_mm", "as_x", "as_y", "as_z",
                "min_graden", "max_graden", "onderdrukt",
            ])
            w.writerows(rows)

        # ---------- Onderdelen ----------
        links = []
        for body in root.bRepBodies:  # losse bodies direct in de root
            try:
                pp = body.physicalProperties
                com = pp.centerOfMass
                links.append([
                    "(root)/" + body.name, "(body)", round(pp.mass * 1000.0, 3),
                    round(com.x * CM_TO_MM, 3), round(com.y * CM_TO_MM, 3), round(com.z * CM_TO_MM, 3),
                    0, "",
                ])
            except Exception:
                links.append(["(root)/" + body.name, "(body)", "", "", "", "", "", "fout"])
        for occ in root.allOccurrences:
            try:
                pp = occ.physicalProperties
                com = pp.centerOfMass
                links.append([
                    occ.fullPathName, occ.component.name,
                    round(pp.mass * 1000.0, 3),
                    round(com.x * CM_TO_MM, 3), round(com.y * CM_TO_MM, 3), round(com.z * CM_TO_MM, 3),
                    occ.childOccurrences.count, "",
                ])
            except Exception:
                links.append([occ.fullPathName, occ.component.name, "", "", "", "", "", "fout"])

        with open(links_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "onderdeel_pad", "component", "massa_fusion_g",
                "com_x_mm", "com_y_mm", "com_z_mm", "aantal_subonderdelen", "echte_massa_g",
            ])
            w.writerows(links)

        ui.messageBox(
            f"{len(rows)} joints met prefix {PREFIX} ({skipped} andere overgeslagen) en "
            f"{len(links)} onderdelen geexporteerd naar:\n"
            f"{joints_path}\n{links_path}\n\n"
            "Vul in duck_links.csv de kolom echte_massa_g in voor motoren, servo's, "
            "accu en elektronica (Fusion rekent die met een verkeerde dichtheid)."
        )
    except Exception:
        if ui:
            ui.messageBox("Mislukt:\n{}".format(traceback.format_exc()))