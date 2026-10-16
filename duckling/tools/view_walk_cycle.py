"""Speel walk_cycle.npz kinematisch af in MuJoCo (geen fysica).

    python view_walk_cycle.py                       # romp volgt de animatie
    python view_walk_cycle.py --hang                # romp vast in de lucht
    python view_walk_cycle.py --scene scene_walk.xml --cycle walk_cycle.npz
"""
import argparse, time
import numpy as np, mujoco, mujoco.viewer

FOLLOW = [True]   # camera follows the robot; G in the viewer toggles

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="scene_walk.xml")   # scene = robot + vloer
ap.add_argument("--cycle", default="walk_cycle.npz")
ap.add_argument("--hang", action="store_true", help="romp vast op 10 cm hoogte")
ap.add_argument("--speed", type=float, default=1.0)
a = ap.parse_args()

m = mujoco.MjModel.from_xml_path(a.scene)
d = mujoco.MjData(m)
c = np.load(a.cycle)
names = [str(n) for n in c["joint_names"]]           # namen staan in het bestand zelf
adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
fps = float(c["fps"])
q, pos, quat = c["joint_pos"], c["root_pos"], c["root_quat"]
stride = pos[-1, :2] - pos[0, :2] + (pos[1, :2] - pos[0, :2])   # verplaatsing per cyclus
print("gewrichten:", ", ".join(names))
print(f"{len(q)} frames @ {fps:g} Hz")

def _key(k):
    if k == ord("G"):
        FOLLOW[0] = not FOLLOW[0]; print("camera volgt:", FOLLOW[0])


with mujoco.viewer.launch_passive(m, d, key_callback=_key) as v:
    loop = 0
    while v.is_running():
        for k in range(len(q)):
            if a.hang:
                d.qpos[:3] = [0, 0, 0.10]; d.qpos[3:7] = [1, 0, 0, 0]
            else:
                d.qpos[:2] = pos[k, :2] + loop * stride
                d.qpos[2] = pos[k, 2]
                d.qpos[3:7] = quat[k]                  # wxyz
            d.qpos[adr] = q[k]
            mujoco.mj_forward(m, d)
            if FOLLOW[0]:
                v.cam.lookat[:] = d.qpos[:3]
            v.sync()
            time.sleep(1 / (fps * a.speed))
            if not v.is_running():
                break
        loop += 1
