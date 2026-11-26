"""add_com_ref.py - zet per rooster-stand het zwaartepunt t.o.v. de standvoet(en) (m, langs de looprichting) in het
gangrooster, voor de beloning com_track in de training (2026-11-26: "in zijn pas vallen").
    python add_com_ref.py grids/gait_grid_pasje_v10.npz
Schrijft com_ref (S, W, Y, T) en stand_com_ref (zwaartepunt t.o.v. het midden van de voeten in HOME) in hetzelfde bestand."""
import sys
import numpy as np
import mujoco
import gait_gen as G

path = sys.argv[1]
d = dict(np.load(path, allow_pickle=True))
kin = G.DuckKinematics(sys.argv[2] if len(sys.argv) > 2 else "scene_walk.xml")
m, md = kin.m, kin.d
sid = [m.site("left_foot").id, m.site("right_foot").id]
S, W, Y = d["length"].shape
T = d["joint_pos"].shape[3]
out = np.zeros((S, W, Y, T))
qadr = kin.qadr
for a in range(S):
    for b in range(W):
        for c in range(Y):
            n = int(d["length"][a, b, c])
            for k in range(n):
                md.qpos[:] = 0
                md.qpos[2] = d["root_height"][a, b, c, k]
                md.qpos[3:7] = d["root_quat_local"][a, b, c, k]
                md.qpos[qadr] = d["joint_pos"][a, b, c, k]
                mujoco.mj_kinematics(m, md); mujoco.mj_comPos(m, md)
                com = md.subtree_com[1]
                ct = d["contact"][a, b, c, k] > 0.5
                feet = [md.site_xpos[sid[i]][0] for i in (0, 1) if ct[i]] or [md.site_xpos[s][0] for s in sid]
                R = md.xmat[1].reshape(3, 3); yaw = np.arctan2(R[1, 0], R[0, 0])
                out[a, b, c, k] = np.cos(yaw) * (com[0] - np.mean(feet))   # langs de romprichting (yaw ~0 lokaal)
md.qpos[:] = 0; md.qpos[2] = float(d["stand_root_height"]); md.qpos[3:7] = d["stand_root_quat_local"]
md.qpos[qadr] = d["stand_joint_pos"]
mujoco.mj_kinematics(m, md); mujoco.mj_comPos(m, md)
d["com_ref"] = out
d["stand_com_ref"] = np.array(md.subtree_com[1][0] - np.mean([md.site_xpos[s][0] for s in sid]))
np.savez_compressed(path, **d)
i = list(d["speeds"]).index(0.3) if 0.3 in list(d["speeds"]) else 5
print("klaar:", path, "| stilstaan %.1f mm | 30 %%: %s mm" % (float(d["stand_com_ref"]) * 1000,
      np.round(out[i, 2, 1, :int(d["length"][i, 2, 1]):4] * 1000, 1)))
