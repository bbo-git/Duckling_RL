"""stap_grid.py - zet de pasjes van stap_gen.py om naar een trainingsrooster (zelfde assen als de pasje-roosters).
(2026-11-26) Alle vooruit-snelheden = dezelfde pas (pasje, pauze, pasje, om en om); stilstaan/achteruit = uit het basisrooster.
Halve cyclus = 1 pas: van midden-pauze tot midden-pauze (gewicht verdeeld), zodat het pasjes-schema van de training
(wacht op de grens van een halve cyclus, dan n halve cycli) precies hele pasjes loopt.
Gebruik (vanuit src/mjlab_microduck/robot/duckling):
  python ../../../../tools/stap_grid.py [grids/gait_grid_stap_v1.npz] ['{"swing_s":0.24}']"""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
out = sys.argv[1] if len(sys.argv) > 1 else "grids/gait_grid_stap_v3.npz"
kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
sys.argv = sys.argv[:1]
import stap_gen as S

basis = dict(np.load("grids/gait_grid_pasje_v11.npz", allow_pickle=True))
names = [str(n) for n in basis["joint_names"]]
fr = S.maak(**kw)
fps = 50
n = len(fr)
hold = int(kw.get("hold_s", 1.0) * fps)
# rollen: begin midden in de laatste pauze (gewicht verdeeld, beide voeten plat)
shift = hold // 2
order = list(range(n - shift, n)) + list(range(0, n - shift))
F = [fr[i] for i in order]
cols = [S.J.index(nm) for nm in names]
q = np.array([f["q"][cols] for f in F])                       # (T, 14) in trainingsvolgorde
b = np.array([f["b"] for f in F])                             # x, y, z, roll
c = np.array([f["contact"] for f in F], dtype=float)
T = n; cyc = T / fps
# vooruitgang per cyclus: 2 passen
dist = (fr[-1]["b"][0] - fr[0]["b"][0])
v = dist / cyc
t = np.arange(T) / fps
x_loc = b[:, 0] - (b[0, 0] + v * t)
x_loc -= x_loc.mean()
root_local = np.stack([x_loc, b[:, 1], b[:, 2]], 1)
quat = np.array([S.euler(r) for r in b[:, 3]])
print("pas: %d standen, cyclus %.2f s, %.1f mm vooruit per cyclus = %.1f mm/s (%.1f %% van v100)" % (T, cyc, dist * 1000, v * 1000, 100 * v / float(basis["v100"])))

nS, nW, nY = basis["length"].shape
Tm = max(T, basis["joint_pos"].shape[3])
def pad(a, T_new):
    sh = list(a.shape); sh[3] = T_new
    o = np.zeros(sh); o[:, :, :, :a.shape[3]] = a; return o
g = {}
for k in ("joint_pos", "root_height", "root_local", "root_quat_local", "contact"):
    g[k] = pad(basis[k], Tm)
g["length"] = basis["length"].copy(); g["cycle_s"] = basis["cycle_s"].copy(); g["vel"] = basis["vel"].copy()
g["yaw_rate"] = basis["yaw_rate"].copy()
S_ax = basis["speeds"]
for i, s in enumerate(S_ax):
    if s <= 0:
        continue
    for w in range(nW):
        for y in range(nY):
            g["joint_pos"][i, w, y, :T] = q; g["joint_pos"][i, w, y, T:] = 0
            g["root_height"][i, w, y, :T] = b[:, 2]
            g["root_local"][i, w, y, :T] = root_local
            g["root_quat_local"][i, w, y, :T] = quat
            g["contact"][i, w, y, :T] = c
            g["length"][i, w, y] = T; g["cycle_s"][i, w, y] = cyc
            g["vel"][i, w, y] = [v, 0.0]; g["yaw_rate"][i, w, y] = 0.0
# eerste vooruit-knoop dicht bij 0: anders mengt een kleine opdracht de pas met het trappelen van de 0-cel
g["speeds"] = basis["speeds"].copy(); g["speeds"][S_ax > 0][0]
i0 = int(np.argmax(S_ax > 0)); g["speeds"][i0] = 0.02
for k in ("turns", "sides", "joint_names", "style", "fps", "v100", "stand_joint_pos", "stand_root_height",
          "stand_root_local", "stand_root_quat_local", "stand_contact"):
    g[k] = basis[k]
g["walk_s0"] = np.array(0.02)          # bij elke vooruit-opdracht meteen de hele pas
g["stap_params"] = json.dumps(kw)
g["stap_v"] = np.array(v)
np.savez_compressed(out, **g)
print("opgeslagen:", out)
