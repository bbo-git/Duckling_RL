"""test_core.py - vergelijkt de C-kern (test_core.cpp) met onnxruntime + de Python-player-rekenwijze."""
import sys, json, subprocess, numpy as np, onnxruntime as ort
onnx, grid, calib, exe = sys.argv[1:5]
rng = np.random.default_rng(0)
sess = ort.InferenceSession(onnx); inp = sess.get_inputs()[0].name
d = np.load(grid); S, W = d["speeds"], d["turns"]; cyc = d["cycle_s"][:, :, int(np.argmin(np.abs(d["sides"])))]; v100 = float(d["v100"])
JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle", "neck_pitch", "head_pitch",
          "head_roll", "head_yaw", "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
R = {"hip_yaw": 20, "hip_roll": 20, "hip_pitch": 60, "knee": (-50, 90), "ankle": (-60, 90), "neck_pitch": 30, "head_pitch": 30, "head_roll": 15, "head_yaw": 90}
def rng_of(j):
    for k, v in R.items():
        if j.endswith(k): return np.radians(v if isinstance(v, tuple) else (-v, v))
LO = np.array([rng_of(j)[0] for j in JOINTS]); HI = np.array([rng_of(j)[1] for j in JOINTS])
HEADJ = ("neck_pitch", "head_pitch", "head_roll", "head_yaw")
ALPHA = np.array([0.25 if j in HEADJ else 0.4 for j in JOINTS])
VE = {"hip_yaw": 9.0, "knee": 9.0, "ankle": 9.0, "neck_pitch": 9.0, "head_pitch": 8.1, "hip_pitch": 6.0, "hip_roll": 6.0, "head_yaw": 5.4, "head_roll": 5.4}
VEST = np.array([next(v for k, v in VE.items() if j.endswith(k)) * 0.75 for j in JOINTS])
def axis(ax, x):
    x = float(np.clip(x, ax[0], ax[-1])); i = int(np.clip(np.searchsorted(ax, x) - 1, 0, len(ax) - 2)); return i, (x - ax[i]) / (ax[i + 1] - ax[i])
def cycle(t):
    (i, wi), (j, wj) = axis(S, t[0] / v100), axis(W, np.degrees(t[2])); c = cyc
    return (1 - wi) * ((1 - wj) * c[i, j] + wj * c[i, j + 1]) + wi * ((1 - wj) * c[i + 1, j] + wj * c[i + 1, j + 1])
def walk(t):
    return float(np.clip(max(abs(t[0] / v100) / 0.15, abs(np.degrees(t[2])) / 30.0, abs(t[1]) * 0.45 / 0.006), 0, 1))
c = json.load(open(calib))
def us(n, deg): return np.clip(1500 + c[n]["teken"] * (c[n]["offset"] + deg) * 10, 600, 2400)
def servos(q):
    dg = np.degrees(q); J = {j: dg[i] for i, j in enumerate(JOINTS)}
    o = []
    for side, h in (("left", "L_heup"), ("right", "R_heup")):
        sp, sr = c[h]["sp"], c[h]["sr"]; p, r = J[f"{side}_hip_pitch"], J[f"{side}_hip_roll"]
        o += [us(h[0] + "_diffA", 1.5 * (sp * p + sr * r)), us(h[0] + "_diffB", 1.5 * (sp * p - sr * r))]
    sp, sr = c["kop"]["sp"], c["kop"]["sr"]
    o += [us("kop_diffA", 1.5 * (sp * J["head_yaw"] + sr * J["head_roll"])), us("kop_diffB", 1.5 * (sp * J["head_yaw"] - sr * J["head_roll"]))]
    o += [us("L_yaw", J["left_hip_yaw"]), us("R_yaw", J["right_hip_yaw"]), us("L_knie", J["left_knee"]), us("R_knie", J["right_knee"]),
          us("L_enkel", J["left_ankle"]), us("R_enkel", J["right_ankle"]), us("nek", J["neck_pitch"]), us("kop_pitch", J["head_pitch"]), us("kaak", 0.0)]
    return np.array(o)
filt = np.zeros(14); est = np.zeros(14); estp = np.zeros(14); la = np.zeros(14); ph = 0.0
rows, ref = [], []
for k in range(300):
    g = rng.normal(0, 0.5, 3); gr = rng.normal(0, 0.15, 3) + [0, 0, -1]; gr /= np.linalg.norm(gr)
    tw = [0.06 * np.sin(k / 40), 0.0, 0.6 * np.sin(k / 23)] if k > 30 else [0.0, 0.0, 0.0]
    hd = rng.normal(0, 0.2, 4)
    rows.append(" ".join("%.7f" % x for x in [*g, *gr, *tw, *hd]))
    moving = np.hypot(tw[0], tw[1]) > 0.02 or abs(tw[2]) > 0.1
    w = walk(tw)
    o = np.concatenate([g, gr, est, (est - estp) / 0.02, la, tw, hd if moving else np.zeros(4), np.zeros(6),
                        [w * np.sin(2 * np.pi * ph), w * np.cos(2 * np.pi * ph)], [1.0, 0.0]]).astype(np.float32)
    a = sess.run(None, {inp: o[None]})[0][0].astype(np.float64)
    ph = (ph + 0.02 / cycle(tw)) % 1.0
    la = a.copy(); t = np.clip(a, LO, HI); filt = filt + ALPHA * (t - filt); estp = est.copy()
    est = est + np.clip(filt - est, -VEST * 0.02, VEST * 0.02)
    ref.append(np.concatenate([a, filt, servos(filt), [ph]]))
out = subprocess.run([exe], input="\n".join(rows) + "\n", capture_output=True, text=True).stdout.strip().split("\n")
C = np.array([[float(x) for x in l.split()] for l in out]); ref = np.array(ref)
# 1) netwerk: onnxruntime op precies de observatie van de C-kern (geen terugkoppeling)
obsC = C[:, 44:].astype(np.float32)
aR = np.array([sess.run(None, {inp: o[None]})[0][0] for o in obsC])
print("netwerk: max verschil C vs onnxruntime op dezelfde observatie (rad):", np.abs(aR - C[:, :14]).max())
# 2) servo-omrekening op de doelen van de C-kern
print("servo's: max verschil (us):", np.abs(np.array([servos(f) for f in C[:, 14:28]]) - C[:, 28:43]).max())
# 3) eerste stap met dezelfde begintoestand: volledige keten
print("stap 0 volledige keten, max verschil:", np.abs(C[0, :44] - ref[0]).max())
print("(een gesloten lus met willekeurige invoer loopt na ~30 stappen uiteen door float32 vs float64 - verwacht)")
