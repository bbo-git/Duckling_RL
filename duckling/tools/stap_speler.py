"""stap_speler.py - speel de pasjes van stap_gen.py af in de viewer, MET fysica (servomodel), ZONDER netwerk
(2026-11-26). Stap ... pauze ... stap ... pauze, om en om, eindeloos.
Gebruik (vanuit src/mjlab_microduck/robot/duckling):
  python ../../../../tools/stap_speler.py                       (stap_frames.npz uit tools/)
  python ../../../../tools/stap_speler.py --kracht 1.0          (zwakkere servo's van het simulatiemodel)
  python ../../../../tools/stap_speler.py --traag 2             (2x vertraagd afspelen, fysica loopt mee)
Valt hij, dan begint hij na 1 s opnieuw. Spatie = pauze."""
import os, sys, time, argparse, io, contextlib
import numpy as np, mujoco, mujoco.viewer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duckling_play_onnx as P
import duckling_servo as DS

ap = argparse.ArgumentParser()
ap.add_argument("frames", nargs="?", default=os.path.join(HERE, "stap_frames.npz"))
ap.add_argument("--kracht", type=float, default=1.6)   # 2026-11-26: ~de echte eend (stap_gen is daarop afgesteld)
ap.add_argument("--traag", type=float, default=1.0)
a = ap.parse_args()
F = np.load(a.frames)
Q, C, names = F["q"], F["contact"], [str(n) for n in F["joint_names"]]
for dd in (DS.ES9051, DS.DIFF_HIP):
    dd["stall"] *= a.kracht
r = P.Duckling("scene_walk.xml"); r.gait_clock = False; r.residual = False
idx = [names.index(n) for n in r.names]
seq = [0] * 50 + list(range(len(Q)))
st = dict(k=0, pauze=False, val_t=None)

def key(c):
    if c == 32: st["pauze"] = not st["pauze"]

print("stap_speler: %d standen (%.1f s per 2 passen), servo's %.1fx, %.1fx vertraagd. Spatie = pauze." % (len(Q), len(Q) / 50, a.kracht, a.traag))
with mujoco.viewer.launch_passive(r.m, r.d, key_callback=key) as v:
    v.cam.lookat[:] = [0.0, 0.0, 0.05]; v.cam.distance = 0.45; v.cam.azimuth = 200; v.cam.elevation = -12
    t_next = time.time(); fase = None
    while v.is_running():
        if not st["pauze"]:
            i = seq[st["k"] % len(seq)] if st["k"] < len(seq) else seq[50 + (st["k"] - 50) % len(Q)]
            with contextlib.redirect_stdout(io.StringIO()):
                r.step(Q[i][idx] - r.home, None)
            c = tuple(C[i]); f = "staan/verschuiven" if c == (1, 1) else ("op LINKS, rechts zwaait" if c == (1, 0) else "op RECHTS, links zwaait")
            if f != fase:
                print("  %s" % f); fase = f
            st["k"] += 1
            if r.fallen and st["val_t"] is None:
                st["val_t"] = time.time(); print("  GEVALLEN")
            if st["val_t"] and time.time() - st["val_t"] > 1.0:
                r.reset(); st.update(k=0, val_t=None); print("  opnieuw")
            v.cam.lookat[:] = [r.d.qpos[0], r.d.qpos[1], 0.05]
        v.sync()
        t_next += 0.02 * a.traag
        time.sleep(max(0.0, t_next - time.time()))
