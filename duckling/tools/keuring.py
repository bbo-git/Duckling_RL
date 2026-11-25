"""keuring.py - vaste meetlat voor een loop-netwerk (2026-11-25).

Draait een onnx een paar keer kort in de simulatie (zonder viewer, zonder rust-modus: het netwerk
staat altijd aan, zoals met `rust uit` op de robot) en meet:
  staan        wegschuiven (mm) bij vx = 0
  loop 0.5     gehaalde snelheid t.o.v. de opdracht (50 % = 4.5 cm/s)
  loop 1.0     idem bij 100 %
  draai        gehaalde draaisnelheid bij 20 gr/s op de plaats
  heup-yaw     gemiddelde stand (tenen naar binnen = negatief; X-poten bij < -5 gr)
  onrust       gemiddelde verandering van de netwerk-uitvoer per stap (gr) - lager = rustiger
  sterk 1.6x   loop 0.5 met 1.6x sterkere servo's (de echte eend); zwak 0.7x idem
en geeft per regel OK / FOUT, plus een eindoordeel. Schrijft een regel bij in onnx/KEURING.md.

Gebruik (vanuit src/mjlab_microduck/robot/duckling, zoals de players):
  python ../../../../tools/keuring.py ../../../../../onnx/duckling_walk_650.onnx [meer.onnx ...]
  python ../../../../tools/keuring.py 650 1150          (kort: getal = onnx/duckling_walk_<n>.onnx)
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import duckling_play_onnx as P   # noqa: E402
import duckling_servo as DS      # noqa: E402

ONNX_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "onnx"))
V100 = 0.09     # m/s bij 100 % ("normaal")
T = 8.0         # s per proef; meten vanaf 2 s
T0 = 2.0

# grenzen (eindoordeel = alles OK)
GRENS = {
    "staan_mm": 30.0,      # hooguit 3 cm wegschuiven in 6 s
    "loop_frac": 0.6,      # minstens 60 % van de gevraagde snelheid
    "draai_frac": 0.5,     # minstens 50 % van de gevraagde draaisnelheid
    "heupyaw_gr": 5.0,     # tenen binnen 5 gr van recht
    "sterk_frac": 0.4,     # met 1.6x sterkere servo's nog 40 % vooruit
}

_BASE = {id(d): dict(d) for d in (DS.ES9051, DS.DIFF_HIP)}
_GRIDS = None


def _yaw(q):
    return 2.0 * np.arctan2(q[3], q[0])


def proef(onnx, vx=0.0, wz=0.0, kracht=1.0):
    global _GRIDS
    for d in (DS.ES9051, DS.DIFF_HIP):
        d["stall"] = _BASE[id(d)]["stall"] * kracht
    if _GRIDS is None:
        _GRIDS = [P.GridClock(p) for _, p in P.STYLES]
    pol = P.Policy(onnx)
    r = P.Duckling("scene_walk.xml", grid=_GRIDS[0])
    r.gait_clock, r.styled, r.grids = True, True, _GRIDS
    r.set_style(0)
    ji = {n: i for i, n in enumerate(r.names)}
    hy = [ji["left_hip_yaw"], ji["right_hip_yaw"]]
    start = None
    qs, da, prev = [], [], None
    n = int(T / 0.02)
    for k in range(n):
        t = k * 0.02
        tw = np.array([vx, 0.0, wz]) if t > 0.5 else np.zeros(3)
        a = pol(r.obs(tw, np.zeros(4)))
        r.step(a, tw)
        if t >= T0:
            if start is None:
                start = (r.d.qpos[:2].copy(), _yaw(r.d.qpos[3:7]), t)
            qs.append(np.degrees(r.d.qpos[r.qadr][hy]))
            if prev is not None:
                da.append(np.degrees(np.abs(a - prev)).mean())
        prev = a.copy()
        if r.fallen:
            return dict(val=True, t=t)
    p0, y0, t0 = start
    dur = T - t0
    h = y0
    d = r.d.qpos[:2] - p0
    vooruit = (np.cos(h) * d[0] + np.sin(h) * d[1])            # m langs de beginrichting
    dyaw = np.degrees(np.arctan2(np.sin(_yaw(r.d.qpos[3:7]) - y0), np.cos(_yaw(r.d.qpos[3:7]) - y0)))
    return dict(val=False, vooruit_mm=vooruit * 1000, v=vooruit / dur, w_gr=dyaw / dur,
                heupyaw=float(np.mean(qs)), onrust=float(np.mean(da)))


def keur(onnx):
    naam = os.path.basename(onnx)
    rij, ok = [], []

    def regel(label, waarde, goed):
        rij.append((label, waarde, goed))
        ok.append(goed)

    s = proef(onnx)
    if s["val"]:
        regel("staan", "VALT", False)
    else:
        regel("staan", "%+.0f mm" % s["vooruit_mm"], abs(s["vooruit_mm"]) <= GRENS["staan_mm"])
        regel("heup-yaw (staan)", "%+.1f gr" % s["heupyaw"], abs(s["heupyaw"]) <= GRENS["heupyaw_gr"])
        regel("onrust (staan)", "%.1f gr/stap" % s["onrust"], True)
    for frac in (0.5, 1.0):
        l = proef(onnx, vx=frac * V100)
        lab = "loop %.1f" % frac
        if l["val"]:
            regel(lab, "VALT", False)
            continue
        f = l["v"] / (frac * V100)
        regel(lab, "%.1f cm/s = %.0f %%" % (l["v"] * 100, f * 100), f >= GRENS["loop_frac"])
        if frac == 0.5:
            regel("heup-yaw (lopen)", "%+.1f gr" % l["heupyaw"], abs(l["heupyaw"]) <= GRENS["heupyaw_gr"])
            regel("onrust (lopen)", "%.1f gr/stap" % l["onrust"], True)
    w = proef(onnx, wz=np.radians(20.0))
    if w["val"]:
        regel("draai 20 gr/s", "VALT", False)
    else:
        regel("draai 20 gr/s", "%+.0f gr/s" % w["w_gr"], w["w_gr"] / 20.0 >= GRENS["draai_frac"])
    for kracht, lab in ((1.6, "loop 0.5, servo 1.6x"), (0.7, "loop 0.5, servo 0.7x")):
        k = proef(onnx, vx=0.5 * V100, kracht=kracht)
        if k["val"]:
            regel(lab, "VALT", False)
        else:
            f = k["v"] / (0.5 * V100)
            regel(lab, "%.0f %%" % (f * 100), f >= GRENS["sterk_frac"])
    for d in (DS.ES9051, DS.DIFF_HIP):
        d["stall"] = _BASE[id(d)]["stall"]
    return naam, rij, all(ok)


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    regels = []
    for a in args:
        f = os.path.join(ONNX_DIR, "duckling_walk_%s.onnx" % a) if a.isdigit() else a
        naam, rij, goed = keur(f)
        print("\n== %s" % naam)
        for label, waarde, g in rij:
            print("  %-24s %-20s %s" % (label, waarde, "ok" if g else "FOUT"))
        print("  oordeel: %s" % ("GOED - mag hangend op de eend" if goed else "NOG NIET"))
        kort = " | ".join("%s %s%s" % (l, w, "" if g else " (FOUT)") for l, w, g in rij)
        regels.append("- %s `%s` **%s**: %s" % (time.strftime("%Y-%m-%d %H:%M"), naam,
                                                "GOED" if goed else "nog niet", kort))
    log = os.path.join(ONNX_DIR, "KEURING.md")
    nieuw = not os.path.exists(log)
    with open(log, "a", encoding="utf-8") as fh:
        if nieuw:
            fh.write("# Keuring loop-netwerken (tools/keuring.py)\n\nGrenzen: %s\n\n" % GRENS)
        fh.write("\n".join(regels) + "\n")
    print("\nbijgeschreven in %s" % log)


if __name__ == "__main__":
    main()
