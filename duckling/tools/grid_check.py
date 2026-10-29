"""grid_check.py STIJL.json - controle per gang van het rooster (servosnelheid, ZMP, zoolkanteling, botsing)."""
import sys, json
import numpy as np
import walk_speed as W
from gait_tune import Scorer, fmt
style = W.Style(W.load_params(sys.argv[1]))
sc = Scorer("scene_walk.xml")
rows = []
for s in W.SPEEDS:
    for w in W.TURNS:
        for y in W.SIDES:
            p = style.params(float(s)); p.turn_dps, p.side_m = float(w), float(y)
            r = sc.metrics(p)
            rows.append((s, w, y, r))
print("snelheid  bocht  zij |  enkel  teen  heup | ZMP-buiten  zool-roll | botsing")
for s in W.SPEEDS:
    sel = [r for r in rows if r[0] == s]
    worst = lambda k: max(x[3]["top"][k] for x in sel)
    zo = max(x[3]["zmp_out"] for x in sel); rm = max(x[3]["roll_mean"] for x in sel); col = max(x[3]["collide_mm"] for x in sel)
    straight = [x for x in sel if x[1] == 0 and x[2] == 0][0][3]
    print(f"{s*100:6.0f}%  {straight['speed']*100:5.1f} cm/s | {worst('knee'):5.1f} {worst('ankle'):5.1f} {worst('hip_pitch'):5.1f} | "
          f"{zo*100:5.0f}%  {rm:5.1f} gr | {col:4.1f} mm   (rechtuit: enkel {straight['top']['knee']:.1f}, ZMP {straight['zmp_out']*100:.0f}%)")
bad = [(s, w, y, r) for s, w, y, r in rows if max(r["top"]["knee"], r["top"]["ankle"], r["top"]["hip_pitch"]) > 9.0]
print(f"{len(bad)} van {len(rows)} gangen boven 9 rad/s")
for s, w, y, r in bad[:15]:
    print(f"   {s*100:.0f}% bocht {w:+.0f} zij {y*1000:+.0f}: {fmt(r)}")
