"""loop_gen.py - vloeiend loopje (2026-11-27, Jairo: 'direct naar een vloeiend loopje, een beetje conservatief').
Anders dan stap_gen (losse fasen, zwaartepunt elke pas terug naar het midden met snelheid 0):
- zwaartepunt opzij = doorlopende sinus (piek boven de standvoet midden in de zwaai, met vaart door het midden);
- zwaartepunt vooruit = constante snelheid;
- dubbele steun kort: achterste voet rolt over de teen op terwijl de voorste van hiel naar plat rolt;
- zwaai: teen eerst omlaag (knie), dan teen op, landen op de hiel.
Een cyclus = pas met links staan + pas met rechts staan. Zelfde uitvoer als stap_gen (frames, npz)."""
import os, sys, json, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import stap_gen as S
from stap_gen import solve, teen_site, hiel_site, Ry_, herm, mj, H0, COM_DX, HEEL_DX, kin, home_feet, J, qi

def maak(Th=0.40, A=0.003, L=0.010, ds=0.30, lift=0.014, teen_af=15.0, teen=25.0, hiel=12.0, rise=0.35, top=0.3,
         com_x=0.0, strek=0.002, breed=0.0, fps=50, aanloop=2, n_stap=8, alles=False):
    """Th = tijd per pas (s); A = zwaartepunt opzij (m); L = paslengte (m, lichaam per pas); ds = fractie dubbele steun.
    aanloop = aantal passen waarin A en L oplopen vanuit stilstand. Uitvoer: de laatste volledige cyclus (2 passen,
    periodiek, voor het rooster), of met alles=True alle passen vanaf stilstand (voor de speler)."""
    y0 = home_feet['left'][1]; xs = home_feet['left'][0]; z0 = home_feet['left'][2]
    n_half = int(round(Th * fps)); d = ds / 2
    q, b = kin.home.copy(), np.array([0, 0, H0, 0.0])
    P = math.radians(teen_af); peak = max(P, math.radians(teen)); HL = math.radians(hiel)
    v_h = np.array([HEEL_DX, 0.0, kin.site_above_floor['left']])
    fx = {'left': xs, 'right': xs}
    frames = []; th_land = 0.0
    for i in range(n_stap):
        stand, swing = ('left', 'right') if i % 2 == 0 else ('right', 'left')
        sgn = 1 if stand == 'left' else -1
        sc = min(1.0, (i + 1) / (aanloop + 1)) if aanloop else 1.0
        Li, Ai = L * sc, A * sc
        st = np.array([fx[stand], sgn * y0, z0]); start = np.array([fx[swing], -sgn * y0, z0])
        end = np.array([fx[stand] + Li, -sgn * (y0 + breed), z0])
        c_from = (fx[stand] + fx[swing]) / 2 + COM_DX + com_x; c_to = (fx[stand] + end[0]) / 2 + COM_DX + com_x
        Pi = P * (sc if i == 0 else 1.0)
        def heel_ref(th):
            return teen_site(start, th) + v_h - Ry_(th) @ v_h
        p0 = heel_ref(Pi)
        for k in range(n_half):
            u = (k + 0.5) / n_half
            cxy = np.array([c_from + (c_to - c_from) * u, sgn * Ai * math.sin(math.pi * u)])
            f = {}; rot = {}
            if u < d:
                w = mj(u / d)
                th_s = th_land * (1 - w); f[stand] = hiel_site(st, th_s); rot[stand] = Ry_(th_s)
                th_w = Pi * w; f[swing] = teen_site(start, th_w); rot[swing] = Ry_(th_w)
                con = (1, 1)
            elif u < 1 - d:
                s_ = (u - d) / (1 - 2 * d)
                f[stand] = st.copy(); rot[stand] = Ry_(0.0)
                p = np.array([herm(s_, 0, 1, p0[0], end[0], 0.0, 0.0), end[1], 0.0])
                zt = z0 + lift * (0.5 + 0.5 * sc)
                p[2] = herm(s_, 0, rise, p0[2], zt, 0.0, 0.0) if s_ < rise else herm(s_, rise, 1, zt, end[2], 0.0, 0.0)
                th = herm(s_, 0, top, Pi, peak, 0.0, 0.0) if s_ < top else (herm(s_, top, 0.8, peak, -HL, 0.0, 0.0) if s_ < 0.8 else -HL)
                f[swing] = hiel_site(p, th); rot[swing] = Ry_(th)
                con = (1, 0) if stand == 'left' else (0, 1)
            else:
                f[stand] = st.copy(); rot[stand] = Ry_(0.0)
                f[swing] = hiel_site(end, -HL); rot[swing] = Ry_(-HL)
                con = (1, 1)
            zz = H0 + strek * math.sin(math.pi * u)
            q, b, res, c = solve(f, cxy, None, q0=q, base0=b, iters=60, z_t=zz, foot_rot=rot)
            if alles or i >= n_stap - 2:
                frames.append(dict(q=q.copy(), b=b.copy(), com=c.copy(), feet=f, contact=con, res=res))
        fx[swing] = end[0]; th_land = -HL
    return frames

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = sys.argv[2] if len(sys.argv) > 2 else 'loop_frames.npz'
    fr = maak(**kw)
    worst = max(max(r['res'][s][0] for s in r['res']) for r in fr)
    print(len(fr), 'standen, max voetfout %.2f mm' % worst)
    np.savez(out, q=np.array([r['q'] for r in fr]), contact=np.array([r['contact'] for r in fr]), joint_names=np.array(J),
             params=json.dumps(kw), fps=50)
