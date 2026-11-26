"""stap_gen.py - pasjes opgebouwd uit standen die (bijna) in balans zijn (2026-11-26, Jairo).
Een pas (rechts zwaait, links staat):
  1 VERSCHUIVEN (dubbele steun, rustig): romp kantelt via heup-roll naar de standvoet, kop leunt mee, zwaartepunt naar
    ~10 mm opzij (net binnen de grens van platte zolen) en een paar mm naar voren boven de standvoet.
  2 ZWAAIEN (kort): voet op, naar voren, neer. Hij hangt hier ~2 mm over de zoolrand en 'valt' langzaam naar de
    zwaaikant; de neergezette voet vangt dat op.
  3 TERUG (dubbele steun, rustig): romp en zwaartepunt terug naar het midden tussen de voeten.
  4 STAAN: stilstaan in balans.
Een cyclus = pas rechts + pas links. Uitvoer: gewrichtshoeken (JOINT_ORDER van gait_gen) per 20 ms.
Gebruik (vanuit src/mjlab_microduck/robot/duckling): python ../../../../tools/stap_gen.py ['{"swing_s":0.3}'] [uit.npz]
Bekijken: python ../../../../tools/stap_speler.py"""
# Oplosser: statische standen met platte zolen: romp (x,y,z,roll) + benen (+ kop-roll) zo dat beide voeten exact op
# hun doel staan (plat, recht vooruit) en het zwaartepunt op een doel-xy ligt.
import os, sys, math, numpy as np, mujoco
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gait_gen as G
kin = G.DuckKinematics(os.environ.get('DUCK_SCENE', 'scene_walk.xml')); m, d = kin.m, kin.d
J = G.JOINT_ORDER
qi = {n: i for i, n in enumerate(J)}
dof_of = {n: m.jnt_dofadr[m.joint(n).id] for n in J}
LEG = [f"{s}_{j}" for s in ("left", "right") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
site = {s: kin.legs[s]["site"] for s in ("left", "right")}
toe = {s: kin.legs[s]["toe"] for s in ("left", "right")}
lo = {n: m.jnt_range[m.joint(n).id][0] for n in J}; hi = {n: m.jnt_range[m.joint(n).id][1] for n in J}
H0 = 0.0592
home_feet = {s: kin.legs[s]["home_site"].copy() for s in ("left", "right")}
TRUNK = kin.trunk

def euler(roll, pitch=0.0, yaw=0.0):
    q = np.zeros(4); mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz"); return q

def com_xy():
    mujoco.mj_comPos(m, d); return d.subtree_com[1][:2].copy()

def solve(feet, com_t, contact, q0=None, base0=None, head_free=True, w_com=1.0, iters=200, pitch=0.0, z_t=H0, w_z=0.3, w_reg=0.002, foot_rot=None):
    """feet: {side: target site pos}; com_t: xy; contact: which feet carry weight (for reporting).
    unknowns: base x,y,z, base roll, leg joints, (head_roll)."""
    q = kin.home.copy() if q0 is None else q0.copy()
    b = np.array([0, 0, H0, 0.0]) if base0 is None else base0.copy()   # x,y,z,roll
    names = LEG + (["head_roll"] if head_free else [])
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv)); jcom = np.zeros((3, m.nv))
    for it in range(iters):
        d.qpos[:] = 0; d.qpos[:3] = b[:3]; d.qpos[3:7] = euler(b[3], pitch); d.qpos[kin.qadr] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        rows, errs = [], []
        cols = [0, 1, 2, 3] + [dof_of[n] for n in names]          # base dofs: 0..2 trans, 3 = rot x (roll) in world? (free joint: angular in local frame)
        for s, tgt in feet.items():
            mujoco.mj_jacSite(m, d, jacp, None, site[s]); mujoco.mj_jacBody(m, d, None, jacr, toe[s])
            R = d.xmat[toe[s]].reshape(3, 3)
            Rt = foot_rot.get(s, np.eye(3)) if foot_rot else np.eye(3)
            e_rot = R @ G._rotvec(R.T @ Rt)   # rotatiefout in wereldframe (voet plat, of gekanteld voor de zwaaivoet)
            rows += [jacp[:, cols], jacr[:, cols] * 0.05]; errs += [tgt - d.site_xpos[site[s]], e_rot * 0.05]
        mujoco.mj_jacSubtreeCom(m, d, jcom, 1)
        rows.append(jcom[:2, cols] * w_com); errs.append((com_t - d.subtree_com[1][:2]) * w_com)
        # zacht: romphoogte = HOME, kop-roll en heup-yaw liever 0 (alleen gebruiken als het moet)
        nc = len(cols)
        rz = np.zeros(nc); rz[2] = 1.0; rows.append(rz[None] * w_z); errs.append(np.array([(z_t - b[2]) * w_z]))
        for k, n in enumerate(names):
            if n in ("head_roll", "left_hip_yaw", "right_hip_yaw"):
                rr = np.zeros(nc); rr[4 + k] = 1.0; rows.append(rr[None] * w_reg); errs.append(np.array([(0 - q[qi[n]]) * w_reg]))
        # zachte regularisatie: kop-roll en hip-yaw liever 0, hoogte liever H0
        A = np.vstack(rows); e = np.concatenate(errs)
        reg = np.zeros(len(cols)); 
        dq = np.linalg.solve(A.T @ A + 1e-7 * np.eye(len(cols)), A.T @ e)
        b[:3] += dq[:3]; b[3] += dq[3]
        for k, n in enumerate(names):
            q[qi[n]] = np.clip(q[qi[n]] + dq[4 + k], lo[n], hi[n])
        if np.linalg.norm(dq) < 1e-8: break
    # rapport
    d.qpos[:] = 0; d.qpos[:3] = b[:3]; d.qpos[3:7] = euler(b[3], pitch); d.qpos[kin.qadr] = q
    mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    res = {}
    for s, tgt in feet.items():
        R = d.xmat[toe[s]].reshape(3, 3)
        res[s] = (np.linalg.norm(tgt - d.site_xpos[site[s]]) * 1000, math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(-math.asin(np.clip(R[2, 0], -1, 1))))
    return q, b, res, d.subtree_com[1][:2].copy()


import json

kin._set(np.array([0, 0, H0]), np.array([1.0, 0, 0, 0]), kin.home); mujoco.mj_comPos(m, d)
COM_DX = d.subtree_com[1][0] - home_feet['left'][0]       # HOME: zwaartepunt t.o.v. de voeten (x)

HEEL_DX = 0.0171   # m: achterrand van de zool achter de voet-site
def Ry_(th):
    return np.array([[math.cos(th), 0, math.sin(th)], [0, 1.0, 0], [-math.sin(th), 0, math.cos(th)]])
def hiel_site(p_flat, th):
    """site-positie bij voetkanteling th (teen op = negatief) zo, dat de hiel op dezelfde plek blijft als bij een
    platte voet op p_flat."""
    h = kin.site_above_floor['left']
    v = np.array([HEEL_DX, 0.0, h])
    heel = p_flat - v
    return heel + Ry_(th) @ v

TOE_DX = 0.0170    # m: voorrand van de zool voor de voet-site
def teen_site(p_flat, th):
    """site-positie bij voetkanteling th (teen omlaag = positief) rond de voorrand van de zool (afzetten)."""
    h = kin.site_above_floor['left']
    v = np.array([-TOE_DX, 0.0, h])
    toe = p_flat - v
    return toe + Ry_(th) @ v

def herm(t, t0, t1, p0, p1, m0, m1):
    """kubische Hermite op [t0, t1] met hellingen m0, m1 (per eenheid t)."""
    h = t1 - t0; x = (t - t0) / h
    return ((2*x**3 - 3*x**2 + 1) * p0 + (x**3 - 2*x**2 + x) * h * m0 + (-2*x**3 + 3*x**2) * p1 + (x**3 - x**2) * h * m1)

def mj(u):  # min-jerk 0..1
    u = np.clip(u, 0, 1); return u**3 * (10 - 15*u + 6*u*u)

def maak(shift_s=0.5, swing_s=0.24, back_s=0.5, hold_s=1.0, half=5.0e-3, lift=0.020, com_y=0.0105, com_x_ahead=-0.002, fall_x=0.0, fall_y=0.6, land_pow=2.0, rise=0.3, strek=0.003, teen=25.0, hiel=12.0, rol=0.2, afzet=0.35, afzet_deg=15.0, top=0.3, stand_rol=0.0, voor=0.0, afzet_exp=1.4, stand_duw=0.0, duw_uit=0.25, breed=0.0,
         head_roll=True, fps=50):
    """half = halve paslengte: in rust staan de voeten +-half verspringend; elke zwaai verplaatst de voet 4*half."""
    y0 = home_feet['left'][1]; xs = home_feet['left'][0]; z0 = home_feet['left'][2]
    frames = []
    feet = {'left': np.array([xs + half, y0, z0]), 'right': np.array([xs - half, -y0, z0])}   # links voor
    foot_th = {'left': 0.0, 'right': 0.0}   # voetkanteling rond de teen (hiel op) van een voet die op de grond staat
    body_x = xs                                   # midden tussen de voeten (x)
    q, b = kin.home.copy(), np.array([0, 0, H0, 0.0])
    def frame(f, cxy, contact, z=H0, rot=None):
        nonlocal q, b
        q, b, res, c = solve({k: v.copy() for k, v in f.items()}, cxy, None, q0=q, base0=b, iters=60, z_t=z, foot_rot=rot)
        frames.append(dict(q=q.copy(), b=b.copy(), com=c.copy(), feet={k: v.copy() for k, v in f.items()},
                           contact=contact, res=res))
    for stand, swing in (('left', 'right'), ('right', 'left')):
        sgn = 1 if stand == 'left' else -1
        mid = np.array([(feet['left'][0] + feet['right'][0]) / 2 + COM_DX, 0.0])  # rust: zwaartepunt als in HOME t.o.v. de voeten
        tgt = np.array([feet[stand][0] + com_x_ahead, sgn * com_y])
        n = int(shift_s * fps)
        start = feet[swing].copy(); end = start.copy(); end[0] += 4 * half
        end[1] = (-sgn) * (y0 + breed)   # 2026-11-26: zwaaivoet landt iets breder: vangt het zijwaarts vallen eerder op
        push = 0.0
        T_all = shift_s + swing_s
        x_end = tgt[0] + 2 * half * fall_x                      # zwaartepunt-x aan het eind van de zwaai
        def cx(t):   # 2026-11-26 (Jairo: geen vertraging): x loopt in EEN beweging door van verschuiven tot landen
            return mid[0] + mj(t / T_all) * (x_end - mid[0])
        Tp = max(afzet * shift_s, 0.02); P = math.radians(afzet_deg)
        for k in range(n):
            ul = (k + 1) / n
            u = (1 - math.cos(math.pi * ul)) / 2   # cosinus: geen stilstand bij het omkeerpunt (min-jerk remde af)
            # 2026-11-26 (Jairo: geen vertraging): afzetten + zwaaien = EEN voetbaan zonder stilstand. Afzetten: hiel op,
            # draaien om de voorrand van de zool, versnellend (kwadratisch) zodat de zwaai met snelheid begint.
            t_in = ul * shift_s - (shift_s - Tp)
            th0 = foot_th[swing]                   # 2026-11-26: achterste voet staat al op de teen (afgerold)
            push = th0 + (P - th0) * (t_in / Tp) ** afzet_exp if t_in > 0 else th0   # exp < 2: hiel komt eerder op
            f = dict(feet); f[swing] = teen_site(start, push)
            frame(f, np.array([cx(ul * shift_s), mid[1] + u * (tgt[1] - mid[1])]), (1, 1), H0 + strek * u, {swing: Ry_(push)})
        foot_th[swing] = 0.0
        # zwaai: begint met de stand EN de snelheid van het afzetten (Hermite-krommen)
        v_h = np.array([HEEL_DX, 0.0, kin.site_above_floor['left']])
        def heel_ref(th):        # hielpunt-referentie (zoals hiel_site verwacht) tijdens het afzetten
            return teen_site(start, th) + v_h - Ry_(th) @ v_h
        p0 = heel_ref(P)
        dth = afzet_exp * P / Tp                                   # rad/s aan het eind van het afzetten
        v0 = (heel_ref(P + 1e-4) - heel_ref(P - 1e-4)) / 2e-4 * dth * swing_s   # m per u
        m0 = dth * swing_s
        peak = max(P, math.radians(teen))
        n = int(swing_s * fps)
        for k in range(n):
            u = (k + 1) / n
            p = np.array([herm(u, 0, 1, p0[0], end[0], v0[0], 0.0), end[1], 0.0])
            zt = z0 + lift
            p[2] = herm(u, 0, rise, p0[2], zt, v0[2], 0.0) if u < rise else herm(u, rise, 1, zt, end[2], 0.0, 0.0)
            f = dict(feet)
            # zwaartepunt gaat verder naar voren (vallen in de pas); opzij valt hij al terug naar de zwaaikant
            fy = (1 - math.cos(math.pi * u)) / 2
            c = np.array([cx(shift_s + u * swing_s), tgt[1] - sgn * com_y * fall_y * fy])
            # voetkanteling: doorlopen vanuit het afzetten (teen omlaag, meer knie), dan teen op voor de hiel-landing
            th = herm(u, 0, top, P, peak, m0, 0.0) if u < top else herm(u, top, 0.8, peak, -math.radians(hiel), 0.0, 0.0) if u < 0.8 else -math.radians(hiel)   # teen op ruim voor het landen (servo's lopen achter)
            f[swing] = hiel_site(p, th)
            # 2026-11-26 (Jairo: gewicht bij het landen): standvoet duwt kort af over de teen rond het landen, zodat het
            # gewicht geleidelijk naar de landende voet gaat
            sd = math.radians(stand_duw) * mj((u - 0.6) / 0.4)
            f[stand] = teen_site(feet[stand], sd)
            frame(f, c, (1, 0) if stand == 'left' else (0, 1), H0 + strek, {swing: Ry_(th), stand: Ry_(sd)})   # standbeen gestrekt
        feet[swing] = end
        # 2026-11-26 (Jairo: de teen van het standbeen moet afrollen): na het landen gaat het gewicht naar de voorste voet
        # en rolt de achterste (de standvoet van deze pas) af over de teen: hiel op; zo staat hij klaar voor de volgende pas
        newmid = np.array([(feet['left'][0] + feet['right'][0]) / 2 + COM_DX + voor, 0.0])
        tr = math.radians(stand_rol)
        c0 = frames[-1]['com'].copy()
        n = int(back_s * fps)
        for k in range(n):
            u = mj((k + 1) / n)
            # hiel staat, voet rolt af (teen ontspant) in het eerste deel van 'terug'
            th = -math.radians(hiel) * (1 - mj((k + 1) / max(1, rol * n)))
            ts = tr * u + math.radians(stand_duw) * (1 - mj((k + 1) / max(1, duw_uit * fps)))   # afduwen ebt weg
            f = dict(feet); f[swing] = hiel_site(end, th); f[stand] = teen_site(feet[stand], ts)
            frame(f, c0 + u * (newmid - c0), (1, 1), H0 + strek * (1 - u), {swing: Ry_(th), stand: Ry_(ts)})
        foot_th[stand] = tr
        for k in range(int(hold_s * fps)):
            f = dict(feet); f[stand] = teen_site(feet[stand], tr)
            frame(f, newmid, (1, 1), H0, {stand: Ry_(tr)})
    return frames

if __name__ == "__main__":
    kw = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = sys.argv[2] if len(sys.argv) > 2 else 'stap_frames.npz'
    fr = maak(**kw)
    print(len(fr), 'standen')
    worst = max(max(r['res'][s][0] for s in r['res']) for r in fr)
    roll = max(max(abs(r['res'][s][1]) for s in r['res']) for r in fr)
    print('max voetfout %.2f mm, max zoolkanteling %.2f gr' % (worst, roll))
    for k in range(0, len(fr), 5):
        r = fr[k]
        print('%3d c%s zw.punt x%+5.1f y%+5.1f | romp roll %+5.1f z %.1f | heup-roll L%+4.0f R%+4.0f kop-roll %+3.0f | zolen roll L%+.1f R%+.1f' % (
            k, r['contact'], r['com'][0]*1000, r['com'][1]*1000, math.degrees(r['b'][3]), r['b'][2]*1000,
            math.degrees(r['q'][qi['left_hip_roll']]), math.degrees(r['q'][qi['right_hip_roll']]), math.degrees(r['q'][qi['head_roll']]),
            r['res']['left'][1], r['res']['right'][1]))
    np.savez(out if out.endswith('.npz') else out.rsplit('.', 1)[0] + '.npz', q=np.array([r['q'] for r in fr]),
             contact=np.array([r['contact'] for r in fr]), joint_names=np.array(J), params=json.dumps(kw), fps=50)
