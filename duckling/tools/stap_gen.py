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

def solve(feet, com_t, contact, q0=None, base0=None, head_free=True, w_com=1.0, iters=200, pitch=0.0, z_t=H0, w_z=0.3, w_reg=0.002):
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
            e_rot = G._rotvec(R.T @ np.eye(3)) ; e_rot = R @ e_rot   # world-frame rotation error to identity
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

def mj(u):  # min-jerk 0..1
    u = np.clip(u, 0, 1); return u**3 * (10 - 15*u + 6*u*u)

def maak(shift_s=0.5, swing_s=0.24, back_s=0.5, hold_s=1.0, half=5.0e-3, lift=0.012, com_y=0.0105, com_x_ahead=0.001, fall_x=0.8, fall_y=0.6, land_pow=2.0, rise=0.3, strek=0.003,
         head_roll=True, fps=50):
    """half = halve paslengte: in rust staan de voeten +-half verspringend; elke zwaai verplaatst de voet 4*half."""
    y0 = home_feet['left'][1]; xs = home_feet['left'][0]; z0 = home_feet['left'][2]
    frames = []
    feet = {'left': np.array([xs + half, y0, z0]), 'right': np.array([xs - half, -y0, z0])}   # links voor
    body_x = xs                                   # midden tussen de voeten (x)
    q, b = kin.home.copy(), np.array([0, 0, H0, 0.0])
    def frame(f, cxy, contact, z=H0):
        nonlocal q, b
        q, b, res, c = solve({k: v.copy() for k, v in f.items()}, cxy, None, q0=q, base0=b, iters=60, z_t=z)
        frames.append(dict(q=q.copy(), b=b.copy(), com=c.copy(), feet={k: v.copy() for k, v in f.items()},
                           contact=contact, res=res))
    for stand, swing in (('left', 'right'), ('right', 'left')):
        sgn = 1 if stand == 'left' else -1
        mid = np.array([(feet['left'][0] + feet['right'][0]) / 2 + COM_DX, 0.0])  # rust: zwaartepunt als in HOME t.o.v. de voeten
        tgt = np.array([feet[stand][0] + com_x_ahead, sgn * com_y])
        n = int(shift_s * fps)
        for k in range(n):
            u = mj((k + 1) / n); frame(feet, mid + u * (tgt - mid), (1, 1), H0 + strek * u)
        # zwaai
        start = feet[swing].copy(); end = start.copy(); end[0] += 4 * half
        n = int(swing_s * fps)
        land = None
        for k in range(n):
            u = (k + 1) / n
            p = start + mj(u) * (end - start)
            # 2026-11-26 (Jairo: snellere voetlift): snel op in de eerste 'rise' van de zwaai, dan zacht neer
            p[2] = z0 + lift * (mj(u / rise) if u < rise else 1 - mj((u - rise) / (1 - rise)) ** 1.0)
            f = dict(feet); f[swing] = p
            # zwaartepunt gaat verder naar voren (vallen in de pas), opzij blijft
            # 2026-11-26 (dubbele bounce): het zwaartepunt valt tijdens de zwaai al terug naar de zwaaikant (zoals het
            # in de fysica echt gaat); anders duwt de romp hem na het neerzetten terug en komt de voet nog eens los
            c = tgt + np.array([mj(u) * 2 * half * fall_x, -sgn * com_y * fall_y * mj(u)])
            frame(f, c, (1, 0) if stand == 'left' else (0, 1), H0 + strek)   # standbeen gestrekt
        feet[swing] = end
        newmid = np.array([(feet['left'][0] + feet['right'][0]) / 2 + COM_DX, 0.0])
        c0 = frames[-1]['com'].copy()
        n = int(back_s * fps)
        for k in range(n):
            u = mj((k + 1) / n); frame(feet, c0 + u * (newmid - c0), (1, 1), H0 + strek * (1 - u))
        for k in range(int(hold_s * fps)):
            frame(feet, newmid, (1, 1))
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
