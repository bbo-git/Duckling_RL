"""keuring.py - vaste keuring van een voorbeeld (rooster) of een netwerk in de simulatie (2026-11-27).
Elke run dezelfde maat, zodat we op cijfers beslissen in plaats van gokken.

Vanuit src/mjlab_microduck/robot/duckling:
  python ../../../../tools/keuring.py voorbeeld grids/gait_grid_stap_loop_vloei_v3.npz
  python ../../../../tools/keuring.py netwerk ../../../../../onnx/duckling_walk_250.onnx [--rooster grids/...npz]

VOORBEELD (zonder netwerk, het loopje uit het rooster herhaald, 12 s): bij servo 1.0x en 1.6x
  valt / blijft staan, voet los per pas (1.0 = schoon, 2+ = dubbele stap), vooruit (mm/s) t.o.v. het ontwerp,
  romp-roll en -pitch.
NETWERK (oefenen: 8 passen, 1.5 s stil, herhaald, 16 s; 40 ms vertraging zoals de echte eend): bij lage en hoge
  snelheid (25 % en 100 % van het rooster) en servo 1.0x / 1.6x: valt, vooruit t.o.v. de opdracht, voet los per pas,
  heup-roll (ontwerp ~6 gr), romp-roll, trillen (gem. sprong van de been-acties per 20 ms; < 0.05 is rustig),
  en in rust: wegschuiven (mm)."""
import os, sys, io, json, math, argparse, contextlib
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duckling_play_onnx as P
import duckling_servo as DS

def _kracht(k):
    base = {id(dd): dd['stall'] for dd in (DS.ES9051, DS.DIFF_HIP)}
    for dd in (DS.ES9051, DS.DIFF_HIP): dd['stall'] = base[id(dd)] * k
    return lambda: [dd.__setitem__('stall', base[id(dd)]) for dd in (DS.ES9051, DS.DIFF_HIP)]

def _contacts(r):
    m = r.m
    if not hasattr(r, '_kgl'):
        r._kgl = [[i for i in range(m.ngeom) if (m.geom(i).name or '') == s + '_foot_collision'] for s in ('left', 'right')]
    return [any(r.d.contact[j].geom1 in g or r.d.contact[j].geom2 in g for j in range(r.d.ncon)) for g in r._kgl]

def _rp(r):
    R = np.zeros(9); P.mujoco.mju_quat2Mat(R, r.d.qpos[3:7]); R = R.reshape(3, 3)
    return math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(-math.asin(np.clip(R[2, 0], -1, 1)))

def voorbeeld(grid, T=12.0):
    d = np.load(grid, allow_pickle=True)
    S = np.asarray(d['speeds']); i = int(np.argmax(S > 0.05)); wi = int(np.argmin(np.abs(np.asarray(d['turns']))))
    L = int(d['length'][i, wi, 0]); Qg = np.asarray(d['joint_pos'][i, wi, 0, :L]); names = [str(n) for n in d['joint_names']]
    Cg = np.asarray(d['contact'][i, wi, 0, :L]) > 0.5
    cyc = float(d['cycle_s'][i, wi, 0]); v_ontw = float(d['stap_v']) if 'stap_v' in d else float(d['vel'][i, wi, 0][0])
    print('VOORBEELD %s: cyclus %.2f s (%.2f s per pas), ontwerp %.1f mm/s' % (os.path.basename(grid), cyc, cyc / 2, v_ontw * 1000))
    res = {}
    for k in (1.0, 1.6):
        herstel = _kracht(k)
        with contextlib.redirect_stdout(io.StringIO()):
            r = P.Duckling('scene_walk.xml'); r.gait_clock = False; r.residual = False
        idx = [names.index(n) for n in r.names]
        Q = Qg[:, idx] - r.home
        seq = [np.zeros_like(Q[0])] * 50 + [Q[0] * (j + 1) / 25 for j in range(25)]
        n_cyc = int(T / cyc) + 1; seq += list(np.tile(Q, (n_cyc, 1)))
        C = []; RP = []; fell = None; x0 = None
        with contextlib.redirect_stdout(io.StringIO()):
            for kk, q in enumerate(seq):
                r.step(q, None)
                if kk == 75: x0 = r.d.qpos[0]
                C.append(_contacts(r)); RP.append(_rp(r))
                if r.fallen: fell = kk / 50; break
        herstel()
        C = np.array(C[75:]); RP = np.array(RP[75:])
        CD = np.tile(Cg, (n_cyc, 1))[:len(C)]
        half = L // 2; n_m = min(len(C), 8 * half); meet = C[-n_m:]; md = CD[len(C) - n_m:]      # laatste 8 passen
        los = sum(1 for j in range(1, len(meet)) for s in (0, 1) if meet[j-1, s] and not meet[j, s]) / max(1, n_m / half)
        extra = sum(1 for j in range(1, len(meet)) for s in (0, 1) if meet[j-1, s] and not meet[j, s] and md[j, s]) / max(1, n_m / half)
        dur = len(C) / 50
        res[k] = dict(valt=fell, los=los, extra=extra, v=(r.d.qpos[0] - x0) / max(dur, 1e-3) * 1000 if x0 is not None else 0,
                      roll=(RP[:, 0].min(), RP[:, 0].max()), pitch=(RP[:, 1].min(), RP[:, 1].max()))
        o = res[k]
        print('  servo %.1fx: %-12s voet los/pas %.1f, waarvan buiten de zwaai %.1f | vooruit %5.1f mm/s (%3.0f %% van ontwerp) | romp roll %+.0f..%+.0f pitch %+.0f..%+.0f' % (
            k, ('VALT %.1f s' % fell) if fell else 'staat', o['los'], o['extra'], o['v'], 100 * o['v'] / max(v_ontw * 1000, 1e-3), *o['roll'], *o['pitch']))
    ok = all(not o['valt'] and o['extra'] <= 0.3 for o in res.values())
    print('  OORDEEL: %s' % ('GOED (staat, voeten alleen los in de zwaai)' if ok else 'NIET GOED (valt of voet los buiten de zwaai = dubbele stap)'))
    return ok

def netwerk(onnx, grid, T=16.0, lag=2):
    P.STYLES[:] = [(n_, grid) for n_, _ in P.STYLES]
    d = np.load(grid, allow_pickle=True)
    f_hi = round(float(d['stap_v']) / float(d['v100']), 3) if 'stap_v' in d else 0.3
    S_ = np.asarray(d['speeds']); i_ = int(np.argmax(S_ > 0.05)); w_ = int(np.argmin(np.abs(np.asarray(d['turns']))))
    Lg = int(d['length'][i_, w_, 0]); Cg = np.asarray(d['contact'][i_, w_, 0, :Lg]) > 0.5
    print('NETWERK %s op %s (opdracht laag %.3f / hoog %.3f, vertraging %d ms)' % (os.path.basename(onnx), os.path.basename(grid), f_hi * 0.25, f_hi, lag * 20))
    legs = None
    for f in (round(f_hi * 0.25, 3), f_hi):
        for k in (1.0, 1.6):
            herstel = _kracht(k)
            with contextlib.redirect_stdout(io.StringIO()):
                pol = P.Policy(onnx); P.REST_MODE[0] = False
                grids = [P.GridClock(p) for _, p in P.STYLES]
                r = P.Duckling('scene_walk.xml', sliders=False, grid=grids[0]); r.residual = True; r.gait_clock = True
                r.styled = pol.n_in == 63 + len(P.STYLES)
                if r.styled: r.grids = grids; r.set_style(0)
                o = P.Oefen(8, f, 1.5, r.grid.v100)
            r.lag = lag
            if legs is None: legs = [j for j, n in enumerate(r.names) if not n.startswith(('neck', 'head'))]
            hr = [j for j, n in enumerate(r.names) if n.endswith('hip_roll')]
            dt = P.TIMESTEP * P.DECIMATION
            A = []; MV = []; C = []; RP = []; X = []; HR = []; PH = []; fell = None
            with contextlib.redirect_stdout(io.StringIO()):
                for kk in range(int(T / dt)):
                    tw = np.array([o.vx(r.phase, dt), 0, 0]); a = pol(r.obs(tw, np.zeros(4))); r.step(a, tw)
                    A.append(a); MV.append(tw[0] > 0); C.append(_contacts(r)); RP.append(_rp(r)); X.append(r.d.qpos[0])
                    HR.append(np.degrees(r.d.qpos[[r.qadr[j] for j in hr]])); PH.append(r.phase)
                    if r.fallen: fell = kk * dt; break
            herstel()
            A = np.array(A); MV = np.array(MV); C = np.array(C); RP = np.array(RP); X = np.array(X); HR = np.array(HR); PH = np.array(PH)
            steps = sum(1 for j in range(1, len(PH)) if MV[j] and int(PH[j] * 2) != int(PH[j-1] * 2))
            los = sum(1 for j in range(1, len(C)) for s in (0, 1) if C[j-1, s] and not C[j, s] and MV[j]) / max(1, steps)
            extra = sum(1 for j in range(1, len(C)) for s in (0, 1) if C[j-1, s] and not C[j, s] and MV[j]
                        and Cg[min(int(PH[j] * Lg), Lg - 1), s]) / max(1, steps)
            t_mv = MV.sum() * dt
            v = 0.0
            for j in range(1, len(X)):
                if MV[j]: v += X[j] - X[j-1]
            v = v / max(t_mv, 1e-3) * 1000; v_cmd = f * float(d['v100']) * 1000
            tril = np.abs(np.diff(A[:, legs], axis=0))[MV[1:]].mean() if MV[1:].any() else 0
            drift = 0.0
            for j in range(1, len(X)):
                if not MV[j]: drift += abs(X[j] - X[j-1])
            hr_ = HR[MV] if MV.any() else HR
            print('  snelheid %.3f servo %.1fx: %-11s vooruit %5.1f mm/s (opdracht %4.1f) | voet los/pas %.1f (buiten zwaai %.1f) | heup-roll %+.0f..%+.0f | romp roll %+.0f..%+.0f | trillen %.3f | rust-schuiven %.0f mm' % (
                f, k, ('VALT %.1f s' % fell) if fell else 'staat', v, v_cmd, los, extra, hr_.min(), hr_.max(), RP[:, 0].min(), RP[:, 0].max(), tril, drift * 1000))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('wat', choices=['voorbeeld', 'netwerk'])
    ap.add_argument('bestand')
    ap.add_argument('--rooster', default=None, help='bij netwerk: rooster (standaard = dat van de player)')
    ap.add_argument('--vertraging', type=int, default=2, help='stappen van 20 ms (echte eend ~2)')
    a = ap.parse_args()
    if a.wat == 'voorbeeld':
        voorbeeld(a.bestand)
    else:
        netwerk(a.bestand, a.rooster or P.STYLES[0][1], lag=a.vertraging)
