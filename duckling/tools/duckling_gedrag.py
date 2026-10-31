r"""
duckling_gedrag.py - het GEDRAG van de Duckling bekijken en uitproberen (2026-10-29).

Een rode bal is de persoon (het saillante punt dat de camera ziet). De bal beweeg je met
de PIJLTOETSEN (klik eerst in het 3D-beeld):

    pijl op / neer       bal van je af / naar je toe (t.o.v. jouw kijkrichting), 3 cm per tik
    pijl links / rechts  bal naar links / rechts in beeld
    End                  bal 50 cm recht voor de robot
    Home                 camera terug op robot + bal
    (vasthouden = blijft schuiven; de camera draai/zoom je gewoon met de muis)

Wat de camera aan beweging ziet en wat de persoon doet stel je in met de SCHUIFJES
onderaan het Control-paneel (rechts, na de 14 motoren):

    intensiteit  0 geen, 1 licht, 2 middel, 3 sterk
    grootte      0 geen, 1 klein (hand), 2 middel, 3 groot (lijf) - op 50 cm; dichterbij
                 ziet de camera hem groter (schijnbare grootte staat linksboven)
    wandel       0 bal staat stil, 1 loopt weg van de robot, 2 loopt rondjes om de robot,
                 3 komt naar de robot toe (tot 20 cm)
    stijl        0 normaal, 1 parmantig
    uit_beeld    1 = persoon weg
    Backspace    reset
(De cijfertoetsen gebruikt de MuJoCo-viewer zelf: die zetten onderdelen aan/uit.)

De nep-ESP rekent hieruit het protocol (richting, afstand, grootte, intensiteit, eigen
beweging van de bal, nadering) en de toestandsmachine (duckling_behavior.py - dezelfde code
als in de training en straks op de RP2350) beslist: RUST (zitten), KIJKEN, NADEREN, VOLGEN,
ORBIT, TERUG, ZOEKEN. Linksboven staat alles wat er gebeurt.

    python ..\..\..\..\tools\duckling_gedrag.py                      # zonder netwerk: voorbeeld
    python ..\..\..\..\tools\duckling_gedrag.py duckling_walk_XXXX.onnx   # met het loop-netwerk

Zonder ONNX loopt hij "kinematisch": benen uit het rooster van de stijl, romp volgt het
commando precies (geen fysica). Zo zie je het gedrag nu al, vóór de training.
Met ONNX: echte fysica, het netwerk loopt; in rust (commando 0) fadet het netwerk uit en
staat de robot in HOME, of zit (hurkt) als de toestand RUST is.
"""
import argparse
import importlib.util
import math
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import duckling_play_onnx as P          # noqa: E402  (Duckling, Policy, roosters, timing)
import walk_speed as W                  # noqa: E402  (rooster afspelen, kinematisch)


def _load_behavior():
    path = os.path.join(HERE, "..", "src", "mjlab_microduck", "tasks", "duckling_behavior.py")
    spec = importlib.util.spec_from_file_location("duckling_behavior", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _load_behavior()
DT = P.TIMESTEP * P.DECIMATION
BALL_H = 0.20                 # m: hoogte van het saillante punt (ongeveer heup/knie)
CAM_OFF = np.array([0.032, 0.0025, 0.0104])   # ESP32-CAM in het kopframe
HEAD_LIMITS = np.array([0.25, 0.25, 1.10, 0.15])   # = HEAD_CAPS training (nek, kop-pitch, kop-yaw, kop-roll)
# hurken (sit_gen.py standaard: romp 44 mm, 10 gr voorover)
CROUCH = {"hip_pitch": 0.1934, "knee": -0.6377, "ankle": -0.2699}
SIT_S = 0.5                   # s gaan zitten (2026-10-29: 3x sneller)
SIT_UP_S = 0.8                # s opstaan
FADE_OUT, FADE_IN = 1.0, 0.25


def min_jerk(u):
    u = min(max(u, 0.0), 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


class FakeESP:
    """Wat de ESP32-CAM over de bal zou sturen (zelfde velden als PersonSim in de training)."""

    def __init__(self):
        self.mint = 0.0
        self.msize = 0.0
        self.absent = False
        self.vel_lp = np.zeros(2)
        self.prev = None
        self.buf = []

    def __call__(self, ball, cam_pos, cam_R):
        if self.prev is None:
            self.prev = ball.copy()
        vel = (ball[:2] - self.prev[:2]) / DT
        self.prev = ball.copy()
        self.vel_lp += min(1.0, DT / 0.8) * (vel - self.vel_lp)
        rel_w = ball - cam_pos
        rel = cam_R.T @ rel_w
        d = max(np.hypot(rel_w[0], rel_w[1]), 1e-3)
        bx = math.atan2(rel[1], rel[0])
        by = math.atan2(rel[2], max(np.hypot(rel[0], rel[1]), 1e-3))
        vis = rel[0] > 0 and abs(bx) < B.FOV_H and abs(by) < B.FOV_V and d < B.CAM_RANGE and not self.absent
        u = rel_w[:2] / d
        tflow = (u[0] * self.vel_lp[1] - u[1] * self.vel_lp[0]) / d
        approach = -float(u @ self.vel_lp)
        vis = vis and (self.mint > 0 or self.msize > 0)      # niets beweegt = niets in beeld
        ms = float(np.clip(round(self.msize * B.APPARENT_D / d), 1, 3)) if self.msize > 0 else 0.0
        self.msize_seen = ms
        frame = dict(vis=float(vis), bx=bx, by=by, dist=d, msize=ms if vis else 0.0,
                     mint=self.mint if vis else 0.0, tflow=tflow, approach=approach)
        self.buf.append(frame)
        out = self.buf.pop(0) if len(self.buf) > B.ESP_DELAY else self.buf[0]
        return {k: np.array([v], dtype=float) for k, v in out.items()}


SLIDERS = (("intensiteit", 0, 3), ("grootte", 0, 3), ("wandel", 0, 3), ("stijl", 0, 1), ("uit_beeld", 0, 1))
BALL_STEP = 0.03              # m per pijltoets-tik
WALK_V = 0.06                 # m/s: wandelsnelheid van de bal


def build_model(scene):
    spec = mujoco.MjSpec.from_file(scene)
    b = spec.worldbody.add_body(name="bal", mocap=True, pos=[0.5, 0, BALL_H])
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], rgba=[0.9, 0.1, 0.1, 1],
               contype=0, conaffinity=0)
    for name, lo, hi in SLIDERS:          # schuifjes in het Control-paneel (geen kracht)
        sb = spec.worldbody.add_body(name="_b_" + name, pos=[0, 0, -1], gravcomp=1.0)
        sb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0,
                    rgba=[0, 0, 0, 0], mass=1e-3)
        sb.add_joint(name="_" + name, type=mujoco.mjtJoint.mjJNT_SLIDE, range=[lo, hi], damping=1.0)
        act = spec.add_actuator(name=name, target="_" + name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.ctrlrange = [lo, hi]; act.ctrllimited = 1; act.gear[0] = 0.0
    return spec.compile()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", nargs="?", default=None, help="loop-netwerk (leeg = kinematisch voorbeeld)")
    ap.add_argument("--scene", default="scene_walk.xml")
    ap.add_argument("--stijl", default=P.STYLES[0][0], choices=[n for n, _ in P.STYLES])
    ap.add_argument("--mooi", action="store_true")
    ap.add_argument("--zit", default="favorites/zit_v2.json",
                    help="zithouding uit sit_gen.py (S slaat sit_motion.npz + .json op); ontbreekt = standaard hurken")
    ap.add_argument("--kop-vol", action="store_true",
                    help="volle kopbereik, ook als het netwerk dat in de training nog niet kreeg")
    ap.add_argument("--test", type=float, default=0.0, help="zonder viewer: zoveel s met een vast script (controle)")
    a = ap.parse_args()

    m = build_model(a.scene)
    ball_mocap = m.body("bal").mocapid[0]
    robot = P.Duckling(a.scene, model=m)
    head_id = m.body("head").id
    grids = [P.GridClock(p) for _, p in P.STYLES]
    full = [W.GaitGrid(np.load(p)) for _, p in P.STYLES]     # voor kinematisch afspelen
    style = [n for n, _ in P.STYLES].index(a.stijl)
    pol = P.Policy(a.onnx) if a.onnx else None
    # 2026-10-30: de training breidt het kopbereik uit (iter 0/500/1000/1500/2000: 5/15/35/65/100 %).
    # Een vroeg netwerk krijgt hier dus niet meer kop dan het kent (anders valt/draait het raar).
    head_lim = HEAD_LIMITS.copy()
    if pol is not None and not a.kop_vol:
        import re
        mm = re.search(r"_(\d+)\.onnx$", a.onnx)
        if mm:
            it = int(mm.group(1))
            frac = 1.0 if it >= 2000 else 0.65 if it >= 1500 else 0.35 if it >= 1000 else 0.15 if it >= 500 else 0.05
            head_lim = HEAD_LIMITS * frac
            print(f"kopbereik {frac:.0%} (zoals in de training op iteratie {it}; --kop-vol = alles)")
    if pol is not None:
        robot.gait_clock = pol.n_in >= 63
        robot.styled = pol.n_in == 63 + len(P.STYLES)
        robot.grids = grids
        robot.set_style(style)
    names = robot.names
    leg_cols = {k: [i for i, n in enumerate(names) if n.endswith("_" + k)] for k in CROUCH}
    head_cols = [names.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]
    crouch = np.zeros(robot.nj)
    for k, v in CROUCH.items():
        crouch[leg_cols[k]] = v
    sit_z = 0.044
    if os.path.exists(a.zit):           # jouw zithouding (benen, nek, kop) uit sit_gen.py
        import json
        import sit_gen
        sp = sit_gen.SitParams(**{k: float(v) for k, v in json.load(open(a.zit)).items()})
        q_sit, sit_z, _ = sit_gen.sit_pose(sit_gen.SitPlanner(a.scene), sp)
        order = list(sit_gen.JOINT_ORDER)
        crouch = np.array([q_sit[order.index(n)] for n in names]) - robot.home
        globals()["SIT_S"] = float(sp.down_s)
        globals()["SIT_UP_S"] = float(sp.up_s)
        print(f"zithouding: {a.zit} (romp {sit_z * 1000:.0f} mm, zakken {sp.down_s:.1f} s, opstaan {sp.up_s:.1f} s)")

    esp = FakeESP()
    beh = B.Behavior(np.zeros(1))
    st = dict(posture="ZIT", t_post=0.0, rest_b=0.0, sit=1.0, phase=0.0, pos=np.zeros(2), yaw=0.0, t=0.0)
    keys = dict(style=style)

    def reset():
        robot.reset()
        beh.reset(np.array([0]))
        esp.__init__()
        st.update(posture="ZIT", t_post=0.0, rest_b=0.0, sit=1.0, phase=0.0, pos=np.zeros(2), yaw=0.0)
        ball[:] = [0.6, 0.0, BALL_H]
        robot.d.qpos[robot.qadr] = robot.home + crouch
        mujoco.mj_forward(m, robot.d)

    sl = {n: m.actuator(n).id for n, _, _ in SLIDERS}

    ball = np.array([0.6, 0.0, BALL_H])
    view = {}

    def key(k):
        if k == 259:   # backspace
            keys["reset"] = True
        elif k in (262, 263, 264, 265):          # pijltoetsen: t.o.v. de kijkrichting
            az = math.radians(view["v"].cam.azimuth) if "v" in view else 0.0
            fwd = np.array([math.cos(az), math.sin(az)]); left = np.array([-fwd[1], fwd[0]])
            d = {265: fwd, 264: -fwd, 263: left, 262: -left}[k]
            ball[:2] += BALL_STEP * d
        elif k == 269:                            # End: bal voor de robot
            yaw = 2 * math.atan2(robot.d.qpos[6], robot.d.qpos[3])
            ball[:2] = robot.d.qpos[:2] + 0.5 * np.array([math.cos(yaw), math.sin(yaw)])
        elif k == 268:                            # Home: camera op robot + bal
            keys["home_cam"] = True

    def wander(mode):
        r = ball[:2] - robot.d.qpos[:2]
        dist = max(np.linalg.norm(r), 1e-3); u = r / dist
        if mode == 1:
            ball[:2] += WALK_V * DT * u
        elif mode == 2:
            ball[:2] += WALK_V * DT * np.array([-u[1], u[0]])
            r2 = ball[:2] - robot.d.qpos[:2]
            ball[:2] = robot.d.qpos[:2] + r2 / max(np.linalg.norm(r2), 1e-3) * dist   # straal houden
        elif mode == 3 and dist > 0.20:
            ball[:2] -= WALK_V * DT * u

    def read_sliders():
        c = lambda n: int(round(float(robot.d.ctrl[sl[n]])))
        esp.mint, esp.msize = float(c("intensiteit")), float(c("grootte"))
        esp.absent = c("uit_beeld") == 1
        keys["wandel"] = c("wandel")
        stl = min(c("stijl"), len(P.STYLES) - 1)
        if stl != keys["style"]:
            keys["style"] = stl
            if pol is not None:
                robot.set_style(stl)

    def camera():
        R = robot.d.xmat[head_id].reshape(3, 3)
        return robot.d.xpos[head_id] + R @ CAM_OFF, R

    def control_step(ball):
        """Eén stap van 20 ms: ESP -> toestandsmachine -> houding -> netwerk/kinematisch."""
        cam_pos, cam_R = camera()
        p = esp(ball, cam_pos, cam_R)
        v100 = np.array([grids[keys["style"]].v100])
        vx, wz, head, fix = beh.step(p, DT, v100)
        head = np.clip(head[0], -head_lim, head_lim)
        # 2026-10-30: de begrensde kopstand terug naar de toestandsmachine (zoals in de training):
        # anders rekent hij met een kophoek die de kop niet heeft -> draait de verkeerde kant op
        beh.h_yaw = np.array([head[2]])
        beh.h_elev = np.array([head[1] - head[0]])
        state = int(beh.state[0])
        # houding: RUST = zitten; uit RUST eerst opstaan (netwerk uit, commando 0)
        want_sit = state == B.REST
        if st["posture"] == "STAAT" and want_sit and st["rest_b"] <= 0.0:
            st.update(posture="GAAT ZITTEN", t_post=0.0)
        elif st["posture"] == "ZIT" and not want_sit:
            st.update(posture="STAAT OP", t_post=0.0)
        st["t_post"] += DT
        if st["posture"] == "GAAT ZITTEN":
            st["sit"] = min_jerk(st["t_post"] / SIT_S)
            if st["t_post"] >= SIT_S:
                st["posture"] = "ZIT"
        elif st["posture"] == "STAAT OP":
            st["sit"] = 1.0 - min_jerk(st["t_post"] / SIT_UP_S)
            if st["t_post"] >= SIT_UP_S:
                st["posture"] = "STAAT"
        standing = st["posture"] == "STAAT"
        twist = np.array([float(vx[0]), 0.0, float(wz[0])]) if standing else np.zeros(3)
        moving = np.hypot(twist[0], twist[1]) > 0.02 or abs(twist[2]) > 0.1
        st["rest_b"] = float(np.clip(st["rest_b"] + (DT / FADE_IN if moving else -DT / FADE_OUT), 0.0, 1.0))
        # gewrichtsdoelen buiten het netwerk: benen HOME/hurken, kop direct uit de toestandsmachine
        offs = st["sit"] * crouch          # incl. nek/kop/scheef van de zithouding
        offs[head_cols] = offs[head_cols] + head
        if pol is not None:
            act = pol(robot.obs(twist, head)) if standing else np.zeros(robot.nj)
            b = st["rest_b"] if standing else 0.0
            robot.step(b * act + (1 - b) * offs, twist)
        else:
            g = full[keys["style"]]
            g.set(twist[0] / g.v100, math.degrees(twist[2]), 0.0)
            st["phase"] = (st["phase"] + DT / g.cycle) % 1.0
            q = g.ref_q(st["phase"])
            w = g.walk if standing else 0.0
            q = w * q + (1 - w) * (robot.home + offs)
            q[head_cols] = q[head_cols] * w + head
            rl = g.at("root_local", st["phase"]) if standing else np.array([0, 0, 0.0592])
            ql = g.at("root_quat_local", st["phase"]) if standing else np.array([1.0, 0, 0, 0])
            z = (1 - st["sit"]) * rl[2] + st["sit"] * sit_z
            c, s = math.cos(st["yaw"]), math.sin(st["yaw"])
            robot.d.qpos[:2] = st["pos"] + np.array([c * rl[0] - s * rl[1], s * rl[0] + c * rl[1]]) * w
            robot.d.qpos[2] = z
            robot.d.qpos[3:7] = W.G._rotz_quat(st["yaw"], ql / np.linalg.norm(ql))
            robot.d.qpos[robot.qadr] = q
            mujoco.mj_forward(m, robot.d)
            st["pos"] += np.array([c * twist[0], s * twist[0]]) * DT
            st["yaw"] += twist[2] * DT
        st["t"] += DT
        return p, state, twist, head, bool(fix[0])

    def hud(p, state, twist, head, fix):
        left = ("toestand\nhouding\nstijl\nESP: zichtbaar\nrichting / afstand\ngrootte / intensiteit\n"
                "eigen beweging / nadering\nnetwerk: vx / wz\nkop yaw / elevatie\n\ntoetsen\n")
        right = (f"{B.STATE_NAMES[state]}\n{st['posture']}  (netwerk {st['rest_b']:.0%})\n{P.STYLES[keys['style']][0]}\n"
                 f"{'ja' if p['vis'][0] > 0.5 else 'NEE'}{' (uit beeld)' if esp.absent else ''}\n"
                 f"{math.degrees(p['bx'][0]):+.0f} gr / {p['dist'][0] * 100:.0f} cm\n"
                 f"{int(esp.msize)} (camera ziet {int(getattr(esp, 'msize_seen', 0))}) / {int(esp.mint)}\n"
                 f"{p['tflow'][0] * p['dist'][0] * 100:+.1f} cm/s / {p['approach'][0] * 100:+.1f} cm/s\n"
                 f"{twist[0] * 100:+.1f} cm/s / {math.degrees(twist[2]):+.0f} gr/s\n"
                 f"{math.degrees(head[2]):+.0f} / {math.degrees(head[1] - head[0]):+.0f} gr{'  (fixeert)' if fix else ''}\n\n"
                 "bal: pijltoetsen, End, Home\nschuifjes: intensiteit, grootte, wandel, stijl, uit_beeld")
        return left, right

    reset()
    if a.test:
        # vast script zonder viewer (de regels van Jairo): (t, intensiteit, grootte, wat de bal doet)
        script = [(0, 0, 0, "1 m voor"), (4, 1, 1, "zijwaarts")] if not os.environ.get("DUCK_RAND") else [(0, 1, 1, "1 m voor"), (2, 1, 1, "rondje")]
        script = script if os.environ.get("DUCK_RAND") else script + [(9, 2, 2, "stil"), (24, 2, 2, "dichtbij"),
                  (32, 3, 3, "loopt weg"), (45, 0, 0, "stil")]
        last, mode = None, None
        for k in range(int(a.test / DT)):
            t = k * DT
            for ts, mi, ms, act in script:
                if abs(t - ts) < DT / 2:
                    esp.mint, esp.msize = float(mi), float(ms)
                    mode = act
                    yaw = 2 * math.atan2(robot.d.qpos[6], robot.d.qpos[3])
                    fwd = np.array([math.cos(yaw), math.sin(yaw)])
                    if act == "1 m voor":
                        ball[:2] = robot.d.qpos[:2] + 1.0 * fwd
                    elif act == "dichtbij":
                        ball[:2] = robot.d.qpos[:2] + 0.25 * fwd
                    print(f"--- t={t:4.1f}  intensiteit {mi}, grootte {ms}, bal: {act}")
            if os.environ.get("DUCK_BAL"):
                ball[:2] = [float(v) for v in os.environ["DUCK_BAL"].split(",")]
            if mode == "zijwaarts":
                ball[1] += 0.10 * DT
            elif mode == "rondje":                 # loopt langzaam om de robot heen (licht)
                wander(2)
            elif mode == "loopt weg":
                wander(1)
            robot.d.mocap_pos[ball_mocap] = ball
            p, state, twist, head, fix = control_step(ball)
            if state != last or k % 100 == 0:
                print(f"t={t:5.1f}  {B.STATE_NAMES[state]:8s} {st['posture']:11s} afstand {p['dist'][0]*100:4.0f} cm  "
                      f"richting {math.degrees(p['bx'][0]):+4.0f}  zichtbaar {int(p['vis'][0])}  vx {twist[0]*100:+5.1f} cm/s  "
                      f"wz {math.degrees(twist[2]):+4.0f} gr/s  kop {math.degrees(head[2]):+3.0f}  romp {robot.d.qpos[2]*1000:.0f} mm")
                last = state
        return

    from mujoco import viewer as mjviewer
    P.fast_model(m) if not a.mooi else None
    with mjviewer.launch_passive(m, robot.d, key_callback=key) as v:
        P.fast_render(v) if not a.mooi else None
        view["v"] = v
        with v.lock():
            v.cam.lookat[:] = [0.3, 0.0, 0.05]; v.cam.distance = 1.2; v.cam.azimuth = 200; v.cam.elevation = -35
        print(__doc__.split("Een rode bal")[0])
        t_next = time.time()
        last_state = None
        # 2026-10-31: tempo zichtbaar: echte tijd vs simulatietijd, en voetstappen per seconde
        soles = [m.geom(f"{x}_foot_collision").id for x in ("left", "right")]
        was_down = [False, False]
        steps = []                      # simulatietijden van neerzetten (beide voeten)
        wall0, sim0 = time.time(), st["t"]
        rate = {"rt": 1.0, "sps": 0.0}
        print("bal: pijltoetsen (klik eerst in het beeld), End = voor de robot, Home = camera erbij")
        while v.is_running():
            if keys.pop("reset", False):
                reset()
            read_sliders()
            wander(keys.get("wandel", 0))
            if keys.pop("home_cam", False):
                with v.lock():
                    v.cam.lookat[:2] = 0.5 * (robot.d.qpos[:2] + ball[:2]); v.cam.lookat[2] = 0.05
            robot.d.mocap_pos[ball_mocap] = ball
            p, state, twist, head, fix = control_step(ball)
            down = [False, False]
            for c in robot.d.contact[:robot.d.ncon]:
                for i, g in enumerate(soles):
                    if g in (c.geom1, c.geom2):
                        down[i] = True
            for i in range(2):
                if down[i] and not was_down[i]:
                    steps.append(st["t"])
            was_down = down
            steps = [x for x in steps if st["t"] - x < 2.0]
            if time.time() - wall0 > 1.0:
                rate["rt"] = (st["t"] - sim0) / (time.time() - wall0)
                rate["sps"] = len(steps) / 2.0
                wall0, sim0 = time.time(), st["t"]
            if state != last_state:
                print(f"{st['t']:6.1f} s  {B.STATE_NAMES[state]:8s} (intensiteit {int(esp.mint)}, grootte {int(esp.msize)}, "
                      f"afstand {p['dist'][0] * 100:.0f} cm, zichtbaar {'ja' if p['vis'][0] > 0.5 else 'nee'})")
                last_state = state
            try:
                lt, rt = hud(p, state, twist, head, fix)
                lt += "\ntempo t.o.v. echt\nvoetstappen/s (L+R)"
                rt += f"\n{rate['rt']:.2f}x\n{rate['sps']:.1f}  (jouw loopje ~4.5)"
                v.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, lt, rt))
            except Exception:
                pass
            v.sync()
            t_next += DT
            time.sleep(max(0.0, t_next - time.time()))
            if time.time() - t_next > 0.5:
                t_next = time.time()


if __name__ == "__main__":
    main()
