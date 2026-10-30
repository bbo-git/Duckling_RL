"""duckling_behavior.py - gedrag van de Duckling (2026-10-29).

Drie lagen, dezelfde code in de training (torch, duizenden envs), de player (numpy, 1 env)
en later op de RP2350 (overzetten naar C/MicroPython: alleen + - * / min max abs atan2):

1. PROTOCOL  ESP32-CAM -> RP2350 (~20 Hz). De ESP doet alle beeldbewerking (optical flow) en
   stuurt alleen dit (velden van `Percept`):
       vis       1 = er is een saillant punt (de persoon) in beeld
       bx, by    richting van het punt in het camerabeeld (rad; + = links / omhoog)
       dist      afstand (m), uit de grootte of de onderkant van de vlek
       msize     bewegings-GROOTTE 0..3   (0 geen, 1 klein = hand, 2 middel, 3 groot = hele lijf)
       mint      bewegings-INTENSITEIT 0..3 (0 geen, 1 licht, 2 middel, 3 sterk)
       tflow     eigen hoeksnelheid van het punt t.o.v. de wereld (rad/s, + = naar links);
                 de ESP haalt de eigen beweging van de camera eruit (mediaan van de flow)
       approach  naderingssnelheid (m/s, + = komt dichterbij), uit de divergentie van de flow
   RP2350 -> ESP: kop-yaw, kop-elevatie, gyro, `fixeer` (kop staat stil: nu bewegingen meten).

2. TOESTANDSMACHINE (`Behavior`) - maakt van het protocol de opdracht aan het loop-netwerk
   (vx, wz, kop: nek, kop-pitch, kop-yaw, kop-roll) en kiest wat de robot doet:
       RUST      zitten (op de robot: vaste beweging, netwerk uit); kop kijkt rond / volgt
       KIJKEN    staan, naar de persoon gericht, kop volgt (netwerk uitgefaded, cmd 0)
       NADEREN   naar ~50 cm lopen en draaien tot hij je aankijkt
       VOLGEN    persoon loopt: volgen op 50 cm
       ORBIT     persoon danst op zijn plek: rondje eromheen, kop naar binnen
       TERUG     persoon te dichtbij: achteruit
       ZOEKEN    persoon kwijt: draaien naar waar hij het laatst was, kop zoekt
   Regels (2026-10-29 avond, Jairo):
       niets beweegt (intensiteit 0) : niets in beeld -> blijft zitten, kijkt rond
       licht                         : blijft zitten, kop volgt het punt; komt het punt verder dan
                                       52 gr opzij (rand van wat de kop kan) -> opstaan en meedraaien
       middel of sterk               : erheen om te verkennen (NADEREN tot 50 cm, dan KIJKEN)
       sterk  middel/groot + punt beweegt : VOLGEN
       sterk  groot     + punt stil       : ORBIT
       te dichtbij (< 33 cm)              : TERUG (ook vanuit zitten)
       ~10 s niets -> RUST, kwijt -> ZOEKEN (5 s) -> RUST
   GROOTTE is de grootte zoals de camera hem ziet: dezelfde hand is dichtbij groter
   (schijnbaar = echte grootte x 50 cm / afstand).
   De kop beweegt als een vogel: stil fixeren, dan een snelle draai (saccade).

3. SIMULATIE (alleen training): virtuele persoon + nep-ESP met ruis, vertraging en uitval.
"""
import math

try:
    import torch
except ImportError:          # player / robot zonder torch
    torch = None
import numpy as np

# ------------------------------------------------------------------ constanten
REST, WATCH, APPROACH, FOLLOW, ORBIT, BACKOFF, SEARCH = range(7)
STATE_NAMES = ("RUST", "KIJKEN", "NADEREN", "VOLGEN", "ORBIT", "TERUG", "ZOEKEN")

FOLLOW_DIST = 0.50        # m
NEAR_OK = 0.07            # m: NADEREN klaar binnen 50 + 7 cm ...
FACE_OK = 0.15            # rad: ... en recht ervoor (9 gr)
FAR_AGAIN, SIDE_AGAIN = 0.15, 0.6   # m, rad: pas opnieuw NADEREN boven 65 cm of 34 gr opzij (hysterese)
D_TAU = 0.4               # s: afstand gefilterd (de grootte van de vlek ruist)
BACK_DIST, BACK_RELEASE = 0.33, 0.40
K_DIST, K_BEAR = 1.0, 1.5
ACC_V, ACC_W = 0.15, 1.5  # m/s2, rad/s2: commando's veranderen geleidelijk (geen geschok)
W_MAX = 0.6               # rad/s
IDLE_TO_REST = 10.0       # s zonder iets -> zitten
LOST_TO_SEARCH, SEARCH_TO_REST = 1.0, 5.0
MOVE_SPEED = 0.03         # m/s: punt "beweegt" boven deze eigen snelheid ...
MOVE_HOLD = 0.6           # ... gedurende zoveel s (ruisbestendig)
STILL_TO_WATCH = 2.0      # s persoon stil (en niet dansen) -> VOLGEN stopt
DANCE_HOLD = 1.0          # s sterk + groot + stil -> ORBIT
DANCE_END = 2.5           # s zonder dansen -> ORBIT stopt
ORBIT_R = 0.50            # m
ORBIT_BEAR = math.radians(85)   # persoon ~opzij tijdens het rondje (kop 63 gr + camera 30 gr = 93 gr zicht)
ORBIT_FRAC = 0.5          # fractie van de 100%-snelheid
HEAD_YAW_MAX = 1.1        # rad (63 gr) = HEAD_CAPS van de training (rondje: persoon op ~90 gr)
HEAD_ELEV_MAX, HEAD_ELEV_MIN = 0.5, -0.25   # rad: nek + kop samen (elk max 0.25)
SACCADE_THR = math.radians(12)   # kop draait pas als het doel > 12 gr verschoven is (was 7: te spastisch) ...
FIX_MIN = 0.6            # ... en hij minstens zo lang heeft gefixeerd (s)
SACCADE_RATE = 3.0        # rad/s: snelle kopdraai (servo ~8)
EDGE_TURN = 0.9           # rad (52 gr): zit hij en staat de actie verder opzij -> opstaan en draaien
LOOK_AMP = 1.0            # rad: rondkijken in rust/zoeken (2026-10-29: 0.6 -> 1.0, zag niets opzij)


# ------------------------------------------------------------------ numpy/torch helpers
def _t(x):
    return torch is not None and isinstance(x, torch.Tensor)


def clip(x, lo, hi):
    if _t(x):
        if _t(lo) or _t(hi):     # gemengd getal/tensor: torch.clamp wil dan twee tensors
            lo = lo if _t(lo) else torch.full_like(x, lo)
            hi = hi if _t(hi) else torch.full_like(x, hi)
        return torch.clamp(x, lo, hi)
    return np.clip(x, lo, hi)


def where(c, a, b):
    if _t(c):
        return torch.where(c, a, b)
    return np.where(c, a, b)


def atan2(y, x):
    return torch.atan2(y, x) if _t(y) else np.arctan2(y, x)


def sin(x):
    return torch.sin(x) if _t(x) else np.sin(x)


def absv(x):
    return x.abs() if _t(x) else np.abs(x)


def copy(x):
    return x.clone() if _t(x) else x.copy()


def zeros_like(x):
    return torch.zeros_like(x) if _t(x) else np.zeros_like(x)


def as_float(c):
    return c.float() if _t(c) else c.astype(float)


# ------------------------------------------------------------------ toestandsmachine
class Behavior:
    """Per env de toestand. step() krijgt het protocol (dict van arrays (N,)) en geeft de
    opdracht aan het loop-netwerk. Gebruikt alleen zijn EIGEN kopcommando als kophoek
    (de servo's hebben geen encoder)."""

    def __init__(self, template):
        """template: nul-array (N,) float - torch-tensor (training) of numpy (player/robot)."""
        z = zeros_like(template)
        self.state = copy(z) + REST       # float voor torch/numpy-gemak
        for k in ("t_state", "t_idle", "t_lost", "t_move", "t_still", "t_dance", "t_nodance",
                  "last_b", "last_d", "last_e", "vx", "wz", "h_yaw", "h_elev", "g_yaw", "g_elev",
                  "t_fix", "orbit_side", "t_look", "d_f"):
            setattr(self, k, copy(z))
        self.last_d = self.last_d + FOLLOW_DIST
        self.d_f = self.d_f + FOLLOW_DIST

    def reset(self, ids, state=REST):
        for k in ("t_state", "t_idle", "t_lost", "t_move", "t_still", "t_dance", "t_nodance",
                  "last_b", "last_e", "vx", "wz", "h_yaw", "h_elev", "g_yaw", "g_elev", "t_fix",
                  "orbit_side", "t_look"):
            getattr(self, k)[ids] = 0.0
        self.last_d[ids] = FOLLOW_DIST
        self.d_f[ids] = FOLLOW_DIST
        self.state[ids] = state

    def _go(self, cond, new):
        self.t_state = where(cond & (self.state != new), self.t_state * 0, self.t_state)
        self.state = where(cond, self.state * 0 + new, self.state)

    def step(self, p, dt, v100):
        """p: dict met de protocolvelden (arrays (N,)); v100: 100%-snelheid van de stijl (N,).
        Geeft (vx, wz, kop[N,4] = nek, kop-pitch, kop-yaw, kop-roll, fixeer)."""
        S = self.state
        vis = p["vis"] > 0.5
        # richting t.o.v. het LIJF = camera-richting + eigen kophoek
        b = p["bx"] + self.h_yaw
        e = p["by"] + self.h_elev
        d = p["dist"]
        # geheugen; buiten beeld: richting bijwerken met de eigen draai (dead reckoning)
        self.last_b = where(vis, b, self.last_b - self.wz * dt)
        self.last_d = where(vis, d, self.last_d)
        self.last_e = where(vis, e, self.last_e)
        self.t_lost = where(vis, self.t_lost * 0, self.t_lost + dt)
        self.d_f = where(vis, self.d_f + min(1.0, dt / D_TAU) * (d - self.d_f), self.d_f)
        b, d = self.last_b, self.d_f
        seen = self.t_lost < LOST_TO_SEARCH
        # wat de persoon doet
        own_v = (p["tflow"] * d) ** 2 + p["approach"] ** 2
        moving_now = vis & (own_v > MOVE_SPEED ** 2)
        self.t_move = where(moving_now, self.t_move + dt, clip(self.t_move - 2 * dt, 0.0, 9.0))
        moving = self.t_move > MOVE_HOLD
        self.t_still = where(moving, self.t_still * 0, self.t_still + dt)
        mint, msize = p["mint"] * as_float(vis), p["msize"] * as_float(vis)
        strong, medium, light = mint > 2.5, (mint > 1.5) & (mint <= 2.5), (mint > 0.5) & (mint <= 1.5)
        big, small = msize > 2.5, msize < 1.5
        dancing_now = strong & big & ~moving
        # lekkende tellers: een enkel ruizig beeld zet ze niet terug op 0
        self.t_dance = where(dancing_now, self.t_dance + dt, clip(self.t_dance - 2 * dt, 0.0, 9.0))
        self.t_nodance = where(strong & big, clip(self.t_nodance - 2 * dt, 0.0, 99.0), self.t_nodance + dt)
        active = (mint > 1.5) | moving
        self.t_idle = where(active, self.t_idle * 0, self.t_idle + dt)
        self.t_state = self.t_state + dt

        near = (d - FOLLOW_DIST < NEAR_OK) & (absv(b) < FACE_OK)
        far = (d - FOLLOW_DIST > FAR_AGAIN) | (absv(b) > SIDE_AGAIN)
        want_follow = seen & moving & strong & ~small
        want_orbit = seen & (self.t_dance > DANCE_HOLD) & (self.t_nodance < 0.5)
        want_approach = seen & (medium | (strong & ~want_follow & ~want_orbit)) & far
        want_watch = seen & (mint > 1.5) & ~far
        too_close = seen & vis & (d < BACK_DIST)

        resting = (S == REST) | (S == WATCH)
        # --- overgangen (volgorde = prioriteit, laatste wint)
        self._go(resting & want_watch, WATCH)
        # 2026-10-30 (Jairo): zittend en de actie aan de rand van wat de kop kan zien -> opstaan en
        # het lijf meedraaien (KIJKEN draait op de plaats boven 25 gr)
        self._go((S == REST) & vis & (mint > 0.5) & (absv(b) > EDGE_TURN), WATCH)
        self._go(resting & want_approach, APPROACH)
        self._go(((S == REST) | (S == WATCH) | (S == APPROACH)) & want_follow, FOLLOW)
        self._go(((S == REST) | (S == WATCH) | (S == APPROACH) | (S == FOLLOW)) & want_orbit, ORBIT)
        # NADEREN klaar
        arrived = (S == APPROACH) & near
        self._go(arrived, WATCH)
        # VOLGEN stopt als de persoon stil staat en niet danst
        self._go((S == FOLLOW) & (self.t_still > STILL_TO_WATCH) & ~want_orbit,
                 APPROACH)
        # ORBIT stopt: geen dansen meer -> aankijken; persoon loopt -> volgen
        self._go((S == ORBIT) & (self.t_nodance > DANCE_END), APPROACH)
        self._go((S == ORBIT) & want_follow & (self.t_dance < 0.1), FOLLOW)
        # te dichtbij
        self._go(((S == WATCH) | (S == APPROACH) | (S == FOLLOW) | (S == ORBIT)
                  | ((S == REST) & (mint > 0.5))) & too_close, BACKOFF)
        self._go((S == BACKOFF) & (d > BACK_RELEASE), WATCH)
        # kwijt
        lost_any = ~seen
        self._go(((S == WATCH) | (S == APPROACH) | (S == FOLLOW) | (S == ORBIT) | (S == BACKOFF)) & lost_any, SEARCH)
        self._go((S == SEARCH) & seen & vis, WATCH)
        self._go((S == SEARCH) & (self.t_state > SEARCH_TO_REST), REST)
        # niets meer te zien of te doen -> zitten
        self._go((S == WATCH) & (self.t_idle > IDLE_TO_REST), REST)
        S = self.state

        # --- commando's per toestand
        face = clip(1.0 - absv(b), 0.0, 1.0)
        v_follow = clip((K_DIST * (d - FOLLOW_DIST) - p["approach"]) * face, 0.0, v100)
        v_appr = clip(K_DIST * (d - FOLLOW_DIST) * face, 0.0, 0.7 * v100)
        # orbit: kant gekozen bij binnenkomst (waar de persoon nu staat)
        enter = (S == ORBIT) & (self.t_state < 1.5 * dt)
        self.orbit_side = where(enter, where(b >= 0, b * 0 + 1.0, b * 0 - 1.0), self.orbit_side)
        side = self.orbit_side
        v_orb = ORBIT_FRAC * v100
        b_goal = side * (ORBIT_BEAR - clip(K_DIST * (d - ORBIT_R), -0.4, 0.4))
        w_orb = side * v_orb / ORBIT_R + K_BEAR * (b - b_goal)
        v_orb = v_orb * clip(1.0 - absv(b - b_goal) / 0.5, 0.0, 1.0)   # eerst opzij draaien, dan lopen
        v_back = -clip(K_DIST * (BACK_RELEASE - d), 0.035, 0.5 * v100)
        w_face = K_BEAR * b
        vx = zeros_like(b)
        wz = zeros_like(b)
        vx = where(S == APPROACH, v_appr, vx)
        wz = where(S == APPROACH, w_face, wz)
        vx = where(S == FOLLOW, v_follow, vx)
        wz = where(S == FOLLOW, w_face, wz)
        vx = where(S == ORBIT, v_orb, vx)
        wz = where(S == ORBIT, w_orb, wz)
        vx = where(S == BACKOFF, v_back, vx)
        wz = where(S == BACKOFF, w_face, wz)
        wz = where(S == SEARCH, clip(K_BEAR * self.last_b, -W_MAX, W_MAX), wz)
        # KIJKEN: draait op de plaats als de persoon > 25 gr opzij staat
        turn = (S == WATCH) & (absv(b) > 0.45)
        wz = where(turn, w_face, wz)
        wz = clip(wz, -W_MAX, W_MAX)
        # geleidelijk
        self.vx = self.vx + clip(vx - self.vx, -ACC_V * dt, ACC_V * dt)
        self.wz = self.wz + clip(wz - self.wz, -ACC_W * dt, ACC_W * dt)
        stop = (S == REST) | ((S == WATCH) & ~turn)
        self.vx = where(stop & (absv(self.vx) < 0.01), self.vx * 0, self.vx)
        self.wz = where(stop & (absv(self.wz) < 0.05), self.wz * 0, self.wz)

        # --- kop: vogel - fixeren, dan saccade
        self.t_look = self.t_look + dt
        look = LOOK_AMP * (0.6 * sin(0.55 * self.t_look) + 0.4 * sin(1.3 * self.t_look + 1.7))
        tgt_yaw = where(vis, b, where((S == REST) | (S == SEARCH), look, self.g_yaw))
        tgt_yaw = clip(tgt_yaw, -HEAD_YAW_MAX, HEAD_YAW_MAX)
        tgt_elev = where(vis, clip(self.last_e, HEAD_ELEV_MIN, HEAD_ELEV_MAX), self.g_elev * 0.98)
        self.t_fix = self.t_fix + dt
        jump = (((tgt_yaw - self.g_yaw) ** 2 + (tgt_elev - self.g_elev) ** 2) > SACCADE_THR ** 2) & (self.t_fix > FIX_MIN)
        self.g_yaw = where(jump, tgt_yaw, self.g_yaw)
        self.g_elev = where(jump, tgt_elev, self.g_elev)
        self.t_fix = where(jump, self.t_fix * 0, self.t_fix)
        r = SACCADE_RATE * dt
        self.h_yaw = self.h_yaw + clip(self.g_yaw - self.h_yaw, -r, r)
        self.h_elev = self.h_elev + clip(self.g_elev - self.h_elev, -r, r)
        fix = (((self.g_yaw - self.h_yaw) ** 2 + (self.g_elev - self.h_elev) ** 2) < 1e-4)
        # nek omlaag (-) en kop omhoog (+) geven samen elevatie: elk de helft
        head = (-0.5 * self.h_elev, 0.5 * self.h_elev, self.h_yaw, zeros_like(self.h_yaw))
        head = torch.stack(head, -1) if _t(self.h_yaw) else np.stack(head, -1)
        return self.vx, self.wz, head, fix


# ------------------------------------------------------------------ simulatie (training)
# virtuele persoon: wat hij doet en hoe het eruit ziet voor de camera
P_ABSENT, P_STILL, P_TALK, P_WAVE, P_WAVE_HARD, P_WALK, P_DANCE, P_COME, P_CLOSE = range(9)
P_NAMES = ("weg", "stil", "praat", "zwaait", "zwaait hard", "loopt", "danst", "komt", "te dichtbij")
#           mint msize  kans
P_LOOK = {P_ABSENT: (0, 0, 0.06), P_STILL: (0, 0, 0.14), P_TALK: (1, 1, 0.12), P_WAVE: (2, 1, 0.12),
          P_WAVE_HARD: (3, 1, 0.08), P_WALK: (3, 3, 0.22), P_DANCE: (3, 3, 0.14), P_COME: (2, 3, 0.06),
          P_CLOSE: (1, 2, 0.06)}
ESP_DELAY = 4            # control steps (80 ms)
DETECT_LAT = (0.3, 1.0)  # s voordat de ESP een nieuwe beweging herkent
FOV_H, FOV_V = math.radians(30), math.radians(22)
CAM_RANGE = 2.5
APPARENT_D = 0.50        # m: op deze afstand is de schijnbare grootte = de echte grootte


class PersonSim:
    """Gevectoriseerd (torch). Persoon op de vloer, saillant punt op hoogte h."""

    def __init__(self, n, device):
        self.n, self.dev = n, device
        f = lambda: torch.zeros(n, device=device)
        self.pos = torch.zeros(n, 2, device=device)
        self.yaw, self.v, self.w, self.h, self.age, self.lat, self.t_next = f(), f(), f(), f(), f(), f(), f()
        self.mode = torch.zeros(n, dtype=torch.long, device=device)
        self.probs = torch.tensor([P_LOOK[k][2] for k in range(9)], device=device)
        self.look = torch.tensor([[P_LOOK[k][0], P_LOOK[k][1]] for k in range(9)], dtype=torch.float32, device=device)
        self.buf = torch.zeros(ESP_DELAY + 1, n, 8, device=device)
        self.phase = f()
        self.vel_lp = torch.zeros(n, 2, device=device)

    def reset(self, ids, robot_xy, robot_yaw):
        k = len(ids)
        if k == 0:
            return
        u = lambda a, b: torch.empty(k, device=self.dev).uniform_(a, b)
        r, bear = u(0.4, 1.4), u(-0.8, 0.8)
        hd = robot_yaw + bear
        self.pos[ids] = robot_xy + r[:, None] * torch.stack([torch.cos(hd), torch.sin(hd)], -1)
        self.yaw[ids] = u(-math.pi, math.pi)
        self.mode[ids] = torch.multinomial(self.probs, k, replacement=True)
        self.h[ids] = u(0.05, 0.40)
        self.t_next[ids] = u(2.0, 6.0)
        self.age[ids] = 0.0
        self.lat[ids] = u(*DETECT_LAT)
        self.v[ids], self.w[ids] = 0.0, 0.0
        self.buf[:, ids] = 0.0
        self.vel_lp[ids] = 0.0
        self._set_motion(ids, robot_xy, robot_yaw)

    def _set_motion(self, ids, robot_xy, robot_yaw):
        k = len(ids)
        u = lambda a, b: torch.empty(k, device=self.dev).uniform_(a, b)
        m = self.mode[ids]
        walk, come = m == P_WALK, m == P_COME
        self.v[ids] = torch.where(walk, u(0.03, 0.09), torch.where(come, u(0.02, 0.06), torch.zeros(k, device=self.dev)))
        self.w[ids] = torch.where(walk, u(-0.5, 0.5), torch.zeros(k, device=self.dev))
        # komt naar de eend toe / staat te dichtbij
        to_duck = robot_xy - self.pos[ids]
        self.yaw[ids] = torch.where(come, torch.atan2(to_duck[:, 1], to_duck[:, 0]), self.yaw[ids])
        close = m == P_CLOSE
        if close.any():
            ci = ids[close]
            hd = robot_yaw[close] + u(-0.3, 0.3)[close]
            self.pos[ci] = robot_xy[close] + u(0.18, 0.30)[close][:, None] * torch.stack([torch.cos(hd), torch.sin(hd)], -1)

    def step(self, dt, robot_xy, robot_yaw, cam_pos, cam_R, noise=True):
        """Beweegt de persoon en geeft het (vertraagde) protocol zoals de ESP het zou sturen."""
        self.age += dt
        self.t_next -= dt
        new = self.t_next <= 0
        if new.any():
            ids = new.nonzero(as_tuple=False).flatten()
            k = len(ids)
            self.mode[ids] = torch.multinomial(self.probs, k, replacement=True)
            self.t_next[ids] = torch.empty(k, device=self.dev).uniform_(2.0, 7.0)
            self.age[ids] = 0.0
            self.lat[ids] = torch.empty(k, device=self.dev).uniform_(*DETECT_LAT)
            self._set_motion(ids, robot_xy[ids], robot_yaw[ids])
        # bewegen
        self.yaw += self.w * dt
        vel = self.v[:, None] * torch.stack([torch.cos(self.yaw), torch.sin(self.yaw)], -1)
        dance = (self.mode == P_DANCE).float()[:, None]
        self.phase += dt
        # dansen: heen en weer op de plek (+-1.5 cm), geen netto verplaatsing
        vel = vel + dance * torch.stack([0.075 * torch.cos(5.0 * self.phase), -0.055 * torch.sin(3.7 * self.phase)], -1)
        stop = (self.mode == P_COME) & (torch.norm(robot_xy - self.pos, dim=-1) < 0.30)
        vel = torch.where(stop[:, None], torch.zeros_like(vel), vel)
        self.pos = self.pos + vel * dt
        # saillant punt in het cameraframe (x vooruit, y links, z omhoog)
        pt = torch.cat([self.pos, self.h[:, None]], -1)
        rel_w = pt - cam_pos
        rel = torch.einsum("nji,nj->ni", cam_R, rel_w)          # R^T * rel_w
        dist_xy = torch.norm(rel_w[:, :2], dim=-1).clamp(min=1e-3)
        bx = torch.atan2(rel[:, 1], rel[:, 0])
        by = torch.atan2(rel[:, 2], torch.norm(rel[:, :2], dim=-1).clamp(min=1e-3))
        vis = (rel[:, 0] > 0) & (bx.abs() < FOV_H) & (by.abs() < FOV_V) & (dist_xy < CAM_RANGE) & (self.mode != P_ABSENT)
        # eigen beweging van het punt (wereld): dwars op de kijklijn en naar de camera toe
        u_los = rel_w[:, :2] / dist_xy[:, None]
        a = min(1.0, dt / 0.8)                                    # de ESP middelt de vlek ~0.8 s
        self.vel_lp = self.vel_lp + a * (vel - self.vel_lp)
        tflow = (u_los[:, 0] * self.vel_lp[:, 1] - u_los[:, 1] * self.vel_lp[:, 0]) / dist_xy
        approach = -(u_los * self.vel_lp).sum(-1)
        look = self.look[self.mode]
        seen_long = self.age > self.lat                           # herkenningsvertraging
        mint = torch.where(seen_long, look[:, 0], torch.zeros_like(look[:, 0]))
        msize = torch.where(seen_long, look[:, 1], torch.zeros_like(look[:, 1]))
        msize = torch.where(msize > 0, (msize * APPARENT_D / dist_xy).round().clamp(1, 3), msize)
        vis = vis & ((mint > 0) | (msize > 0))                    # niets beweegt = niets in beeld
        d = dist_xy
        if noise:
            n = self.n
            bx = bx + math.radians(1.5) * torch.randn(n, device=self.dev)
            by = by + math.radians(1.5) * torch.randn(n, device=self.dev)
            d = d * (1.0 + 0.12 * torch.randn(n, device=self.dev))
            tflow = tflow + 0.03 * torch.randn(n, device=self.dev)
            approach = approach + 0.01 * torch.randn(n, device=self.dev)
            flip = torch.rand(n, device=self.dev) < 0.02                    # af en toe een niveau ernaast
            mint = torch.where(flip, (mint + torch.randint(-1, 2, (n,), device=self.dev)).clamp(0, 3), mint)
            msize = torch.where(flip, (msize + torch.randint(-1, 2, (n,), device=self.dev)).clamp(0, 3), msize)
            vis = vis & (torch.rand(n, device=self.dev) > 0.05)              # uitval
        frame = torch.stack([vis.float(), bx, by, d, msize, mint, tflow, approach], -1)
        self.buf = torch.roll(self.buf, 1, 0)
        self.buf[0] = frame
        out = self.buf[-1]
        keys = ("vis", "bx", "by", "dist", "msize", "mint", "tflow", "approach")
        return {k: out[:, i] for i, k in enumerate(keys)}
