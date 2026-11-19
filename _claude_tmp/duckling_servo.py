"""duckling_servo.py - servomodel van de Duckling V2 (numpy; zelfde model als de training).

Elke servo: PD op de servo-as, begrensd door een koppel-snelheidscurve
    max koppel = stall * (1 - snelheid / onbelast)   (lineair, zoals een DC-motor)
Differentiëlen (heup: pitch + roll, kop: yaw + roll): twee servo's via tandwielen 16:24 (1,5x):
    servo A = R (uit + drager),  servo B = R (uit - drager)
    koppel uit = R (tA + tB),    koppel drager = R (tA - tB)
Het netwerk (en de firmware-interface) werkt met GEWRICHTSHOEKEN; de omrekening zit hier.

Waarden op 1S (3.7 V), geschaald van de 4.8 V-specs:
    ES9051  60.5 mNm stall, ~9.0 rad/s onbelast   (heup, enkel, teen, nek)
    ES9251  18.9 mNm stall, ~8.1 rad/s onbelast   (kop)
"""
import numpy as np

RATIO = 24.0 / 16.0           # differentieel-tandwielen
DEADBAND = np.radians(0.8)    # 2026-10-31: dode zone servo-as (training: 0.3-1.5 gr per env)


def _db(e):
    return np.sign(e) * max(abs(e) - DEADBAND, 0.0)
ES9051 = dict(stall=0.0605, speed=9.0, kp=2.0, kd=0.02)
ES9251 = dict(stall=0.0189, speed=8.1, kp=0.6, kd=0.006)
# servo-PD van een differentieel-servo (servo-as): lager dan enkel, want via 2 servo's en 1,5^2
DIFF_HIP = dict(ES9051, kp=1.0, kd=0.01)
DIFF_HEAD = dict(ES9251, kp=0.3, kd=0.003)

SINGLE = {   # gewricht -> servo
    "left_hip_yaw": ES9051, "right_hip_yaw": ES9051, "left_knee": ES9051, "right_knee": ES9051,
    "left_ankle": ES9051, "right_ankle": ES9051, "neck_pitch": ES9051, "head_pitch": ES9251,
}
DIFFS = (    # (naam, uit-gewricht, drager-gewricht, servo)
    ("heup_L", "left_hip_pitch", "left_hip_roll", DIFF_HIP),
    ("heup_R", "right_hip_pitch", "right_hip_roll", DIFF_HIP),
    ("kop", "head_yaw", "head_roll", DIFF_HEAD),
)


def _curve(t, w, sv):
    hi = np.clip(sv["stall"] * (1.0 - w / sv["speed"]), -sv["stall"], sv["stall"])
    lo = np.clip(sv["stall"] * (-1.0 - w / sv["speed"]), -sv["stall"], sv["stall"])
    return np.minimum(np.maximum(t, lo), hi)


class ServoModel:
    """Koppels op de gewrichten uit gewrichtsdoelen, per fysica-stap."""

    def __init__(self, names):
        self.names = list(names)
        ix = self.names.index
        self.single = [(ix(n), sv) for n, sv in SINGLE.items() if n in self.names]
        self.diffs = [(nm, ix(o), ix(c), sv) for nm, o, c, sv in DIFFS if o in self.names]
        self.last = {}              # servo -> (koppel / stall, snelheid / onbelast)

    def torques(self, q, qd, target):
        tau = np.zeros(len(self.names))
        for i, sv in self.single:
            t = sv["kp"] * _db(target[i] - q[i]) - sv["kd"] * qd[i]
            tau[i] = _curve(t, qd[i], sv)
            self.last[self.names[i]] = (tau[i] / sv["stall"], qd[i] / sv["speed"])
        R = RATIO
        for nm, o, c, sv in self.diffs:
            out = []
            for sgn, tag in ((1.0, "A"), (-1.0, "B")):
                s, w, st = R * (q[o] + sgn * q[c]), R * (qd[o] + sgn * qd[c]), R * (target[o] + sgn * target[c])
                t = _curve(sv["kp"] * _db(st - s) - sv["kd"] * w, w, sv)
                out.append(t)
                self.last[f"{nm}{tag}"] = (t / sv["stall"], w / sv["speed"])
            tau[o] = R * (out[0] + out[1])
            tau[c] = R * (out[0] - out[1])
        return tau


def servo_speeds(q, fps, names):
    """Servosnelheden (rad/s) uit een gewrichtstraject q (T, J), periodiek. Geeft dict
    servo -> (max |snelheid|, onbelaste snelheid)."""
    q = np.asarray(q)
    v = np.diff(np.vstack([q, q[:1]]), axis=0) * fps
    ix = list(names).index
    out = {}
    for n, sv in SINGLE.items():
        if n in names:
            out[n] = (float(np.abs(v[:, ix(n)]).max()), sv["speed"])
    for nm, o, c, sv in DIFFS:
        if o in names:
            vo, vc = v[:, ix(o)], v[:, ix(c)]
            out[nm + "A"] = (float(np.abs(RATIO * (vo + vc)).max()), sv["speed"])
            out[nm + "B"] = (float(np.abs(RATIO * (vo - vc)).max()), sv["speed"])
    return out


def worst_by_group(speeds):
    """Samenvatting: heupservo, enkel (knee), teen (ankle), heup-yaw, nek, kop."""
    g = {"heupservo": ("heup_L", "heup_R"), "enkel": ("left_knee", "right_knee"),
         "teen": ("left_ankle", "right_ankle"), "heup_yaw": ("left_hip_yaw", "right_hip_yaw"),
         "nek": ("neck_pitch",), "kop": ("head_pitch", "kop")}
    res = {}
    for k, pre in g.items():
        vals = [(v, lim) for n, (v, lim) in speeds.items() if any(n.startswith(p) for p in pre)]
        if vals:
            res[k] = max(vals)
    return res


def neutralize_xml_actuators(m, nj):
    """De <position>-actuators uit robot_walk.xml krachtloos maken (het servomodel levert de koppels)."""
    m.actuator_gainprm[:nj, :] = 0.0
    m.actuator_biasprm[:nj, :] = 0.0


ARMATURE = {"ES9051": 2e-5, "ES9251": 8e-6}   # servo-as (zoals in robot_walk.xml)


def apply_armature(m, names, dof_ids):
    """Differentieel-gewrichten: twee servo's via 1,5x -> 2 * 1.5^2 * armatuur van een servo."""
    for nm, o, c, sv in DIFFS:
        a = ARMATURE["ES9051" if sv["stall"] > 0.03 else "ES9251"] * 2 * RATIO ** 2
        for j in (o, c):
            if j in names:
                m.dof_armature[dof_ids[list(names).index(j)]] = a
