"""Servomodel van de Duckling V2 in mjlab (2026-10-29). Zelfde model als tools/duckling_servo.py.

* Enkele servo's (heup-yaw, enkel, teen, nek: ES9051; kop-pitch: ES9251): PD + lineaire
  koppel-snelheidscurve (mjlab DcMotorActuator): stall bij stilstand, 0 bij de onbelaste snelheid.
* Differentiëlen (heup pitch/roll, kop yaw/roll): twee servo's via tandwielen 16:24 (R = 1.5)
      servo A = R (uit + drager),  servo B = R (uit - drager)
      koppel uit = R (tA + tB),    koppel drager = R (tA - tB)
  PD en koppel-snelheidscurve per SERVO. Het netwerk geeft gewrichtsdoelen; de omrekening
  zit hier (en straks in de firmware).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.actuator import DcMotorActuatorCfg, IdealPdActuator, IdealPdActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd

RATIO = 24.0 / 16.0
ES9051 = dict(stall=0.0605, speed=9.0, kp=2.0, kd=0.02, armature=2e-5, friction=0.003)
ES9251 = dict(stall=0.0189, speed=8.1, kp=0.6, kd=0.006, armature=8e-6, friction=0.0015)
DIFF_HIP = dict(ES9051, kp=1.0, kd=0.01)
DIFF_HEAD = dict(ES9251, kp=0.3, kd=0.003)
DELAY = dict(delay_min_lag=0, delay_max_lag=2)   # 0-4 ms commandovertraging


@dataclass(kw_only=True)
class DiffServoActuatorCfg(IdealPdActuatorCfg):
    """Differentieel: target_names_expr = (uit-gewricht, drager-gewricht).
    stiffness/damping = PD van één SERVO (servo-as); effort_limit = max gewrichtskoppel = 2 R stall."""

    out_joint: str
    carrier_joint: str
    ratio: float = RATIO
    servo_speed: float = 9.0

    def build(self, entity, target_ids, target_names):
        return DiffServoActuator(self, entity, target_ids, target_names)


class DiffServoActuator(IdealPdActuator[DiffServoActuatorCfg]):
    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self._io = self._target_names.index(self.cfg.out_joint)
        self._ic = self._target_names.index(self.cfg.carrier_joint)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        R, io, ic = self.cfg.ratio, self._io, self._ic
        kp, kd = self.stiffness[:, io], self.damping[:, io]
        stall = self.force_limit[:, io] / (2.0 * R)          # gerandomiseerd via effort_limits
        w0 = self.cfg.servo_speed
        q, qd, qt, vt = cmd.pos, cmd.vel, cmd.position_target, cmd.velocity_target
        taus = []
        for sgn in (1.0, -1.0):
            s = R * (q[:, io] + sgn * q[:, ic])
            w = R * (qd[:, io] + sgn * qd[:, ic])
            st = R * (qt[:, io] + sgn * qt[:, ic])
            wt = R * (vt[:, io] + sgn * vt[:, ic])
            t = kp * (st - s) + kd * (wt - w)
            hi = torch.clamp(stall * (1.0 - w / w0), min=-stall, max=stall)
            lo = torch.clamp(stall * (-1.0 - w / w0), min=-stall, max=stall)
            taus.append(torch.minimum(torch.maximum(t, lo), hi))
        out = torch.zeros_like(q)
        out[:, io] = R * (taus[0] + taus[1])
        out[:, ic] = R * (taus[0] - taus[1])
        return out


def _single(names, sv):
    return DcMotorActuatorCfg(
        target_names_expr=names, stiffness=sv["kp"], damping=sv["kd"],
        effort_limit=sv["stall"], saturation_effort=sv["stall"], velocity_limit=sv["speed"],
        armature=sv["armature"], frictionloss=sv["friction"], **DELAY)


def _diff(out, carrier, sv):
    return DiffServoActuatorCfg(
        target_names_expr=(f"^{out}$", f"^{carrier}$"), out_joint=out, carrier_joint=carrier,
        stiffness=sv["kp"], damping=sv["kd"], effort_limit=2 * RATIO * sv["stall"],
        servo_speed=sv["speed"], armature=2 * RATIO ** 2 * sv["armature"],
        frictionloss=2 * RATIO * sv["friction"], **DELAY)


def duckling_servo_actuators():
    return (
        _single((r"^(left|right)_hip_yaw$", r"^(left|right)_knee$", r"^(left|right)_ankle$", r"^neck_pitch$"), ES9051),
        _single((r"^head_pitch$",), ES9251),
        _diff("left_hip_pitch", "left_hip_roll", DIFF_HIP),
        _diff("right_hip_pitch", "right_hip_roll", DIFF_HIP),
        _diff("head_yaw", "head_roll", DIFF_HEAD),
    )
