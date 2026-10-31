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
from mjlab.actuator import DcMotorActuator, DcMotorActuatorCfg, IdealPdActuator, IdealPdActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd

RATIO = 24.0 / 16.0
ES9051 = dict(stall=0.0605, speed=9.0, kp=2.0, kd=0.02, armature=2e-5, friction=0.003)
ES9251 = dict(stall=0.0189, speed=8.1, kp=0.6, kd=0.006, armature=8e-6, friction=0.0015)
DIFF_HIP = dict(ES9051, kp=1.0, kd=0.01)
DIFF_HEAD = dict(ES9251, kp=0.3, kd=0.003)
DELAY = dict(delay_min_lag=1, delay_max_lag=6)   # 2026-10-31: 2-12 ms commandovertraging (was 0-4 ms)
# 2026-10-31: echte hobbyservo's zijn minder soepel dan het model. Per env (vast voor de run):
DEADBAND_DEG = (0.3, 1.5)    # dode zone: servo reageert pas op een fout groter dan dit (servo-as)
SPEED_SCALE = (0.8, 1.1)     # onbelaste snelheid ten opzichte van de spec


def _per_env(lo, hi, n_env, n_j, device):
    return lo + (hi - lo) * torch.rand(n_env, n_j, device=device)


def _deadband(err, db):
    return torch.sign(err) * torch.clamp(err.abs() - db, min=0.0)


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
        n = data.nworld
        self._db = torch.deg2rad(_per_env(*DEADBAND_DEG, n, 2, device))      # per servo A, B
        self._w0 = self.cfg.servo_speed * _per_env(*SPEED_SCALE, n, 2, device)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        R, io, ic = self.cfg.ratio, self._io, self._ic
        kp, kd = self.stiffness[:, io], self.damping[:, io]
        stall = self.force_limit[:, io] / (2.0 * R)          # gerandomiseerd via effort_limits
        q, qd, qt, vt = cmd.pos, cmd.vel, cmd.position_target, cmd.velocity_target
        taus = []
        for k, sgn in enumerate((1.0, -1.0)):
            w0 = self._w0[:, k]
            s = R * (q[:, io] + sgn * q[:, ic])
            w = R * (qd[:, io] + sgn * qd[:, ic])
            st = R * (qt[:, io] + sgn * qt[:, ic])
            wt = R * (vt[:, io] + sgn * vt[:, ic])
            t = kp * _deadband(st - s, self._db[:, k]) + kd * (wt - w)
            hi = torch.clamp(stall * (1.0 - w / w0), min=-stall, max=stall)
            lo = torch.clamp(stall * (-1.0 - w / w0), min=-stall, max=stall)
            taus.append(torch.minimum(torch.maximum(t, lo), hi))
        out = torch.zeros_like(q)
        out[:, io] = R * (taus[0] + taus[1])
        out[:, ic] = R * (taus[0] - taus[1])
        return out


@dataclass(kw_only=True)
class ServoActuatorCfg(DcMotorActuatorCfg):
    """Enkele servo: DC-koppel/snelheidscurve + dode zone + snelheidsspreiding per env."""

    def build(self, entity, target_ids, target_names):
        return ServoActuator(self, entity, target_ids, target_names)


class ServoActuator(DcMotorActuator[ServoActuatorCfg]):
    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        n, j = data.nworld, len(self._target_names)
        self._db = torch.deg2rad(_per_env(*DEADBAND_DEG, n, j, device))
        self.velocity_limit_motor = self.velocity_limit_motor * _per_env(*SPEED_SCALE, n, j, device)
        self._vel_at_effort_lim = self.velocity_limit_motor * (1 + self.force_limit / self.saturation_effort)

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        self._joint_vel_clipped[:] = cmd.vel
        t = self.stiffness * _deadband(cmd.position_target - cmd.pos, self._db)
        t = t + self.damping * (cmd.velocity_target - cmd.vel) + cmd.effort_target
        return self._clip_effort(t)


def _single(names, sv):
    return ServoActuatorCfg(
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
