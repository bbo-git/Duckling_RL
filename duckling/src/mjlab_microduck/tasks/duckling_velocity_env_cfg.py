"""Duckling velocity (walking) task, built around a reference walk cycle.

Scene, observations, commands, sensors and domain randomization come from the
Microduck velocity task; the reward set is Duckling's own (see REWARDS below):
follow the Blender walk cycle (joint angles, foot-contact timing, trunk height) while
tracking the velocity command, stand in HOME when the command is zero.
Episodes start at a random phase of the cycle in the reference pose (reference state
initialisation), so every part of the cycle is practised from the first iteration.

Actor obs: Microduck's 61 + gait clock (sin, cos of the phase) = 63.
Action = 14 joint position targets (offsets from HOME) at 50 Hz.
"""

import json
import math
import os

import torch
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs import mdp as envs_mdp
from pathlib import Path

import numpy as np
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg

from mjlab_microduck.robot.duckling_constants import DUCKLING_WALK_ROBOT_CFG, SPAWN_Z
from dataclasses import dataclass as _dataclass

from mjlab_microduck.tasks.mdp import VelocityCommandCommandOnly, VelocityCommandCommandOnlyCfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

SCALE = 0.6  # Duckling / Microduck length ratio

# Physics: smaller robot -> faster dynamics -> smaller timestep. 2 ms x 10 = 50 Hz policy.
TIMESTEP = 0.002
DECIMATION = 10

# 2026-10-29 (V2, alles servo's): HOME = CAD-nulstand, romp staat op 59.2 mm
HOME_TRUNK_HEIGHT = 0.0592  # m, trunk_base standing in HOME
STANCE_MAX_WIDTH = 0.065   # m (V1). V2: per stijl = voetbreedte van het loopje + STANCE_SLACK
STANCE_SLACK = 0.012       # m boven de voetbreedte van de stijl
MIN_TRUNK_HEIGHT = 0.046  # m; V2 loopt op 60-65 mm, hurkt tot ~44 mm (hurken is een aparte beweging)

# --- 2026-10-29: STIJLEN. Eén policy, meerdere loopjes ("karakter"). Per episode krijgt elke env
# een stijl; de policy ziet hem als one-hot in de observaties, en de referentie (imitatie, klok,
# contact, romphoogte, kop) komt uit het rooster van die stijl. Op de robot kiest de controller.
STYLES = [   # (naam, rooster in robot/duckling/grids/)
    ("normaal", "gait_grid_normaal_v3.npz"),
    ("parmantig", "gait_grid_parmantig_v5.npz"),
]
STYLE_PROBS = (0.5, 0.5)
# Taak: "lopen" (snelheidscommando's, eerst trainen) of "volgen" (persoon volgen, fine-tune daarna)
TASK = os.environ.get("DUCK_TASK", "lopen")
# 2026-10-29: GEDRAG - in dit deel van de envs komen vx, wz en het kopcommando uit de toestandsmachine
# van de robot (duckling_behavior.py) met een virtuele persoon en nep-ESP, zodat de policy alle
# overgangen oefent (rust -> naderen -> volgen -> orbit -> terug -> ...). De rest: willekeurig.
BEHAVIOR_FRAC = 0.5

HEAD_BODY_NAMES = ("neck", "head_pitch", "head_roll", "head")
ACTUATED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))

# Final head-pose command caps (rad, deltas from HOME) in the order
# neck_pitch, head_pitch, head_yaw, head_roll — ~90 % of the joint ranges in the CAD export.
# 2026-10-13: halved (was 0.47, 0.47, 0.93, 0.31 = ~90 % of the joint range). With full-range
# look commands from iteration 2000 on the heavy head swung +-50 deg and the walking got worse.
HEAD_CAPS = (0.25, 0.25, 0.50, 0.15)
# 2026-10-29: kop-yaw tot 63 gr (orbit: persoon op ~90 gr opzij, kop naar binnen + 30 gr camerabeeld);
# nek + kop samen 29 gr omhoog
HEAD_CAPS = (0.25, 0.25, 1.10, 0.15)


def _head_stage(frac: float):
    return tuple((-c * frac, c * frac) for c in HEAD_CAPS)


def _bounded_ang_vel_penalty(env, asset_cfg: SceneEntityCfg, max_sq: float = 4.0):
    """Trunk roll/pitch rate penalty sum(w_xy^2), clipped at max_sq.

    The stock version is unbounded. The duckling is small and light, so a stumble
    gives ~10 rad/s and a penalty far larger than everything it earns per step:
    dying fast then beats surviving, and PPO learns to collapse (episode length 3).
    """
    w = env.scene[asset_cfg.name].data.body_link_ang_vel_w[:, asset_cfg.body_ids, :2]
    return torch.clamp(torch.sum(torch.square(w.squeeze(1)), dim=1), max=max_sq)


def _stance_too_wide(env, max_width: float, asset_cfg: SceneEntityCfg, per_style: bool = True):
    """Sideways foot spacing (trunk heading frame) beyond max_width, in m.

    Without it the policy settles in a wide straddle with tiny shuffling steps:
    very stable, but it never learns real steps. HOME spacing is ~47 mm.
    """
    asset = env.scene[asset_cfg.name]
    feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]          # (N, 2, 2)
    diff = feet[:, 0] - feet[:, 1]
    q = asset.data.root_link_quat_w                                  # w, x, y, z
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                      1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    lateral = -torch.sin(yaw) * diff[:, 0] + torch.cos(yaw) * diff[:, 1]
    if per_style:   # 2026-10-29: voetbreedte van de stijl + STANCE_SLACK (parmantig staat breder)
        max_width = _style_tensor(env, "width")[env_style(env)] + STANCE_SLACK
    return torch.clamp(lateral.abs() - max_width, min=0.0)


# --- Reference walk cycle (Blender) -------------------------------------------
# walk_cycle.npz: one gait cycle, CAD-zero joint angles at 50 Hz (convert_anim.py).
# The policy gets a gait clock (sin, cos of the phase) as 2 extra observations and
# is rewarded for following the reference leg angles while it is asked to move.
WALK_CYCLE_FILE = Path(__file__).parents[1] / "robot" / "duckling" / "walk_cycle.npz"
IMITATION_WEIGHT = 4.0   # 2026-10-15 (volg13): 2 -> 4 - the gait was eroding into shuffles (14: 4 -> 2)
# Style reference: your walk (gait_grid.npz from walk_speed.py: speed x turn x side) instead
# of the single Blender cycle. The gait is picked per env from the twist command
# (vx -> speed, wz -> turn) and blended between grid neighbours, with its own clock.
USE_GAIT_GRID = True
SIDE_CYCLE_S = 0.45          # s, typical cycle: sideways command (m/s) -> grid side axis (m per cycle)
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]   # = head_pose command order
GAIT_GRID_FILE = Path(__file__).parents[1] / "robot" / "duckling" / "gait_grid.npz"
# Centre of mass ahead of the stance feet at 100 % speed (com_forward reward). None = from your
# walk in the grid (com_ahead_m, the "zwaartepunt_voor_mm" slider). 0 below 20 %, more above 100 %.
COM_AHEAD_100 = None
SWING_LIFT = 0.010            # m (V1). V2: per stijl 0.8 x lift_m van het loopje
IMITATION_STD = 0.25          # rad, RMS joint error that gives exp(-1)
LEG_JOINTS = [f"{s}_{j}" for s in ("left", "right")
              for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
# 2026-10-30: bereik van de servodoelen (graden, t.o.v. HOME = CAD-nul) = gewrichtsbereik in robot_walk.xml
JOINT_TARGET_RANGE_DEG = {
    r"^(left|right)_hip_yaw$": (-20, 20), r"^(left|right)_hip_roll$": (-20, 20),
    r"^(left|right)_hip_pitch$": (-60, 60), r"^(left|right)_knee$": (-50, 90),
    r"^(left|right)_ankle$": (-60, 90), r"^neck_pitch$": (-30, 30), r"^head_pitch$": (-30, 30),
    r"^head_roll$": (-20, 20), r"^head_yaw$": (-90, 90),
}
# Per-joint weight in the imitation error. The walk cycle is a style guide for the
# fore-aft leg motion (hip pitch, knee, ankle). Hip roll/yaw are left mostly free: the
# animation has no sideways weight shift, so the policy must add its own sway to stay
# balanced (the stance_width penalty still stops it from splaying).
# The ankle is left to the policy too: in the animation the stance foot is tilted
# 10-30 deg (it would stand on an edge); the flat_feet reward keeps the sole flat instead.
# 2026-10-14: hip_yaw 0.3 -> 1.0: at 0.3 the policy wiggled the hip yaw +-15 deg for balance
# (straight and turning alike) instead of using it to steer
# 2026-10-14: hip_roll 0.3 -> 1.0: it lifted the swing foot sideways with hip roll instead of hip pitch
# 2026-10-15: hip_yaw back 1.0 -> 0.3: pinned to the grid's (weak) turn pattern it could not steer;
# free again, the follow rewards say where to go
# 2026-10-30 (Jairo): hip_yaw 0.3 -> 1.0: de animatie stuurt de heup-yaw (tenen stonden naar binnen;
# het rooster draait met de heup-yaw)
IMITATION_JOINT_WEIGHT = {"hip_yaw": 1.0, "hip_roll": 1.0, "hip_pitch": 1.0, "knee": 1.0, "ankle": 0.1}


_STYLE_CACHE = {}


def style_tables():
    """Per stijl uit de roosters: v100 (m/s bij 100%), zwaartepunt-voor, voetbreedte, voethoogte."""
    if not _STYLE_CACHE:
        base = Path(__file__).parents[1] / "robot" / "duckling" / "grids"
        rows = []
        for name, fn in STYLES:
            d = np.load(base / fn)
            st = json.loads(str(d["style"]))
            rows.append(dict(name=name, file=base / fn, v100=float(d["v100"]),
                             com100=COM_AHEAD_100 if COM_AHEAD_100 is not None else float(st.get("com_ahead_m", 0.0)),
                             width=float(st.get("width_m", 0.05)), lift=float(st.get("lift_m", 0.008))))
        _STYLE_CACHE["rows"] = rows
    return _STYLE_CACHE["rows"]


def _style_tensor(env, key):
    t = getattr(env, "_duck_style_t_" + key, None)
    if t is None:
        t = torch.tensor([r[key] for r in style_tables()], dtype=torch.float32, device=env.device)
        setattr(env, "_duck_style_t_" + key, t)
    return t


def env_style(env) -> torch.Tensor:
    """Stijl-index per env (gekozen door het twist-commando bij elke episode-reset)."""
    return env.command_manager.get_term("twist").style


def style_obs(env) -> torch.Tensor:
    """One-hot stijl (len(STYLES)) - de controller op de robot zet hem."""
    return torch.nn.functional.one_hot(env_style(env), len(STYLES)).float()


class _StyleMixin:
    """Houdt de stijl per env bij; nieuwe stijl bij elke episode-reset. De reset-event
    (reset_to_reference) kiest hem al vóór de commando-reset, zodat de startpose klopt."""

    def _init_style(self):
        self.style = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._style_p = torch.tensor(STYLE_PROBS, dtype=torch.float32, device=self.device)
        self._v100 = torch.tensor([r["v100"] for r in style_tables()], dtype=torch.float32, device=self.device)
        self._style_fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _pick_style(self, env_ids, from_event=False):
        if not hasattr(self, "_style_fresh"):
            return
        if not from_event:                    # al gekozen door de reset-event? dan niet opnieuw
            env_ids = env_ids[~self._style_fresh[env_ids]]
        if len(env_ids):
            self.style[env_ids] = torch.multinomial(self._style_p, len(env_ids), replacement=True)
        if from_event:
            self._style_fresh[env_ids] = True
        else:
            self._style_fresh[:] = False


class StyleVelocityCommand(_StyleMixin, VelocityCommandCommandOnly):
    """Lopen: vx uit ranges.lin_vel_x is een FRACTIE van de 100%-snelheid van de stijl van die env."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._init_style()

    def reset(self, env_ids):
        self._pick_style(env_ids)
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, 0] *= self._v100[self.style[env_ids]]
        self.vel_command_w[env_ids] = self.vel_command_b[env_ids]


@_dataclass(kw_only=True)
class StyleVelocityCommandCfg(VelocityCommandCommandOnlyCfg):
    def build(self, env):
        return StyleVelocityCommand(self, env)


class BehaviorVelocityCommand(StyleVelocityCommand):
    """2026-10-29: in BEHAVIOR_FRAC van de envs maakt de toestandsmachine van de robot het
    commando (vx, wz) en het kopcommando, op basis van een virtuele persoon gezien door een
    nep-ESP (zelfde protocol als straks de echte ESP32-CAM). Zie duckling_behavior.py."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from mjlab_microduck.tasks import duckling_behavior as B
        self.B = B
        n, dev = self.num_envs, self.device
        self.is_beh = torch.zeros(n, dtype=torch.bool, device=dev)
        self.behavior = B.Behavior(torch.zeros(n, device=dev))
        self.person = B.PersonSim(n, dev)
        self.head_cmd = torch.zeros(n, 4, device=dev)
        self.head_id = self.robot.find_bodies(["head"], preserve_order=True)[0][0]
        self.cam_off = torch.tensor([0.032, 0.0025, 0.0104], device=dev)   # ESP32-CAM in het kopframe
        self.metrics["beh_state"] = torch.zeros(n, device=dev)
        self._place = torch.ones(n, dtype=torch.bool, device=dev)   # persoon neerzetten zodra de pose klopt

    def reset(self, env_ids):
        out = super().reset(env_ids)
        if len(env_ids):
            k = len(env_ids)
            self.is_beh[env_ids] = torch.rand(k, device=self.device) < BEHAVIOR_FRAC
            start = torch.where(torch.rand(k, device=self.device) < 0.5, 0, 1)   # RUST of KIJKEN
            self.behavior.reset(env_ids)
            self.behavior.state[env_ids] = start.float()
            self.head_cmd[env_ids] = 0.0
            # de robotpose is pas na de volgende fysica-stap bijgewerkt: persoon dan pas neerzetten
            self._place[env_ids] = True
            self.is_standing_env[env_ids] &= ~self.is_beh[env_ids]
        return out

    def _camera(self):
        from mjlab.utils.lab_api.math import matrix_from_quat
        R = matrix_from_quat(self.robot.data.body_link_quat_w[:, self.head_id])      # (N, 3, 3)
        pos = self.robot.data.body_link_pos_w[:, self.head_id] + torch.einsum("nij,j->ni", R, self.cam_off)
        return pos, R

    def _update_command(self):
        super()._update_command()
        dt = self._env.step_dt
        if self._place.any():
            ids = self._place.nonzero(as_tuple=False).flatten()
            self.person.reset(ids, self.robot.data.root_link_pos_w[ids, :2], self.robot.data.heading_w[ids])
            self._place[ids] = False
        cam_pos, R = self._camera()
        p = self.person.step(dt, self.robot.data.root_link_pos_w[:, :2], self.robot.data.heading_w, cam_pos, R)
        vx, wz, head, _ = self.behavior.step(p, dt, self._v100[self.style])
        b = self.is_beh
        self.vel_command_b[b, 0] = vx[b]
        self.vel_command_b[b, 1] = 0.0
        self.vel_command_b[b, 2] = wz[b]
        # kop begrensd op het huidige bereik van het kop-curriculum; de toestandsmachine rekent
        # verder met de kophoek die er echt gevraagd wordt (zoals op de robot: geen encoders)
        hp = self._env.command_manager.get_term("head_pose")
        lo = torch.tensor([r[0] for r in hp.cfg.ranges], device=self.device)
        hi = torch.tensor([r[1] for r in hp.cfg.ranges], device=self.device)
        head = torch.maximum(torch.minimum(head, hi), lo)
        self.behavior.h_yaw = head[:, 2].clone()
        self.behavior.h_elev = head[:, 1] - head[:, 0]
        self.head_cmd = head
        self.metrics["beh_state"] = torch.where(b, self.behavior.state, self.metrics["beh_state"])


@_dataclass(kw_only=True)
class BehaviorVelocityCommandCfg(StyleVelocityCommandCfg):
    def build(self, env):
        return BehaviorVelocityCommand(self, env)


def _cycle_period() -> float:
    return float(np.load(WALK_CYCLE_FILE)["cycle_s"])


def _phase_offset(env) -> torch.Tensor:
    """Per-env phase at the start of the episode (set by the reset_to_reference event)."""
    off = getattr(env, "_duck_phase_offset", None)
    if off is None:
        off = torch.zeros(env.num_envs, device=env.device)
        env._duck_phase_offset = off
    return off


def _gait_phase(env) -> torch.Tensor:
    """Phase in [0, 1) of the walk cycle (gait grid: its own clock per env)."""
    if USE_GAIT_GRID:
        return _ref(env).phase_now(env)
    t = env.episode_length_buf.float() * env.step_dt / _gait_phase.period
    return torch.remainder(t + _phase_offset(env), 1.0)


_gait_phase.period = 1.0


def gait_clock(env) -> torch.Tensor:
    ph = 2 * math.pi * _gait_phase(env)
    clk = torch.stack([torch.sin(ph), torch.cos(ph)], dim=-1)
    if USE_GAIT_GRID:   # 2026-10-13: fades to (0, 0) when told to stand - no rhythm, no nodding
        clk = clk * _ref(env).walk[:, None]
    return clk


class walk_imitation:
    """exp(-mean((q - q_ref(phase))^2) / std^2) on the 10 leg joints, only for envs that
    are commanded to move (standing envs keep the HOME pose reward)."""

    def __init__(self, cfg, env):
        w = [IMITATION_JOINT_WEIGHT[j.split("_", 1)[1]] for j in LEG_JOINTS]
        self.w = torch.tensor(w, device=env.device) / sum(w)

    def __call__(self, env, std: float, std_fine: float = 0.0, command_name: str = "twist") -> torch.Tensor:
        ref = _ref(env)
        q_ref = ref.q_now(env)
        q = env.scene["robot"].data.joint_pos[:, ref.leg_ids]
        e = torch.sum(self.w * torch.square(q - q_ref), dim=-1)
        r = torch.exp(-e / std**2)
        if std_fine > 0:   # 2026-10-30: twee schalen - grof (richting) + fijn (echt jouw loopje)
            r = 0.5 * r + 0.5 * torch.exp(-e / std_fine**2)
        return r * _moving(env, command_name).float()


class contact_schedule:
    """Fraction of feet whose ground contact matches the reference schedule at the
    current phase (contact/no contact per foot, from walk_cycle.npz). Pins the cadence:
    without it the air_time reward pulls the policy to quick shuffling steps."""

    def __init__(self, cfg, env):
        pass

    def __call__(self, env, sensor_name: str = "feet_ground_contact",
                 command_name: str = "twist") -> torch.Tensor:
        ref = _ref(env)
        found = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
        match = (found == ref.contact_now(env)).float().mean(dim=-1)
        return match * _moving(env, command_name).float()


class swing_clearance:
    """Foot lift during the reference swing phase (2026-10-13). Without it the policy
    shuffles: feet hardly leave the floor (3-4 mm steps), they slide, and the trunk
    hardly moves forward. For each foot that the walk cycle has in the air, reward its
    height above the other foot, saturating at `target` (the Blender cycle lifts 15-22 mm;
    10 mm is enough and stays within the servo speeds)."""

    def __init__(self, cfg, env):
        asset = env.scene["robot"]
        self.site_ids = asset.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]

    def __call__(self, env, target: float = 0.010, command_name: str = "twist") -> torch.Tensor:
        ref = _ref(env)
        swing = (~ref.contact_now(env)).float()                          # (N, 2)
        z = env.scene["robot"].data.site_pos_w[:, self.site_ids, 2]      # (N, 2)
        lift = torch.stack([z[:, 0] - z[:, 1], z[:, 1] - z[:, 0]], dim=-1)
        if USE_GAIT_GRID:   # 2026-10-29: doel = 0.8 x voethoogte van de stijl
            target = ref.lift_s[env_style(env)][:, None]
        r = torch.clamp(lift / target, 0.0, 1.0) * swing
        n = swing.sum(-1)
        r = torch.where(n > 0, r.sum(-1) / n.clamp(min=1), torch.zeros_like(n))
        return r * _moving(env, command_name).float()


class _Reference:
    """walk_cycle.npz on the env device, with phase interpolation."""

    def __init__(self, env):
        data = np.load(WALK_CYCLE_FILE)
        names = [str(n) for n in data["joint_names"]]
        asset = env.scene["robot"]
        self.leg_cols = [names.index(j) for j in LEG_JOINTS]
        self.leg_ids = torch.tensor(asset.find_joints(LEG_JOINTS, preserve_order=True)[0],
                                    device=env.device)
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=env.device)
        self.q = f(data["joint_pos"][:, self.leg_cols])                      # (T, 10)
        self.dq = (torch.roll(self.q, -1, 0) - self.q) * float(data["fps"])  # (T, 10)
        self.height = f(data["root_height"])                                 # (T,)
        self.contact = torch.tensor(data["contact"], dtype=torch.bool, device=env.device)
        self.vel_xy = f(data["vel_xy"])
        self.n = self.q.shape[0]
        _gait_phase.period = float(data["cycle_s"])

    def at(self, x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        ph = phase * self.n
        i0 = ph.long() % self.n
        w = (ph - ph.floor()).view(-1, *([1] * (x.dim() - 1)))
        return (1 - w) * x[i0] + w * x[(i0 + 1) % self.n]

    # accessors shared with _GridReference
    def q_now(self, env):
        return self.at(self.q, _gait_phase(env))

    def height_now(self, env):
        return self.at(self.height, _gait_phase(env))

    def contact_now(self, env):
        i = (_gait_phase(env) * self.n).long() % self.n
        return self.contact[i]

    def reset_state(self, env, env_ids, phase):
        """(leg q, leg dq, trunk height, forward speed) at `phase` for a reset."""
        return self.at(self.q, phase), self.at(self.dq, phase), self.at(self.height, phase), float(self.vel_xy[0])


class _GridReference:
    """gait_grid.npz on the env device. Per env: the gait for the current twist command
    (speed = vx / v100, turn = wz in deg/s, side 0), blended over the 8 grid neighbours
    and towards standing for tiny commands, with a phase that runs at that gait's cadence."""

    def __init__(self, env):
        # 2026-10-29: alle stijl-roosters achter elkaar (zelfde assen), index = stijl * G + gang
        grids = [np.load(r["file"]) for r in style_tables()]
        d = grids[0]
        dev = env.device
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        names = [str(n) for n in d["joint_names"]]
        cols = [names.index(j) for j in LEG_JOINTS]
        hcols = [names.index(j) for j in HEAD_JOINTS]
        asset = env.scene["robot"]
        self.leg_ids = torch.tensor(asset.find_joints(LEG_JOINTS, preserve_order=True)[0], device=dev)
        for g in grids[1:]:
            for k in ("speeds", "turns", "sides", "stand_joint_pos"):
                assert np.allclose(g[k], d[k]), f"stijl-roosters verschillen in {k}"
            assert [str(n) for n in g["joint_names"]] == names
        self.S, self.W, self.Y = f(d["speeds"]), f(d["turns"]), f(d["sides"])
        nS, nW, nY = len(self.S), len(self.W), len(self.Y)
        self.dims = (nS, nW, nY)
        G = nS * nW * nY
        self.G = G
        T = max(np.asarray(g["joint_pos"]).shape[3] for g in grids)

        def stack(key, shape_tail):
            out = []
            for g in grids:
                a = np.asarray(g[key]).reshape(G, -1, *shape_tail)
                pad = np.zeros((G, T - a.shape[1], *shape_tail))
                out.append(np.concatenate([a, pad], 1))
            return np.concatenate(out, 0)

        jp = stack("joint_pos", (len(names),))
        self.q = f(jp[..., cols])                                           # (nStyle*G, T, 10)
        self.h = f(stack("root_height", ()))                                # (nStyle*G, T)
        self.c = f(stack("contact", (2,)))                                  # (nStyle*G, T, 2)
        self.head = f(jp[..., hcols])                                       # (nStyle*G, T, 4)
        self.length = torch.tensor(np.concatenate([np.asarray(g["length"]).reshape(-1) for g in grids]),
                                   dtype=torch.long, device=dev)
        self.cycle = f(np.concatenate([np.asarray(g["cycle_s"]).reshape(-1) for g in grids]))
        self.vel = f(np.concatenate([np.asarray(g["vel"]).reshape(G, 2) for g in grids]))
        self.stand_q = f(np.asarray(d["stand_joint_pos"])[cols])
        self.stand_head = f(np.asarray(d["stand_joint_pos"])[hcols])
        self.stand_h = float(d["stand_root_height"])
        self.v100_s = f([float(g["v100"]) for g in grids])                 # per stijl
        i100 = (int(np.argmin(np.abs(np.asarray(d["speeds"]) - 1.0))) * nW
                + int(np.argmin(np.abs(np.asarray(d["turns"])))) ) * nY + int(np.argmin(np.abs(np.asarray(d["sides"]))))
        self.i100_s = torch.tensor([s * G + i100 for s in range(len(grids))], dtype=torch.long, device=dev)
        self.lift_s = f([0.8 * r["lift"] for r in style_tables()])
        self.width_s = f([r["width"] for r in style_tables()])
        self.phase = torch.zeros(env.num_envs, device=dev)
        self.dt = env.step_dt
        self.last_step = -1
        self._update(env)

    def _axis(self, ax, x):
        x = torch.clamp(x, float(ax[0]), float(ax[-1]))
        i = torch.clamp(torch.searchsorted(ax, x.contiguous()) - 1, 0, len(ax) - 2)
        return i, (x - ax[i]) / (ax[i + 1] - ax[i])

    def _update(self, env):
        """Once per control step: gait for the current command, then the clock moves on."""
        if env.common_step_counter == self.last_step:
            return
        first = self.last_step < 0
        self.last_step = env.common_step_counter
        cmd = env.command_manager.get_command("twist")
        self.style = env_style(env)
        s = cmd[:, 0] / self.v100_s[self.style]      # negative = backwards
        turn = torch.rad2deg(cmd[:, 2])
        side = cmd[:, 1] * SIDE_CYCLE_S               # m/s -> m per cycle (grid axis)
        (i, wi), (j, wj), (k, wk) = self._axis(self.S, s), self._axis(self.W, turn), self._axis(self.Y, side)
        nS, nW, nY = self.dims
        idx, wts = [], []
        for a, wa in ((i, 1 - wi), (i + 1, wi)):
            for b, wb in ((j, 1 - wj), (j + 1, wj)):
                for c, wc in ((k, 1 - wk), (k + 1, wk)):
                    idx.append(self.style * self.G + (a * nW + b) * nY + c); wts.append(wa * wb * wc)
        self.idx, self.wts = torch.stack(idx, 1), torch.stack(wts, 1)
        self.walk = torch.clamp(torch.maximum(torch.maximum(s.abs() / 0.15, turn.abs() / 30.0),
                                              side.abs() / 0.006), 0.0, 1.0)
        cyc = (self.wts * self.cycle[self.idx]).sum(1)
        if not first:
            self.phase = torch.remainder(self.phase + self.dt / cyc, 1.0)

    def _sample(self, table, phase, idx=None, wts=None):
        idx = self.idx if idx is None else idx
        wts = self.wts if wts is None else wts
        out = 0.0
        for c in range(idx.shape[1]):
            g = idx[:, c]
            n = self.length[g]
            fph = phase * n
            i0 = fph.long() % n
            i1 = (i0 + 1) % n
            w = (fph - fph.floor()).view(-1, *([1] * (table.dim() - 2)))
            wc = wts[:, c].view(-1, *([1] * (table.dim() - 2)))
            out = out + wc * ((1 - w) * table[g, i0] + w * table[g, i1])
        return out

    def phase_now(self, env):
        self._update(env)
        return self.phase

    def q_now(self, env):
        self._update(env)
        w = self.walk[:, None]
        return w * self._sample(self.q, self.phase) + (1 - w) * self.stand_q

    def height_now(self, env):
        self._update(env)
        return self.walk * self._sample(self.h, self.phase) + (1 - self.walk) * self.stand_h

    def head_offset_now(self, env):
        """Rhythmic head motion of your walk (nod, sway, follow-through) relative to the
        standing head pose, faded out when standing. Order: HEAD_JOINTS."""
        self._update(env)
        return self.walk[:, None] * (self._sample(self.head, self.phase) - self.stand_head)

    def contact_now(self, env):
        self._update(env)
        c = self._sample(self.c, self.phase)
        return (c > 0.5) | (self.walk[:, None] < 0.5)

    def reset_state(self, env, env_ids, phase):
        """Reset pose from the env's style at 100%, straight (the command may not be set yet)."""
        i = self.i100_s[env_style(env)[env_ids]]
        idx = i[:, None]
        wts = torch.ones(len(env_ids), 1, device=phase.device)
        dph = self.dt / self.cycle[i]
        q = self._sample(self.q, phase, idx, wts)
        q1 = self._sample(self.q, torch.remainder(phase + dph, 1.0), idx, wts)
        h = self._sample(self.h, phase, idx, wts)
        return q, (q1 - q) / self.dt, h, self.vel[i, 0]


def _ref(env):
    r = getattr(env, "_duck_reference", None)
    if r is None:
        r = _GridReference(env) if USE_GAIT_GRID else _Reference(env)
        env._duck_reference = r
    return r


def _moving(env, command_name: str = "twist") -> torch.Tensor:
    cmd = env.command_manager.get_command(command_name)
    return (torch.norm(cmd[:, :2], dim=-1) > 0.02) | (cmd[:, 2].abs() > 0.1)


def reset_to_reference(env, env_ids: torch.Tensor, prob: float = 0.8):
    """Reference state initialisation: with probability `prob` the episode starts at a
    random phase of the walk cycle, in the reference leg pose and trunk height, already
    moving at the reference speed. Otherwise it starts in HOME at phase 0 (as on the robot)."""
    if env_ids is None or len(env_ids) == 0:
        return
    tw = env.command_manager.get_term("twist")
    if hasattr(tw, "_pick_style"):            # stijl van de nieuwe episode vóór de startpose
        tw._pick_style(env_ids, from_event=True)
    ref = _ref(env)
    asset = env.scene["robot"]
    k = len(env_ids)
    use = torch.rand(k, device=env.device) < prob
    phase = torch.where(use, torch.rand(k, device=env.device), torch.zeros(k, device=env.device))
    _phase_offset(env)[env_ids] = phase
    if USE_GAIT_GRID:
        ref.phase[env_ids] = phase                       # the grid keeps its own clock
    sel = env_ids[use]
    if len(sel) == 0:
        return
    ph = phase[use]
    q_leg, dq_leg, h_ref, v = ref.reset_state(env, sel, ph)
    q = asset.data.joint_pos[sel].clone()
    dq = torch.zeros_like(q)
    q[:, ref.leg_ids] = q_leg
    dq[:, ref.leg_ids] = dq_leg
    asset.write_joint_state_to_sim(q, dq, env_ids=sel)
    qpos = asset.data.data.qpos
    adr = asset.data.indexing.free_joint_q_adr
    pose = qpos[sel][:, adr].clone()                        # x y z qw qx qy qz
    pose[:, 2] = h_ref + 0.001
    qw, qz = pose[:, 3], pose[:, 6]
    yaw = 2 * torch.atan2(qz, qw)                          # spawn is level, yaw only
    vel = torch.zeros(len(sel), 6, device=env.device)
    vel[:, 0] = v * torch.cos(yaw)     # v: tensor (grid, per stijl) of float (walk_cycle)
    vel[:, 1] = v * torch.sin(yaw)
    asset.write_root_link_pose_to_sim(pose, env_ids=sel)
    asset.write_root_link_velocity_to_sim(vel, env_ids=sel)


def head_pose_with_style(env, command_name: str = "head_pose", std: float = 0.5,
                         fine_std: float = 0.1, fine_weight: float = 0.5) -> torch.Tensor:
    """head_pose_tracking, but the target is the head command (looking around) PLUS the
    rhythmic head motion of your walk at the current gait and phase (2026-10-13: the head
    was the least convincing part - it only followed the look command, never the walk).
    Mean over the 4 head joints of a wide + narrow Gaussian on the error."""
    asset = env.scene["robot"]
    ids = getattr(env, "_duck_head_ids", None)
    if ids is None:
        ids = torch.tensor(asset.find_joints(HEAD_JOINTS, preserve_order=True)[0], device=env.device)
        env._duck_head_ids = ids
    cmd = head_command_when_moving(env, command_name)                        # (N, 4), 0 at rest
    target = cmd + (_ref(env).head_offset_now(env) if USE_GAIT_GRID else 0.0)
    err = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids] - target
    per = (1 - fine_weight) * torch.exp(-(err / std) ** 2) + fine_weight * torch.exp(-(err / fine_std) ** 2)
    return per.mean(-1)


def head_level(env, command_name: str = "head_pose", std: float = 0.3, fine_std: float = 0.06) -> torch.Tensor:
    """volg16b: head_yaw and head_roll exactly on their target (look command + walk style).
    volg16 held its head 8-12 deg yawed and 4-15 deg rolled to the LEFT while walking; in the
    4-joint mean of head_pose_tracking that cost almost nothing. Own term, narrow scales:
    0 deg -> 1.0, 5 -> 0.52, 10 -> 0.36, 15 -> 0.23, 20 -> 0.13, 30 -> 0.02."""
    asset = env.scene["robot"]
    ids = getattr(env, "_duck_head_ids", None)
    if ids is None:
        ids = torch.tensor(asset.find_joints(HEAD_JOINTS, preserve_order=True)[0], device=env.device)
        env._duck_head_ids = ids
    cmd = head_command_when_moving(env, command_name)
    target = cmd + (_ref(env).head_offset_now(env) if USE_GAIT_GRID else 0.0)
    err = (asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids] - target)[:, 2:4]   # yaw, roll
    per = 0.5 * torch.exp(-(err / std) ** 2) + 0.5 * torch.exp(-(err / fine_std) ** 2)
    return per.mean(-1)


def stand_still(env, std: float = 0.01, sensor_name: str = "feet_ground_contact",
                command_name: str = "twist") -> torch.Tensor:
    """Commanded to stand (2026-10-13: at vx = 0 it kept waddling): trunk not moving
    (exp(-|v_xy|^2 / std^2), std 1 cm/s) x fraction of feet on the floor x calm joints."""
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    found = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    qd = env.scene["robot"].data.joint_vel                       # incl. the head: no nodding
    calm = torch.exp(-torch.mean(torch.square(qd), -1) / 0.25**2)  # rad/s (0.5 still let ankles nod)
    r = torch.exp(-torch.sum(torch.square(v), -1) / std**2) * found.float().mean(-1) * calm
    return r * (~_moving(env, command_name)).float()


# --- Follow a target (2026-10-15) ----------------------------------------------------
# The policy no longer gets a velocity command but SEES a target, as a camera would:
# bearing (rad, + = left), distance - FOLLOW_DIST (m) and "visible", with noise, a 60 deg
# field of view, 60-100 ms delay and dropouts. Out of view or dropped it gets the last
# seen values (visible = 0). The twist command still exists - a P-controller on the TRUE
# target position - and drives the gait grid, the imitation and the tracking rewards,
# so everything else stays as it was. Joystick on the robot = a virtual target.
FOLLOW_DIST = 0.5          # m
FOV_HALF = math.radians(30.0)
BEARING_NOISE = math.radians(2.0)
DIST_NOISE = 0.15          # relative
DROPOUT = 0.05             # per control step
K_DIST, K_BEARING = 1.0, 1.5
# 2026-10-15: calm commands. REST (cmd 0) when the person stands and the duck is within
# REST_IN of 50 cm and REST_IN_B of facing it; it only leaves REST beyond REST_OUT / REST_OUT_B.
REST_IN, REST_OUT, REST_IN_B, REST_OUT_B = 0.06, 0.12, 0.10, 0.25
ACC_V, ACC_W = 0.15, 1.5         # m/s^2, rad/s^2: the command changes smoothly (no gait flipping)
BACK_FACE = 0.8                  # rad (~45 deg): back away only while roughly facing the person
WAIT_FAR, WAIT_NEAR = 0.8, 0.6   # m: the "person" stops when the duck is further than WAIT_FAR
                                 # behind, and walks on once it is back within WAIT_NEAR
TARGET_MAX_V = 0.09              # m/s (V1 0.13; V2: normaal 100% = ~9 cm/s, parmantig ~7.4)
BACK_DIST = 0.36                 # m: closer than this the duck steps back (carefully); between
                                 # BACK_DIST and FOLLOW_DIST it just stands its ground
BACK_MAX_V = 0.06                # m/s backwards (grid goes to -50 % = -0.0625)
WALK_MAX_V = 0.085               # m/s: normal following = walking (~70 % of the grid: 4.8 steps/s)
RUN_DIST, RUN_RELEASE = 0.75, 0.60  # m: further than RUN_DIST -> RUN (afraid to lose it) until
RUN_V = 0.135                    # back within RUN_RELEASE; run = ~110 % of the grid (~9-10 steps/s)
BACK_MIN_V = 0.035               # m/s: back up at a real stepping speed, never a creep (a creep
BACK_RELEASE = 0.42              # sits between stand and walk in the grid = trembling); stop
                                 # backing only beyond BACK_RELEASE (hysteresis)


class TargetFollowCommand(_StyleMixin, VelocityCommandCommandOnly):
    """Target in the world moving as a unicycle (speed, turn rate), re-sampled every few
    seconds: stopped (curriculum's rel_standing_envs), walking/curving, or jumping to a
    spot next to/behind the robot (turn on the spot). cmd = P-controller towards it."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self.tgt = torch.zeros(n, 2, device=dev)
        self.tgt_yaw = torch.zeros(n, device=dev)
        self.tgt_v = torch.zeros(n, device=dev)
        self.tgt_w = torch.zeros(n, device=dev)
        self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
        self.waiting = torch.zeros(n, dtype=torch.bool, device=dev)
        self.running = torch.zeros(n, dtype=torch.bool, device=dev)
        self.resting = torch.zeros(n, dtype=torch.bool, device=dev)
        self.backing = torch.zeros(n, dtype=torch.bool, device=dev)
        self.dist = torch.full((n,), FOLLOW_DIST, device=dev)
        self.bearing = torch.zeros(n, device=dev)
        self.held = torch.stack([torch.zeros(n, device=dev), torch.full((n,), FOLLOW_DIST, device=dev)], -1)
        # fewer graphs: the stock velocity-error metrics are meaningless here (always 0)
        self.metrics.pop("error_vel_xy", None)
        self.metrics.pop("error_vel_yaw", None)
        self.metrics["follow_dist_err"] = torch.zeros(n, device=dev)
        self.metrics["follow_bearing_err"] = torch.zeros(n, device=dev)
        self._init_style()

    def reset(self, env_ids):
        self.fresh[env_ids] = True
        self._pick_style(env_ids)
        return super().reset(env_ids)

    def _place(self, ids, bearing):
        pos = self.robot.data.root_link_pos_w[ids, :2]
        h = self.robot.data.heading_w[ids] + bearing
        self.tgt[ids] = pos + FOLLOW_DIST * torch.stack([torch.cos(h), torch.sin(h)], -1)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        k, dev = len(env_ids), self.device
        u = lambda a, b: torch.empty(k, device=dev).uniform_(a, b)
        fresh = self.fresh[env_ids]
        if fresh.any():
            ids = env_ids[fresh]
            self._place(ids, torch.empty(len(ids), device=dev).uniform_(-0.6, 0.6))
            self.tgt_yaw[ids] = self.robot.data.heading_w[ids] + torch.empty(len(ids), device=dev).uniform_(-1.0, 1.0)
            self.fresh[ids] = False
        r = torch.rand(k, device=dev)
        p0 = self.cfg.rel_standing_envs
        stand = r < p0
        jump = (r >= p0) & (r < p0 + 0.15)
        toward = (r >= p0 + 0.15) & (r < p0 + 0.35)      # 2026-10-15: walks towards / past the duck (20 %)
        close = (r >= p0 + 0.35) & (r < p0 + 0.45)       # 2026-10-15: stands TOO close (15-28 cm): back off
        self.tgt_v[env_ids] = torch.where(stand | jump | close, torch.zeros(k, device=dev), u(0.0, TARGET_MAX_V))
        self.tgt_w[env_ids] = torch.where(stand | jump | toward, torch.zeros(k, device=dev), u(-0.5, 0.5))
        if toward.any():
            ids = env_ids[toward]; m = len(ids)
            self._place(ids, torch.empty(m, device=dev).uniform_(-0.4, 0.4))
            self.tgt[ids] = self.robot.data.root_link_pos_w[ids, :2] + 2.0 * (self.tgt[ids] - self.robot.data.root_link_pos_w[ids, :2])  # 1 m away
            to_duck = self.robot.data.root_link_pos_w[ids, :2] - self.tgt[ids]
            self.tgt_yaw[ids] = torch.atan2(to_duck[:, 1], to_duck[:, 0]) + torch.empty(m, device=dev).uniform_(-0.4, 0.4)
            self.tgt_v[ids] = torch.empty(m, device=dev).uniform_(0.02, 0.08)
        if close.any():
            ids = env_ids[close]; m = len(ids)
            self._place(ids, torch.empty(m, device=dev).uniform_(-0.3, 0.3))
            pos = self.robot.data.root_link_pos_w[ids, :2]
            self.tgt[ids] = pos + (self.tgt[ids] - pos) * torch.empty(m, 1, device=dev).uniform_(0.30, 0.56)  # 15-28 cm
        if jump.any():
            ids = env_ids[jump]
            side = torch.where(torch.rand(len(ids), device=dev) < 0.5, -1.0, 1.0)
            self._place(ids, side * torch.empty(len(ids), device=dev).uniform_(0.6, 2.8))
        self._relative(ids=env_ids[fresh])
        self.held[env_ids] = torch.stack([self.bearing[env_ids], self.dist[env_ids]], -1)

    def _relative(self, smooth: bool = False, ids=None):
        rel = self.tgt - self.robot.data.root_link_pos_w[:, :2]
        h = self.robot.data.heading_w
        c, s_ = torch.cos(h), torch.sin(h)
        dx, dy = c * rel[:, 0] + s_ * rel[:, 1], -s_ * rel[:, 0] + c * rel[:, 1]
        self.dist = torch.sqrt(dx * dx + dy * dy)
        self.bearing = torch.atan2(dy, dx)
        # P-controller on the TRUE target: turn first, walk when roughly facing it
        v_now = torch.where(self.waiting, torch.zeros_like(self.tgt_v), self.tgt_v)
        moving_tgt = v_now > 1e-4
        along = v_now * torch.cos(self.tgt_yaw - h)
        face = torch.clamp(1.0 - self.bearing.abs() / 1.0, 0.0, 1.0)
        vx = torch.clamp((K_DIST * (self.dist - FOLLOW_DIST) + along) * face, min=0.0)
        wz = K_BEARING * self.bearing
        # rest, with hysteresis: in close to 50 cm and facing it, out only well beyond
        err, berr = (self.dist - FOLLOW_DIST).abs(), self.bearing.abs()
        self.resting = (~moving_tgt) & (((err < REST_IN) & (berr < REST_IN_B))
                                        | (self.resting & (err < REST_OUT) & (berr < REST_OUT_B)))
        vx = torch.where(self.resting, torch.zeros_like(vx), vx)
        wz = torch.where(self.resting, torch.zeros_like(wz), wz)
        # person too close -> step back, decisively (min speed + hysteresis), while roughly facing it
        self.backing = (self.dist < BACK_DIST) | (self.backing & (self.dist < BACK_RELEASE))
        back_v = -torch.clamp(K_DIST * (BACK_RELEASE - self.dist), BACK_MIN_V, BACK_MAX_V)
        back_v = torch.maximum(back_v, -0.5 * self._v100[self.style])
        vx = torch.where(self.backing, back_v * (berr < BACK_FACE).float(), vx)
        lo_w, hi_w = self.cfg.ranges.ang_vel_z
        # 2026-10-29: per stijl - vooruit tot 100% van de stijl, achteruit tot -50% (rooster)
        v100 = self._v100[self.style]
        raw_v, raw_w = torch.maximum(torch.minimum(vx, v100), -0.5 * v100), torch.clamp(wz, lo_w, hi_w)
        cmd = self.vel_command_b
        if smooth:                       # once per control step: rate-limited
            dt = self._env.step_dt
            cmd[:, 0] += torch.clamp(raw_v - cmd[:, 0], -ACC_V * dt, ACC_V * dt)
            cmd[:, 2] += torch.clamp(raw_w - cmd[:, 2], -ACC_W * dt, ACC_W * dt)
            # snap to exactly 0 once it has ramped down into rest (so the rest mode engages)
            cmd[:, 0] = torch.where(self.resting & (cmd[:, 0].abs() < 0.01), torch.zeros_like(raw_v), cmd[:, 0])
            cmd[:, 2] = torch.where(self.resting & (cmd[:, 2].abs() < 0.05), torch.zeros_like(raw_w), cmd[:, 2])
        elif ids is not None:            # new episode / new target: start at the right command
            cmd[ids, 0], cmd[ids, 2] = raw_v[ids], raw_w[ids]
        cmd[:, 1] = 0.0

    def _update_command(self):
        dt = self._env.step_dt
        # a person waits when the duck falls behind (the twist uses the speed it walks now)
        rel = self.tgt - self.robot.data.root_link_pos_w[:, :2]
        away = torch.cos(self.tgt_yaw) * rel[:, 0] + torch.sin(self.tgt_yaw) * rel[:, 1] > 0   # walking away from the duck
        self.waiting = ((self.waiting & (self.dist > WAIT_NEAR)) | (self.dist > WAIT_FAR)) & away
        v = torch.where(self.waiting, torch.zeros_like(self.tgt_v), self.tgt_v)
        self.tgt_yaw += torch.where(self.waiting, torch.zeros_like(self.tgt_w), self.tgt_w) * dt
        self.tgt += (v * dt)[:, None] * torch.stack([torch.cos(self.tgt_yaw), torch.sin(self.tgt_yaw)], -1)
        self._relative(smooth=True)

    def _update_metrics(self):
        self.metrics["follow_dist_err"] += (self.dist - FOLLOW_DIST).abs() * self._env.step_dt / self.cfg.resampling_time_range[1]
        self.metrics["follow_bearing_err"] += self.bearing.abs() * self._env.step_dt / self.cfg.resampling_time_range[1]


@_dataclass(kw_only=True)
class TargetFollowCommandCfg(VelocityCommandCommandOnlyCfg):
    def build(self, env):
        return TargetFollowCommand(self, env)


def target_obs(env, command_name: str = "twist") -> torch.Tensor:
    """What a camera gives: [bearing, distance - FOLLOW_DIST, visible] (see above)."""
    t = env.command_manager.get_term(command_name)
    n = env.num_envs
    b = t.bearing + BEARING_NOISE * torch.randn(n, device=env.device)
    d = t.dist * (1.0 + DIST_NOISE * torch.randn(n, device=env.device))
    vis = (t.bearing.abs() < FOV_HALF) & (torch.rand(n, device=env.device) > DROPOUT)
    t.held = torch.where(vis[:, None], torch.stack([b, d], -1), t.held)
    return torch.stack([t.held[:, 0], torch.clamp(t.held[:, 1] - FOLLOW_DIST, -0.5, 1.0), vis.float()], -1)


def follow_distance(env, std_fine: float = 0.05, std_coarse: float = 0.25, command_name: str = "twist") -> torch.Tensor:
    """The result counts, not the gait: true distance to the target. Narrow + wide Gaussian
    (half each) so it pays to get a bit closer even when far off."""
    t = env.command_manager.get_term(command_name)
    # 2026-10-15: no error between BACK_DIST and FOLLOW_DIST (it may stand its ground there)
    e = torch.square(torch.clamp(t.dist - FOLLOW_DIST, min=0.0)) + torch.square(torch.clamp(BACK_DIST - t.dist, min=0.0))
    r = 0.5 * torch.exp(-e / std_fine**2) + 0.5 * torch.exp(-e / std_coarse**2)
    # 2026-10-15: only counts while facing it - no points for crabbing towards it sideways
    return r * torch.clamp(torch.cos(t.bearing), min=0.0)


def follow_bearing(env, std_fine: float = 0.15, command_name: str = "twist") -> torch.Tensor:
    """True heading to the target: half a narrow Gaussian (0.15 rad = 9 deg) for aiming, half
    LINEAR in |bearing| (1 facing it, 0 facing away) so every degree towards it pays, also
    when the target is behind (a Gaussian is flat there)."""
    t = env.command_manager.get_term(command_name)
    b = t.bearing.abs()
    return 0.5 * torch.exp(-torch.square(b / std_fine)) + 0.5 * (1.0 - b / math.pi)


def turn_toward(env, full: float = 0.3, max_rate: float = 1.0, command_name: str = "twist") -> torch.Tensor:
    """2026-10-15: d(yaw)/dt towards the target, rewarded hard: + yaw rate (rad/s) towards the
    target, - away from it. Scaled by min(|bearing| / full, 1): full effect from 17 deg off,
    fading to 0 when facing it (no reward for spinning past it - overshooting flips the sign)."""
    t = env.command_manager.get_term(command_name)
    w = torch.clamp(env.scene["robot"].data.root_link_ang_vel_w[:, 2], -max_rate, max_rate)
    return torch.sign(t.bearing) * w * torch.clamp(t.bearing.abs() / full, max=1.0)


class push_off:
    """2026-10-15: reward the feet PUSHING BACK on the floor (= the floor pushing the duck
    forward) while it has to catch up: facing the target, more than 5 cm too far, and slower
    than asked. Net horizontal ground force along the heading / body weight, 0..0.5 -> 0..1.
    The 'slower than asked' gate stops it from earning by surging and braking."""

    def __init__(self, cfg, env):
        asset = env.scene["robot"]
        self.body_idx = asset.indexing.body_ids

    def __call__(self, env, sensor_name: str = "feet_ground_contact", command_name: str = "twist") -> torch.Tensor:
        asset = env.scene["robot"]
        data = env.scene.sensors[sensor_name].data
        f = data.force.reshape(env.num_envs, -1, 3)[:, :2]                   # (N, 2, 3) global
        f = f * torch.sign(f[..., 2:3])                                      # force ON the foot (floor pushes up)
        q = asset.data.root_link_quat_w
        h = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        fwd = (torch.cos(h)[:, None] * f[..., 0] + torch.sin(h)[:, None] * f[..., 1]).sum(-1)
        weight = env.sim.model.body_mass[:, self.body_idx].sum(-1) * 9.81
        t = env.command_manager.get_term(command_name)
        need = (torch.clamp((t.dist - FOLLOW_DIST - 0.05) / 0.05, 0.0, 1.0)
                * torch.clamp((0.5 - t.bearing.abs()) / (0.5 - 0.17), 0.0, 1.0))
        v, _ = _yaw_frame_vel(env)
        slow = (v[:, 0] < env.command_manager.get_command(command_name)[:, 0]).float()
        return torch.clamp(fwd / weight, 0.0, 0.5) / 0.5 * need * slow


def back_off(env, command_name: str = "twist") -> torch.Tensor:
    """2026-10-15: while it should back away (person < 30 cm), reward actually moving
    backwards: 0..1 for 0..3.5 cm/s back. volg10 walked INTO the person (+3-4 cm/s) at a
    -6 cm/s command, trampling ~13 steps/s."""
    t = env.command_manager.get_term(command_name)
    v, _ = _yaw_frame_vel(env)
    return torch.clamp(-v[:, 0] / BACK_MIN_V, 0.0, 1.0) * t.backing.float()


def spin_limit(env, max_rate: float = 1.0) -> torch.Tensor:
    """2026-10-15: penalty on yaw rate above max_rate (rad/s): it spun 60 deg in 0.3 s
    (~3.5 rad/s) towards a target and then hunted +-15 deg - too violent for the servos."""
    w = env.scene["robot"].data.root_link_ang_vel_w[:, 2]
    return torch.clamp(w.abs() - max_rate, min=0.0)


class toe_out:
    """2026-10-15: penalty for feet turned out (V-stance): per foot the sole's yaw relative to
    the trunk heading, outwards positive, beyond `free` rad (12 deg). volg14 stood and walked
    with both feet ~29 deg out (hip yaw at its limit). Turning keeps its +-10 deg room."""

    def __init__(self, cfg, env):
        self.ids = env.scene["robot"].find_bodies(["L_toe", "R_toe"], preserve_order=True)[0]

    # 2026-10-29: ook NAAR BINNEN (duiven-tenen): run v2-servo liep met beide heup-yaws op -23 gr
    # (tenen in een punt), want alleen naar buiten werd bestraft. Binnen 15 gr vrij (bochten: 13 gr).
    def __call__(self, env, free: float = 0.21, free_in: float = 0.26) -> torch.Tensor:
        asset = env.scene["robot"]
        yaw = lambda q: torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
                                    1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))
        h = yaw(asset.data.root_link_quat_w)
        f = yaw(asset.data.body_link_quat_w[:, self.ids])                    # (N, 2)
        rel = torch.atan2(torch.sin(f - h[:, None]), torch.cos(f - h[:, None]))
        out = torch.stack([rel[:, 0], -rel[:, 1]], -1)                      # left +, right - = outwards
        return (torch.clamp(out - free, min=0.0) + torch.clamp(-out - free_in, min=0.0)).sum(-1)


class feet_parallel:
    """2026-10-30 (Jairo): BELONING voor tenen recht vooruit, zoals jouw loopje: per voet de yaw van
    de zool t.o.v. de romp (naar buiten +) tegen die van de referentie (= referentie-heup-yaw; in
    bochten tot 13 gr). exp(-fout^2/std^2), gemiddeld over de twee voeten. Run 900 stond in V
    (+22 gr), run 500 met tenen naar binnen (-14 gr)."""

    def __init__(self, cfg, env):
        asset = env.scene["robot"]
        self.ids = asset.find_bodies(["L_toe", "R_toe"], preserve_order=True)[0]
        self.hy = [C_LEG.index(f"{s}_hip_yaw") for s in ("left", "right")]

    def __call__(self, env, std: float = 0.15, command_name: str = "twist") -> torch.Tensor:
        asset = env.scene["robot"]
        yaw = lambda q: torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
                                    1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))
        h = yaw(asset.data.root_link_quat_w)
        f = yaw(asset.data.body_link_quat_w[:, self.ids])
        rel = torch.atan2(torch.sin(f - h[:, None]), torch.cos(f - h[:, None]))
        out = torch.stack([rel[:, 0], -rel[:, 1]], -1)                      # naar buiten +
        ref = _ref(env).q_now(env)[:, self.hy]                               # heup-yaw ref = voet-yaw ref
        return torch.exp(-torch.square(out - ref).mean(-1) / std**2)


C_LEG = LEG_JOINTS


def _yaw_frame_vel(env):
    """Trunk linear velocity (xy) in the heading frame and yaw rate (world z)."""
    asset = env.scene["robot"]
    q = asset.data.root_link_quat_w
    yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    v = asset.data.root_link_lin_vel_w[:, :2]
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] + s_ * v[:, 1], -s_ * v[:, 0] + c * v[:, 1]], -1), asset.data.root_link_ang_vel_w[:, 2]


def track_lin_two_scale(env, std_fine: float, std_coarse: float, command_name: str = "twist") -> torch.Tensor:
    """2026-10-14: velocity tracking with a narrow AND a wide Gaussian (half each). With only
    std 0.03 m/s the reward was ~0 for anything under 6 cm/s at a 12 cm/s command: no
    gradient, so the policy never found out that walking a bit faster pays (it trampled)."""
    v, _ = _yaw_frame_vel(env)
    e = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - v), -1)
    return 0.5 * torch.exp(-e / std_fine**2) + 0.5 * torch.exp(-e / std_coarse**2)


def track_ang_two_scale(env, std_fine: float, std_coarse: float, command_name: str = "twist") -> torch.Tensor:
    """Same for the yaw rate (rad/s): not turning at a 29 deg/s command scored 0.02 of 1."""
    _, w = _yaw_frame_vel(env)
    e = torch.square(env.command_manager.get_command(command_name)[:, 2] - w)
    return 0.5 * torch.exp(-e / std_fine**2) + 0.5 * torch.exp(-e / std_coarse**2)


class com_forward:
    """2026-10-14: centre of mass AHEAD of the stance feet while walking forward, by the amount
    of your walk (gait_grid style: com_ahead_m at 100 %, 0 below 20 %, growing above 100 %).
    walk_imitation only scores joint angles: the policy copied the legs but kept the CoM over
    (or behind) the feet, lifted the foot early and hardly moved. Distance along the trunk
    heading between the whole-robot CoM and the mean of the foot sites in contact;
    exp(-((ahead - target) / std)^2), only for envs commanded to move."""

    def __init__(self, cfg, env):
        asset = env.scene["robot"]
        self.site_ids = asset.find_sites(("left_foot", "right_foot"), preserve_order=True)[0]
        self.body_idx = asset.indexing.body_ids
        # volg16b: CoM WITHOUT the head chain (65 of ~190 g): volg16 earned this reward by
        # hanging its head forward (neck +30 deg, head_pitch -28 deg) instead of leaning in
        self.head_cols = asset.find_bodies(["neck", "head_pitch", "head_roll", "head", "jaw"], preserve_order=True)[0]
        # 2026-10-29: per stijl (zwaartepunt-voor en 100%-snelheid uit het rooster van de stijl)
        self.com100_s = torch.tensor([r["com100"] for r in style_tables()], device=env.device)
        self.v100_s = torch.tensor([r["v100"] for r in style_tables()], device=env.device)

    def __call__(self, env, std: float = 0.004, sensor_name: str = "feet_ground_contact",
                 command_name: str = "twist") -> torch.Tensor:
        asset = env.scene["robot"]
        m = env.sim.model.body_mass[:, self.body_idx].clone()                # (N or 1, B)
        m[:, self.head_cols] = 0.0                                           # body CoM, head excluded
        p = asset.data.body_com_pos_w[:, :, :2]                              # (N, B, 2)
        com = (m[..., None] * p).sum(1) / m.sum(1, keepdim=True)            # (N, 2)
        found = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
        n = found.float().sum(-1)
        feet = asset.data.site_pos_w[:, self.site_ids, :2]                   # (N, 2, 2)
        support = (feet * found.float()[..., None]).sum(1) / n.clamp(min=1)[:, None]
        q = asset.data.root_link_quat_w
        yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
        ahead = torch.cos(yaw) * (com[:, 0] - support[:, 0]) + torch.sin(yaw) * (com[:, 1] - support[:, 1])
        st = env_style(env)
        com100, v100 = self.com100_s[st], self.v100_s[st]
        s = env.command_manager.get_command(command_name)[:, 0] / v100
        a = s.abs()                                    # symmetric: backwards -> CoM behind the feet
        into = torch.clamp((a - 0.2) / 0.8, 0.0, 1.0) * (1.0 + torch.clamp(a - 1.0, min=0.0))
        # 2026-10-15 (follow): lean in AT ONCE when facing the target and it is further than
        # FOLLOW_DIST - full target from 5 cm too far and within 10 deg, gone at 30 deg
        t = env.command_manager.get_term(command_name)
        if hasattr(t, "bearing"):
            need = (torch.clamp((t.dist - FOLLOW_DIST) / 0.05, 0.0, 1.0)
                    * torch.clamp((0.5 - t.bearing.abs()) / (0.5 - 0.17), 0.0, 1.0))
            s = torch.where(need > 0, torch.ones_like(s), s)
            into = torch.maximum(into, need)
            # ... and BEHIND the feet at once when the person is closer than BACK_DIST
            back = (torch.clamp((BACK_DIST - t.dist) / 0.05, 0.0, 1.0)
                    * torch.clamp((0.5 - t.bearing.abs()) / (0.5 - 0.17), 0.0, 1.0))
            s = torch.where(back > 0, -torch.ones_like(s), s)
            into = torch.maximum(into, back)
        # 2026-10-15: one-sided - only falling SHORT of the target costs (it reached ~half of it)
        short = torch.sign(s) * (torch.sign(s) * com100 * into - ahead)
        r = torch.exp(-torch.square(torch.clamp(short, min=0.0) / std))
        # 2026-10-15: leaning only pays while it actually WALKS that way. volg7 leaned 9.4 mm
        # forward standing still (enkel 66 % saturated, trunk 4 mm lower) at 0.5 cm/s.
        v, _ = _yaw_frame_vel(env)
        want = env.command_manager.get_command(command_name)[:, 0]
        progress = torch.where(want.abs() > 0.02, torch.clamp(v[:, 0] / want, 0.0, 1.0), torch.ones_like(want))
        r = r * progress
        return torch.where(n > 0, r, torch.zeros_like(r)) * _moving(env, command_name).float()


class stance_grip:
    """Feet that carry weight do not slide (2026-10-13: the feet kept slipping; the stock
    foot_slip penalty at -0.1 was ~0.003 per episode, i.e. nothing). Mean over the feet in
    contact of exp(-|v_xy|^2 / std^2), foot site velocity, std 2 cm/s."""

    def __init__(self, cfg, env):
        self.site_ids = env.scene["robot"].find_sites(("left_foot", "right_foot"), preserve_order=True)[0]

    def __call__(self, env, std: float = 0.02, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
        v = env.scene["robot"].data.site_lin_vel_w[:, self.site_ids, :2]           # (N, 2, 2)
        found = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
        r = torch.exp(-torch.sum(torch.square(v), -1) / std**2)
        n = found.float().sum(-1)
        return torch.where(n > 0, (r * found.float()).sum(-1) / n.clamp(min=1), torch.zeros_like(n))


def _behavior_head_command(base_cmd_cls):
    """Kopcommando-term: in gedrag-envs komt het kopcommando uit de toestandsmachine, begrensd op
    het huidige bereik van het kop-curriculum; anders het willekeurige commando."""

    class BehaviorHeadCommand(base_cmd_cls):
        @property
        def command(self):
            tw = self._env.command_manager.get_term("twist")
            if not hasattr(tw, "head_cmd"):
                return self._command
            lo = torch.tensor([r[0] for r in self.cfg.ranges], device=self.device)
            hi = torch.tensor([r[1] for r in self.cfg.ranges], device=self.device)
            beh = torch.maximum(torch.minimum(tw.head_cmd, hi), lo)
            return torch.where(tw.is_beh[:, None], beh, self._command)

    return BehaviorHeadCommand


from mjlab_microduck.tasks import mdp as _mdp

_BehaviorHeadPoseCommand = _behavior_head_command(_mdp.UniformPoseCommand)


@_dataclass(kw_only=True)
class _BehaviorHeadPoseCfg(_mdp.UniformPoseCommandCfg):
    def build(self, env):
        return _BehaviorHeadPoseCommand(self, env)


def head_command_when_moving(env, command_name: str = "head_pose") -> torch.Tensor:
    """The look-around command, but zero for envs told to stand still (2026-10-13: the head
    kept jittering at rest - standing envs were sent new random look targets every few
    seconds over the full range). Used for the observation AND the head reward."""
    cmd = env.command_manager.get_command(command_name)
    return cmd * _moving(env).float()[:, None]


def stand_action_rate(env) -> torch.Tensor:
    """Action change per step, only for envs told to stand still: at rest the targets
    must not wiggle (2026-10-13: ankles and head kept jittering)."""
    d = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(d), dim=1) * (~_moving(env)).float()


def trunk_height_tracking(env, std: float = 0.006) -> torch.Tensor:
    """exp(-(z - z_ref)^2 / std^2): reference trunk height while walking, HOME height
    (~56 mm) while standing. Stops the deep crouch."""
    ref = _ref(env)
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    z_ref = torch.where(_moving(env), ref.height_now(env),
                        torch.full_like(z, HOME_TRUNK_HEIGHT))
    return torch.exp(-torch.square(z - z_ref) / std**2)


def home_pose_standing(env, std: float = 0.15) -> torch.Tensor:
    """Leg joints at HOME, only for envs commanded to stand still."""
    asset = env.scene["robot"]
    ids = _ref(env).leg_ids
    err = asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]
    r = torch.exp(-torch.mean(torch.square(err), dim=-1) / std**2)
    return r * (~_moving(env)).float()


def flat_feet(env, sensor_name: str = "feet_ground_contact", std_deg: float = 8.0) -> torch.Tensor:
    """Soles flat on the floor while they carry weight (no ankle roll on this robot, so a
    tilted sole means standing on its edge). Mean over the feet in contact."""
    asset = env.scene["robot"]
    ids = getattr(env, "_duck_toe_ids", None)
    if ids is None:
        ids = asset.find_bodies(["L_toe", "R_toe"], preserve_order=True)[0]
        env._duck_toe_ids = ids
    quat = asset.data.body_link_quat_w[:, ids]                       # (N, 2, 4) wxyz
    x, y = quat[..., 1], quat[..., 2]
    cos_tilt = 1 - 2 * (x * x + y * y)                               # body z-axis . world z
    tilt = torch.acos(torch.clamp(cos_tilt, -1.0, 1.0))
    r = torch.exp(-torch.square(tilt / math.radians(std_deg)))
    found = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    n = found.float().sum(-1)
    return torch.where(n > 0, (r * found.float()).sum(-1) / n.clamp(min=1), torch.zeros_like(n))


def make_duckling_velocity_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    # --- robot + physics
    cfg.scene.entities = {"robot": DUCKLING_WALK_ROBOT_CFG}
    cfg.sim.mujoco.timestep = TIMESTEP
    cfg.decimation = DECIMATION
    cfg.events["reset_base"].params["pose_range"]["z"] = SPAWN_Z
    # Only the soles collide with the floor in the walk model, so nothing stops the
    # policy from sinking into a deep crouch with the trunk at floor level and
    # shuffling. End the episode when the trunk drops > ~12 mm below its stance height.
    cfg.terminations["too_low"] = TerminationTermCfg(
        func=envs_mdp.root_height_below_minimum,
        params={"minimum_height": MIN_TRUNK_HEIGHT},
    )
    # Head look command only while moving (standing = look straight, keep still).
    for grp in ("actor", "critic"):
        if "head_command" in cfg.observations[grp].terms:
            cfg.observations[grp].terms["head_command"] = ObservationTermCfg(
                func=head_command_when_moving, params={"command_name": "head_pose"})
    # Reference walk cycle: gait clock in the observations (actor 61 -> 63).
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms["gait_clock"] = ObservationTermCfg(func=gait_clock)
    # 2026-10-29: stijl (one-hot) in de observaties (63 -> 63 + len(STYLES))
    for grp in ("actor", "critic"):
        cfg.observations[grp].terms["style"] = ObservationTermCfg(func=style_obs)
    # Start episodes inside the walk cycle (after reset_base / reset_robot_joints).
    cfg.events["reset_to_reference"] = EventTermCfg(
        func=reset_to_reference, mode="reset", params={"prob": 0.8})

    # ================= REWARDS =================
    # Duckling's own reward set. Microduck's gait-shaping terms (air time window, foot
    # clearance/swing height, HOME pose while walking, angular momentum, body/head pose
    # extras) are dropped: they were tuned for another robot and fought the walk cycle.
    # 2026-10-15: own copy - the Microduck cfg shares its reward terms with the other tasks
    # (removing 'upright' here broke ball_kick / ground_pick at registration: KeyError)
    cfg.rewards = deepcopy(cfg.rewards)
    keep = {"track_linear_velocity", "track_angular_velocity", "upright", "body_ang_vel",
            "dof_pos_limits", "action_rate_l2", "self_collisions",   # 2026-10-15: foot_slip out (stance_grip does it)
            "head_pose_tracking"}
    for name in list(cfg.rewards):
        if name not in keep:
            del cfg.rewards[name]
    cfg.curriculum.pop("head_pose_bias_weight", None)

    R = cfg.rewards
    # walking: the cycle
    R["walk_imitation"] = RewardTermCfg(func=walk_imitation, weight=IMITATION_WEIGHT,
                                        params={"std": IMITATION_STD, "std_fine": 0.10})   # 2026-10-30: + fijn (250 waggelde: heup-roll 37 gr i.p.v. 12)
    R["contact_schedule"] = RewardTermCfg(func=contact_schedule, weight=4.0)   # timing as a guide
    R["swing_clearance"] = RewardTermCfg(func=swing_clearance, weight=1.5,   # lift the swing foot
                                         params={"target": SWING_LIFT})
    R["trunk_height"] = RewardTermCfg(func=trunk_height_tracking, weight=1.5,   # 2026-10-13: 1 -> 1.5, std 6 -> 4 mm:
                                      params={"std": 0.004})                       # the springy sag is character
    R["flat_feet"] = RewardTermCfg(func=flat_feet, weight=1.0)
    # standing: HOME
    R["home_pose"] = RewardTermCfg(func=home_pose_standing, weight=2.0)
    R["stand_still"] = RewardTermCfg(func=stand_still, weight=3.0)   # 2026-10-13: really stop at vx = 0
    # 2026-10-15: reward the RESULT - true distance and heading to the target - and heavily
    # (~45 % of the walking reward); the turn rate / gait only as a light guide
    R["follow_distance"] = RewardTermCfg(func=follow_distance, weight=4.0)   # stay 50 cm away
    R["follow_bearing"] = RewardTermCfg(func=follow_bearing, weight=6.0)     # ... and face it
    R["turn_toward"] = RewardTermCfg(func=turn_toward, weight=4.0)           # turn towards it, hard
    R["push_off"] = RewardTermCfg(func=push_off, weight=2.0)                 # push the floor back to catch up
    R["back_off"] = RewardTermCfg(func=back_off, weight=3.0)                 # really step back when too close
    R["spin_limit"] = RewardTermCfg(func=spin_limit, weight=-2.0)            # no violent spins (> 1 rad/s)
    # 2026-10-29 (Jairo): toe_out eruit - geen straf op de voetstand, de animatie (walk_imitation) doet het werk
    # R["toe_out"] = RewardTermCfg(func=toe_out, weight=-5.0)
    R["feet_parallel"] = RewardTermCfg(func=feet_parallel, weight=2.0, params={"std": 0.15})   # 2026-10-30
    R["com_forward"] = RewardTermCfg(func=com_forward, weight=5.0)   # 2026-10-14: CoM ahead of the feet (your walk)
    R["stance_grip"] = RewardTermCfg(func=stance_grip, weight=1.5)   # 2026-10-13: no sliding feet
    R["stand_action_rate"] = RewardTermCfg(func=stand_action_rate, weight=-1.0)   # at rest: no wiggling
    # task
    R["track_linear_velocity"].weight = 2.0      # 2026-10-15: 4 -> 2, the follow rewards lead (13: 2 -> 4)
    R["track_linear_velocity"].func = track_lin_two_scale      # 2026-10-14: fine 0.03 + coarse 0.10 m/s
    R["track_linear_velocity"].params = {"std_fine": 0.03, "std_coarse": 0.10, "command_name": "twist"}
    # (2026-10-13: 0.05 -> 0.03; standing still earned
                                                       # 44 % of it at full backwards speed
    # 2026-10-15: track_angular_velocity out - the turn rate was the P-controller on the bearing,
    # follow_bearing + turn_toward now reward that directly
    del R["track_angular_velocity"]
    R["upright"].weight = 1.0
    R["head_pose_tracking"].weight = 3.0          # volg16b: 1.5 -> 3, the head hung forward          # 2026-10-13: 1 -> 1.5, now incl. the walk's head style
    R["head_pose_tracking"].func = head_pose_with_style
    # volg16b: coarse std 0.25 -> 0.6 rad: at 30 deg off the old one paid ~0 = no way back
    R["head_pose_tracking"].params = {"command_name": "head_pose", "std": 0.6, "fine_std": 0.1}  # std 0.5 -> 0.25: a busy head was nearly free
    # survival: ending an episode must never pay off (else PPO learns to collapse)
    R["head_level"] = RewardTermCfg(func=head_level, weight=3.0, params={"command_name": "head_pose"})   # volg16b
    R["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=0.5)
    R["termination"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0)
    # regularisation
    R["body_ang_vel"].func = _bounded_ang_vel_penalty
    R["body_ang_vel"].params["max_sq"] = 4.0
    R["body_ang_vel"].weight = -0.05
    R["dof_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_).*",))   # passive jaw rests on its limit
    R["stance_width"] = RewardTermCfg(
        func=_stance_too_wide, weight=-100.0,
        params={"max_width": STANCE_MAX_WIDTH,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    )

    # 2026-10-15: lean reward set. Drive: follow (bearing, turn, distance, speed); style:
    # imitation + contact timing + head; aids: CoM forward, push-off; guards that cost
    # nothing while it walks well: termination/alive, action rate, joint limits, self
    # collisions, stance width, trunk height, HOME at rest. Out (old helpers, now covered):
    # (stance_grip stays: without it the backward steps became sliding/scuffling - 2026-10-15)
    # 2026-10-15 (volg13): swing_clearance stays too - without it the steps became shuffles;
    # the gait (imitation 4 + contact timing 4 + lift 1.5) weighs ~10 against ~24 for the outcome
    # rewards: from volg7 on the step rate crept from 5.5 to 16.5 /s (your walk: 4.8)
    for name in ("stand_still", "stand_action_rate", "flat_feet"):
        R.pop(name, None)
    # upright / body_ang_vel out too (the tracker task now checks for them) - fewer graphs
    R.pop("upright", None)
    R.pop("body_ang_vel", None)

    # --- BAM-specific events -> stock mjlab domain randomization
    cfg.events.pop("randomize_joint_friction", None)
    cfg.events.pop("randomize_motor_gains", None)
    cfg.events["randomize_joint_friction"] = EventTermCfg(
        func=dr.dof_frictionloss, mode="reset",
        params={"asset_cfg": deepcopy(ACTUATED), "operation": "scale", "ranges": (0.5, 2.0)},
    )
    cfg.events["randomize_joint_damping"] = EventTermCfg(
        func=dr.dof_damping, mode="reset",
        params={"asset_cfg": deepcopy(ACTUATED), "operation": "scale", "ranges": (0.5, 2.0)},
    )
    cfg.events["randomize_pd_gains"] = EventTermCfg(
        func=dr.pd_gains, mode="reset",
        params={"asset_cfg": deepcopy(ACTUATED), "kp_range": (0.8, 1.2), "kd_range": (0.7, 1.3)},
    )
    # Motor strength is the biggest unknown until the bench test: train for 85-115 %.
    cfg.events["randomize_effort_limits"] = EventTermCfg(
        func=dr.effort_limits, mode="reset",
        params={"asset_cfg": deepcopy(ACTUATED), "effort_limit_range": (0.85, 1.15)},
    )
    if "randomize_head_com" in cfg.events:
        cfg.events["randomize_head_com"].params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=HEAD_BODY_NAMES)

    # --- pushes, commands, foot heights: scaled to the smaller robot
    if "push_robot" in cfg.events:
        r = 0.3 * SCALE
        cfg.events["push_robot"].params["velocity_range"] = {"x": (-r, r), "y": (-r, r)}
    # 2026-10-30: servodoelen binnen het gewrichtsbereik (zoals de firmware straks ook doet).
    # Run 900 stuurde doelen tot 300 gr (!): heup-yaw permanent +200 gr = tegen de aanslag
    # gedrukt (V-voeten, gewricht "op slot" voor stabiliteit). Een echte servo kan dat niet.
    act_cfg = cfg.actions["joint_pos"]
    act_cfg.clip = {k: (math.radians(lo), math.radians(hi)) for k, (lo, hi) in JOINT_TARGET_RANGE_DEG.items()}
    # 2026-10-30: massa per lichaam -10 .. +15 % (was alleen de romp +-5 %): het echte robotje
    # weegt waarschijnlijk meer dan het model (hars, draden, schroeven)
    if "randomize_mass_inertia" in cfg.events:
        ev = cfg.events["randomize_mass_inertia"]
        ev.params["asset_cfg"] = SceneEntityCfg("robot", body_names=(".*",))
        ev.params["alpha_range"] = (math.log(0.90) / 2.0, math.log(1.15) / 2.0)
    # 2026-10-30: action_rate liep op van -0.1 naar -1.0 (iter 500..1500): bij -1.0 kost hij meer dan
    # walk_imitation kan opleveren -> de imitatie zakte vanaf ~500. Nu hooguit -0.2.
    if "action_rate_weight" in cfg.curriculum:
        st = cfg.curriculum["action_rate_weight"].params["weight_stages"]
        cfg.curriculum["action_rate_weight"].params["weight_stages"] = [
            dict(x, weight=max(x["weight"], -0.2)) for x in st]
    cmd = cfg.commands["twist"]
    # Around the speed of the reference walk cycle (~0.09 m/s at a 1 s cycle).
    # 2026-10-13: forward + turning only. With backwards and sideways commands (the walk
    # cycle only shows forward walking) the policy settled on shuffling in place.
    # 2026-10-14: 'lopen'-policy = vooruit + draaien + stoppen (+ hoofd). Achteruit en zijwaarts
    # worden later aparte policies, gewisseld via stilstand (rust-modus).
    cmd.ranges.lin_vel_x = (-0.06, 0.14)   # 2026-10-15: up to 0.14 (~112 % of the grid); back only when the person is < 30 cm
    cmd.ranges.lin_vel_y = (0.0, 0.0)
    # 2026-10-13: mjlab's default sends 20 % of the envs a forward-only command of at
    # least 0.3 m/s - three times the duckling's top speed. Off.
    cmd.rel_forward_envs = 0.0
    cmd.ranges.ang_vel_z = (-0.6, 0.6)
    cmd.viz.z_offset = 0.25
    # 2026-10-15: follow a target; the actor sees the target (camera-like), not the twist
    cmd.resampling_time_range = (3.0, 8.0)
    if TASK == "volgen":
        cfg.commands["twist"] = TargetFollowCommandCfg(**vars(cmd))
        cfg.observations["actor"].terms["command"] = ObservationTermCfg(
            func=target_obs, params={"command_name": "twist"},
            delay_min_lag=3, delay_max_lag=5, delay_update_period=50)   # 60-100 ms
    else:
        # 2026-10-29 LOPEN: vx = fractie (0..1) van de 100%-snelheid van de stijl, draaien +-0.6 rad/s.
        # Eerst dit trainen op V2; volgen (DUCK_TASK=volgen) is een fine-tune daarna.
        cmd.ranges.lin_vel_x = (0.0, 1.0)
        cmd.rel_turn_in_place_envs = 0.2   # 2026-10-30: 0.1 -> 0.2 (gedrag draait veel op de plaats)
        cfg.commands["twist"] = BehaviorVelocityCommandCfg(**vars(cmd))
        cfg.commands["head_pose"] = _BehaviorHeadPoseCfg(**vars(cfg.commands["head_pose"]))
        for name in ("follow_distance", "follow_bearing", "turn_toward", "push_off", "back_off"):
            R.pop(name, None)
        R["track_linear_velocity"].weight = 4.0
        R["track_angular_velocity"] = RewardTermCfg(
            func=track_ang_two_scale, weight=4.0,
            params={"std_fine": 0.15, "std_coarse": 0.6, "command_name": "twist"})   # 2026-10-30: 0.25 -> 0.15 (1050 draaide niet)
        R["com_forward"].weight = 2.0
        R["stand_still"] = RewardTermCfg(func=stand_still, weight=3.0)
        R["flat_feet"] = RewardTermCfg(func=flat_feet, weight=1.0)

    # --- head pose commands for Duckling's joint ranges
    # Curricula (restored 2026-10-13): a fresh run must not start with the full head range
    # and many standing envs - that run fell over with the head slammed into its limits.
    # (They do NOT restart on resume: the runner restores the step counter on load.)
    cfg.commands["head_pose"].ranges = _head_stage(0.05)
    cfg.curriculum["head_pose_range"].params["range_stages"] = [
        {"step": 0, "ranges": _head_stage(0.05)},
        {"step": 500 * 24, "ranges": _head_stage(0.15)},
        {"step": 1000 * 24, "ranges": _head_stage(0.35)},
        {"step": 1500 * 24, "ranges": _head_stage(0.65)},
        {"step": 2000 * 24, "ranges": _head_stage(1.0)},
    ]

    # --- CoM randomization scaled to the smaller support polygon
    if "com_range" in cfg.curriculum:
        cfg.curriculum["com_range"].params["range_stages"] = [
            {"step": 0, "range": 0.002}, {"step": 500 * 24, "range": 0.003},
            {"step": 1000 * 24, "range": 0.006}, {"step": 1500 * 24, "range": 0.009},
        ]
    if "head_com_range" in cfg.curriculum:
        cfg.curriculum["head_com_range"].params["range_stages"] = [
            {"step": 0, "range": 0.002}, {"step": 500 * 24, "range": 0.003},
            {"step": 1000 * 24, "range": 0.006},
        ]
    return cfg


# Smaller actor so the policy fits on the RP2350 (~57 k parameters, ~1-2 ms per step).
DucklingRlCfg = deepcopy(MicroduckRlCfg)
DucklingRlCfg.actor = RslRlModelCfg(
    hidden_dims=(256, 128, 64),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5, "std_type": "scalar"},
)
DucklingRlCfg.wandb_project = "duckling"
DucklingRlCfg.experiment_name = "duckling_velocity"
DucklingRlCfg.run_name = "duckling_velocity"
