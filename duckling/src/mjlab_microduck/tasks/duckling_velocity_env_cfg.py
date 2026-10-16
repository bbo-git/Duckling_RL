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

import math

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
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

SCALE = 0.6  # Duckling / Microduck length ratio

# Physics: smaller robot -> faster dynamics -> smaller timestep. 2 ms x 10 = 50 Hz policy.
TIMESTEP = 0.002
DECIMATION = 10

HOME_TRUNK_HEIGHT = 0.0556  # m, trunk_base standing in HOME
STANCE_MAX_WIDTH = 0.065   # m. 0.055 punished your walk (v3 stands 55-58 mm); 0.075 let the wide-legged jitter back
MIN_TRUNK_HEIGHT = 0.042  # m; trunk_base stands at ~0.057 in the HOME stance

HEAD_BODY_NAMES = ("neck", "head_pitch", "head_roll", "head")
ACTUATED = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))

# Final head-pose command caps (rad, deltas from HOME) in the order
# neck_pitch, head_pitch, head_yaw, head_roll — ~90 % of the joint ranges in the CAD export.
# 2026-10-13: halved (was 0.47, 0.47, 0.93, 0.31 = ~90 % of the joint range). With full-range
# look commands from iteration 2000 on the heavy head swung +-50 deg and the walking got worse.
HEAD_CAPS = (0.25, 0.25, 0.50, 0.15)


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


def _stance_too_wide(env, max_width: float, asset_cfg: SceneEntityCfg):
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
    return torch.clamp(lateral.abs() - max_width, min=0.0)


# --- Reference walk cycle (Blender) -------------------------------------------
# walk_cycle.npz: one gait cycle, CAD-zero joint angles at 50 Hz (convert_anim.py).
# The policy gets a gait clock (sin, cos of the phase) as 2 extra observations and
# is rewarded for following the reference leg angles while it is asked to move.
WALK_CYCLE_FILE = Path(__file__).parents[1] / "robot" / "duckling" / "walk_cycle.npz"
IMITATION_WEIGHT = 4.0
# Style reference: your walk (gait_grid.npz from walk_speed.py: speed x turn x side) instead
# of the single Blender cycle. The gait is picked per env from the twist command
# (vx -> speed, wz -> turn) and blended between grid neighbours, with its own clock.
USE_GAIT_GRID = True
SIDE_CYCLE_S = 0.45          # s, typical cycle: sideways command (m/s) -> grid side axis (m per cycle)
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]   # = head_pose command order
GAIT_GRID_FILE = Path(__file__).parents[1] / "robot" / "duckling" / "gait_grid.npz"
SWING_LIFT = 0.010            # m, swing-foot lift that earns the full swing_clearance reward
IMITATION_STD = 0.25          # rad, RMS joint error that gives exp(-1)
LEG_JOINTS = [f"{s}_{j}" for s in ("left", "right")
              for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]
# Per-joint weight in the imitation error. The walk cycle is a style guide for the
# fore-aft leg motion (hip pitch, knee, ankle). Hip roll/yaw are left mostly free: the
# animation has no sideways weight shift, so the policy must add its own sway to stay
# balanced (the stance_width penalty still stops it from splaying).
# The ankle is left to the policy too: in the animation the stance foot is tilted
# 10-30 deg (it would stand on an edge); the flat_feet reward keeps the sole flat instead.
IMITATION_JOINT_WEIGHT = {"hip_yaw": 0.3, "hip_roll": 0.3, "hip_pitch": 1.0, "knee": 1.0, "ankle": 0.1}


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

    def __call__(self, env, std: float, command_name: str = "twist") -> torch.Tensor:
        ref = _ref(env)
        q_ref = ref.q_now(env)
        q = env.scene["robot"].data.joint_pos[:, ref.leg_ids]
        r = torch.exp(-torch.sum(self.w * torch.square(q - q_ref), dim=-1) / std**2)
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
        return self.at(self.q, phase), self.at(self.dq, phase), self.at(self.height, phase), self.vel_xy[0]


class _GridReference:
    """gait_grid.npz on the env device. Per env: the gait for the current twist command
    (speed = vx / v100, turn = wz in deg/s, side 0), blended over the 8 grid neighbours
    and towards standing for tiny commands, with a phase that runs at that gait's cadence."""

    def __init__(self, env):
        d = np.load(GAIT_GRID_FILE)
        dev = env.device
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        names = [str(n) for n in d["joint_names"]]
        cols = [names.index(j) for j in LEG_JOINTS]
        asset = env.scene["robot"]
        self.leg_ids = torch.tensor(asset.find_joints(LEG_JOINTS, preserve_order=True)[0], device=dev)
        self.S, self.W, self.Y = f(d["speeds"]), f(d["turns"]), f(d["sides"])
        nS, nW, nY = len(self.S), len(self.W), len(self.Y)
        self.dims = (nS, nW, nY)
        G = nS * nW * nY
        q = np.asarray(d["joint_pos"]).reshape(G, -1, len(names))[..., cols]
        self.q = f(q)                                                       # (G, T, 10)
        self.h = f(np.asarray(d["root_height"]).reshape(G, -1))            # (G, T)
        self.c = f(np.asarray(d["contact"]).reshape(G, -1, 2))             # (G, T, 2)
        self.length = torch.tensor(np.asarray(d["length"]).reshape(-1), dtype=torch.long, device=dev)
        self.cycle = f(np.asarray(d["cycle_s"]).reshape(-1))
        self.vel = f(np.asarray(d["vel"]).reshape(G, 2))
        self.stand_q = f(np.asarray(d["stand_joint_pos"])[cols])
        hcols = [names.index(j) for j in HEAD_JOINTS]
        self.head = f(np.asarray(d["joint_pos"]).reshape(G, -1, len(names))[..., hcols])   # (G, T, 4)
        self.stand_head = f(np.asarray(d["stand_joint_pos"])[hcols])
        self.stand_h = float(d["stand_root_height"])
        self.v100 = float(d["v100"])
        self.i100 = (int(np.argmin(np.abs(np.asarray(d["speeds"]) - 1.0))) * nW
                     + int(np.argmin(np.abs(np.asarray(d["turns"])))) ) * nY + int(np.argmin(np.abs(np.asarray(d["sides"]))))
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
        s = cmd[:, 0] / self.v100                     # negative = backwards
        turn = torch.rad2deg(cmd[:, 2])
        side = cmd[:, 1] * SIDE_CYCLE_S               # m/s -> m per cycle (grid axis)
        (i, wi), (j, wj), (k, wk) = self._axis(self.S, s), self._axis(self.W, turn), self._axis(self.Y, side)
        nS, nW, nY = self.dims
        idx, wts = [], []
        for a, wa in ((i, 1 - wi), (i + 1, wi)):
            for b, wb in ((j, 1 - wj), (j + 1, wj)):
                for c, wc in ((k, 1 - wk), (k + 1, wk)):
                    idx.append((a * nW + b) * nY + c); wts.append(wa * wb * wc)
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
        """Reset pose from your walk at 100%, straight (the command may not be set yet)."""
        k = len(env_ids)
        idx = torch.full((k, 1), self.i100, dtype=torch.long, device=phase.device)
        wts = torch.ones(k, 1, device=phase.device)
        dph = self.dt / float(self.cycle[self.i100])
        q = self._sample(self.q, phase, idx, wts)
        q1 = self._sample(self.q, torch.remainder(phase + dph, 1.0), idx, wts)
        h = self._sample(self.h, phase, idx, wts)
        return q, (q1 - q) / self.dt, h, float(self.vel[self.i100, 0])


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
    vel[:, 0] = v * torch.cos(yaw)
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
    # Start episodes inside the walk cycle (after reset_base / reset_robot_joints).
    cfg.events["reset_to_reference"] = EventTermCfg(
        func=reset_to_reference, mode="reset", params={"prob": 0.8})

    # ================= REWARDS =================
    # Duckling's own reward set. Microduck's gait-shaping terms (air time window, foot
    # clearance/swing height, HOME pose while walking, angular momentum, body/head pose
    # extras) are dropped: they were tuned for another robot and fought the walk cycle.
    keep = {"track_linear_velocity", "track_angular_velocity", "upright", "body_ang_vel",
            "dof_pos_limits", "action_rate_l2", "foot_slip", "self_collisions",
            "head_pose_tracking"}
    for name in list(cfg.rewards):
        if name not in keep:
            del cfg.rewards[name]
    cfg.curriculum.pop("head_pose_bias_weight", None)

    R = cfg.rewards
    # walking: the cycle
    R["walk_imitation"] = RewardTermCfg(func=walk_imitation, weight=IMITATION_WEIGHT,
                                        params={"std": IMITATION_STD})
    R["contact_schedule"] = RewardTermCfg(func=contact_schedule, weight=1.0)   # timing as a guide
    R["swing_clearance"] = RewardTermCfg(func=swing_clearance, weight=2.0,   # lift the swing foot
                                         params={"target": SWING_LIFT})
    R["trunk_height"] = RewardTermCfg(func=trunk_height_tracking, weight=1.5,   # 2026-10-13: 1 -> 1.5, std 6 -> 4 mm:
                                      params={"std": 0.004})                       # the springy sag is character
    R["flat_feet"] = RewardTermCfg(func=flat_feet, weight=1.0)
    # standing: HOME
    R["home_pose"] = RewardTermCfg(func=home_pose_standing, weight=2.0)
    R["stand_still"] = RewardTermCfg(func=stand_still, weight=3.0)   # 2026-10-13: really stop at vx = 0
    R["stance_grip"] = RewardTermCfg(func=stance_grip, weight=1.5)   # 2026-10-13: no sliding feet
    R["stand_action_rate"] = RewardTermCfg(func=stand_action_rate, weight=-1.0)   # at rest: no wiggling
    # task
    R["track_linear_velocity"].weight = 4.0      # 2026-10-13: 2 -> 4, forward progress first
    R["track_linear_velocity"].params["std"] = 0.03   # 2026-10-13: 0.05 -> 0.03; standing still earned
                                                       # 44 % of it at full backwards speed
    R["track_angular_velocity"].weight = 2.0     # 2026-10-13: 1 -> 2, no drifting when asked straight
    R["track_angular_velocity"].params["std"] = 0.25  # rad/s (was 0.71: not turning cost almost nothing)
    R["upright"].weight = 1.0
    R["head_pose_tracking"].weight = 1.5          # 2026-10-13: 1 -> 1.5, now incl. the walk's head style
    R["head_pose_tracking"].func = head_pose_with_style
    R["head_pose_tracking"].params = {"command_name": "head_pose", "std": 0.25, "fine_std": 0.1}  # std 0.5 -> 0.25: a busy head was nearly free
    # survival: ending an episode must never pay off (else PPO learns to collapse)
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
    cmd = cfg.commands["twist"]
    # Around the speed of the reference walk cycle (~0.09 m/s at a 1 s cycle).
    # 2026-10-13: forward + turning only. With backwards and sideways commands (the walk
    # cycle only shows forward walking) the policy settled on shuffling in place.
    cmd.ranges.lin_vel_x = (-0.045, 0.12)    # 2026-10-13: backwards too (grid has -50 %)
    cmd.ranges.lin_vel_y = (-0.02, 0.02)     # 2026-10-13: sidestep back (grid: +-10 mm per cycle)
    # 2026-10-13: mjlab's default sends 20 % of the envs a forward-only command of at
    # least 0.3 m/s - three times the duckling's top speed. Off.
    cmd.rel_forward_envs = 0.0
    cmd.ranges.ang_vel_z = (-0.6, 0.6)
    cmd.viz.z_offset = 0.25

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
