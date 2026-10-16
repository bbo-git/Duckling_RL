"""Duckling universal gait tracker.

One policy that follows ANY gait from the gait generator (gait_gen.py): every episode
gets a random gait from gait_library.npz (hundreds of checked walks, plus standing), and
the policy sees where its joints should be in the next 20, 60 and 120 ms. After training,
new gaits are just new parameter sets for the generator - no retraining.

Actor observations (94):
    base ang vel (3), projected gravity (3), joint pos - HOME (14), joint vel (14),
    last action (14), reference joint pos - HOME at +1, +3, +6 control steps (42),
    reference forward speed (1), reference trunk height - 0.0556 (1), gait clock (2)
Action: 14 joint targets relative to HOME (unchanged).
"""

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab_microduck.tasks.duckling_velocity_env_cfg import (
    DucklingRlCfg,
    HOME_TRUNK_HEIGHT,
    _bounded_ang_vel_penalty,
    _stance_too_wide,
    flat_feet,
    make_duckling_velocity_env_cfg,
)

LIBRARY_FILE = Path(__file__).parents[1] / "robot" / "duckling" / "gait_library.npz"
LOOKAHEAD = (1, 3, 6)                     # control steps (20 ms each)
LEG_W = {"hip_yaw": 0.5, "hip_roll": 0.5, "hip_pitch": 1.0, "knee": 1.0, "ankle": 1.0}
HEAD_W = 0.5
TRACK_STD = 0.25                          # rad, weighted RMS joint error -> exp(-1)


class _Library:
    def __init__(self, env):
        d = np.load(LIBRARY_FILE)
        dev = env.device
        f = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        names = [str(n) for n in d["joint_names"]]
        asset = env.scene["robot"]
        self.joint_ids = torch.tensor(asset.find_joints(names, preserve_order=True)[0], device=dev)
        self.home = asset.data.default_joint_pos[0, self.joint_ids].clone()
        self.q = f(d["joint_pos"])                           # (K, N, 14)
        self.height = f(d["root_height"])                    # (K, N)
        self.rp = f(d["root_rp"])                            # (K, N, 2) roll, pitch
        self.contact = torch.tensor(d["contact"], dtype=torch.bool, device=dev)
        self.length = torch.tensor(d["length"], dtype=torch.long, device=dev)
        self.cycle = f(d["cycle_s"])
        self.vel = f(d["vel_xy"])
        self.K = self.q.shape[0]
        w = []
        for n in names:
            if n.startswith(("left_", "right_")):
                w.append(LEG_W[n.split("_", 1)[1]])
            else:
                w.append(HEAD_W)
        self.w = f(w) / sum(w)
        self.n_env = env.num_envs
        self.gait = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
        self.offset = torch.zeros(env.num_envs, device=dev)

    def phase(self, env, ahead_steps: int = 0) -> torch.Tensor:
        t = (env.episode_length_buf.float() + ahead_steps) * env.step_dt
        return torch.remainder(t / self.cycle[self.gait] + self.offset, 1.0)

    def at(self, x, phase):
        """x: (K, N, ...) table, sampled per env at its gait and phase (linear interp)."""
        n = self.length[self.gait]
        f = phase * n
        i0 = f.long() % n
        i1 = (i0 + 1) % n
        w = (f - f.floor()).view(-1, *([1] * (x.dim() - 2)))
        return (1 - w) * x[self.gait, i0] + w * x[self.gait, i1]


def lib(env) -> _Library:
    L = getattr(env, "_duck_lib", None)
    if L is None:
        L = _Library(env)
        env._duck_lib = L
    return L


# ---------------------------------------------------------------- observations
def ref_features(env) -> torch.Tensor:
    L = lib(env)
    parts = [L.at(L.q, L.phase(env, a)) - L.home for a in LOOKAHEAD]
    ph = 2 * math.pi * L.phase(env)
    parts += [L.vel[L.gait, :1],
              (L.at(L.height, L.phase(env)) - HOME_TRUNK_HEIGHT).unsqueeze(-1) * 10.0,
              torch.stack([torch.sin(ph), torch.cos(ph)], -1)]
    return torch.cat(parts, -1)


# ---------------------------------------------------------------- events
def reset_gait(env, env_ids: torch.Tensor, rsi_prob: float = 0.8, stand_prob: float = 0.1):
    """New random gait per episode; start inside it (reference state initialisation)."""
    if env_ids is None or len(env_ids) == 0:
        return
    L = lib(env)
    k = len(env_ids)
    dev = env.device
    g = torch.randint(0, L.K, (k,), device=dev)
    stand = torch.rand(k, device=dev) < stand_prob
    g[stand] = L.K - 1                                      # last entry = stand still
    L.gait[env_ids] = g
    use = torch.rand(k, device=dev) < rsi_prob
    L.offset[env_ids] = torch.where(use, torch.rand(k, device=dev), torch.zeros(k, device=dev))
    sel = env_ids[use]
    if len(sel) == 0:
        return
    asset = env.scene["robot"]
    saved = L.gait.clone(), L.offset.clone()
    # sample the reference for the selected envs only (phase uses episode_length_buf = 0)
    ph = L.offset[sel]
    gs = L.gait[sel]
    n = L.length[gs]
    f = ph * n
    i0 = f.long() % n
    i1 = (i0 + 1) % n
    w = (f - f.floor()).unsqueeze(-1)
    qr = (1 - w) * L.q[gs, i0] + w * L.q[gs, i1]
    dq = (L.q[gs, i1] - L.q[gs, i0]) * float(1.0 / env.step_dt) * 0 + \
         (L.q[gs, i1] - L.q[gs, i0]) / (L.cycle[gs] / n).unsqueeze(-1)
    q = asset.data.joint_pos[sel].clone()
    v = torch.zeros_like(q)
    q[:, L.joint_ids] = qr
    v[:, L.joint_ids] = dq
    asset.write_joint_state_to_sim(q, v, env_ids=sel)
    qpos = asset.data.data.qpos
    adr = asset.data.indexing.free_joint_q_adr
    pose = qpos[sel][:, adr].clone()
    h = (1 - w[:, 0]) * L.height[gs, i0] + w[:, 0] * L.height[gs, i1]
    pose[:, 2] = h + 0.001
    yaw = 2 * torch.atan2(pose[:, 6], pose[:, 3])
    vel = torch.zeros(len(sel), 6, device=dev)
    vel[:, 0] = L.vel[gs, 0] * torch.cos(yaw)
    vel[:, 1] = L.vel[gs, 0] * torch.sin(yaw)
    asset.write_root_link_pose_to_sim(pose, env_ids=sel)
    asset.write_root_link_velocity_to_sim(vel, env_ids=sel)


# ---------------------------------------------------------------- rewards
def joint_tracking(env, std: float = TRACK_STD) -> torch.Tensor:
    L = lib(env)
    q = env.scene["robot"].data.joint_pos[:, L.joint_ids]
    err = torch.sum(L.w * torch.square(q - L.at(L.q, L.phase(env))), -1)
    return torch.exp(-err / std**2)


def contact_tracking(env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    L = lib(env)
    n = L.length[L.gait]
    i = (L.phase(env) * n).long() % n
    ref = L.contact[L.gait, i]
    found = env.scene.sensors[sensor_name].data.found.reshape(env.num_envs, -1)[:, :2] > 0
    return (found == ref).float().mean(-1)


def height_tracking(env, std: float = 0.005) -> torch.Tensor:
    L = lib(env)
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    return torch.exp(-torch.square(z - L.at(L.height, L.phase(env))) / std**2)


def speed_tracking(env, std: float = 0.04) -> torch.Tensor:
    """Forward speed of the reference, no sideways drift (trunk heading frame)."""
    L = lib(env)
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    target = torch.stack([L.vel[L.gait, 0], torch.zeros_like(L.vel[L.gait, 0])], -1)
    return torch.exp(-torch.sum(torch.square(v - target), -1) / std**2)


def heading_hold(env, std: float = 0.2) -> torch.Tensor:
    """No turning (the gaits walk straight): yaw rate -> 0. Stops walking in circles."""
    wz = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    return torch.exp(-torch.square(wz) / std**2)


def make_duckling_tracker_env_cfg(play: bool = False, rough: bool = False):
    cfg = make_duckling_velocity_env_cfg(play=play, rough=rough)

    # observations: drop the twist/head/body commands and the single-cycle clock,
    # add the reference preview
    for grp in ("actor", "critic"):
        terms = cfg.observations[grp].terms
        for name in ("command", "head_command", "body_command", "gait_clock"):
            terms.pop(name, None)
        terms["reference"] = ObservationTermCfg(func=ref_features)

    # events: gait per episode instead of the walk-cycle start
    cfg.events.pop("reset_to_reference", None)
    cfg.events["reset_gait"] = EventTermCfg(func=reset_gait, mode="reset",
                                            params={"rsi_prob": 0.8, "stand_prob": 0.1})

    # rewards
    keep = {"upright", "body_ang_vel", "dof_pos_limits", "action_rate_l2", "foot_slip",
            "self_collisions", "alive", "termination", "flat_feet"}
    for name in list(cfg.rewards):
        if name not in keep:
            del cfg.rewards[name]
    R = cfg.rewards
    R["joint_tracking"] = RewardTermCfg(func=joint_tracking, weight=5.0)
    R["contact_tracking"] = RewardTermCfg(func=contact_tracking, weight=1.5)
    R["height_tracking"] = RewardTermCfg(func=height_tracking, weight=1.0)
    R["speed_tracking"] = RewardTermCfg(func=speed_tracking, weight=2.0)
    R["heading_hold"] = RewardTermCfg(func=heading_hold, weight=1.5)
    R["upright"].weight = 1.0
    # the waddle gaits stand up to ~75 mm wide
    R["stance_width"] = RewardTermCfg(
        func=_stance_too_wide, weight=-100.0,
        params={"max_width": 0.085,
                "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot"))},
    )
    # commands are not used any more; curricula that drive them are switched off
    for c in ("standing_envs", "head_pose_range", "body_pose_range", "head_pose_bias_weight"):
        cfg.curriculum.pop(c, None)
    return cfg


DucklingTrackerRlCfg = deepcopy(DucklingRlCfg)
DucklingTrackerRlCfg.experiment_name = "duckling_tracker"
DucklingTrackerRlCfg.run_name = "duckling_tracker"
DucklingTrackerRlCfg.wandb_project = "duckling"
