"""Left-right mirror for the Duckling walk task (63-dim actor obs), for the rsl_rl
mirror loss. 2026-10-13: turning worked to one side only.

NOT the microduck table: the Duckling's joint axes differ. Checked with forward
kinematics (feet and head mirror exactly): after swapping left and right legs, ALL
leg joints keep their sign (the right-side axes are already mirrored in the model);
of the head joints only head_roll and head_yaw flip.

Actuator / joint block order (14):
    0-4 left  hip_yaw, hip_roll, hip_pitch, knee, ankle
    5 neck_pitch, 6 head_pitch, 7 head_roll, 8 head_yaw
    9-13 right hip_yaw, hip_roll, hip_pitch, knee, ankle
Actor obs (63): ang_vel 0:3, gravity 3:6, joint_pos 6:20, joint_vel 20:34,
last_action 34:48, target 48:51 (bearing, distance, visible - 2026-10-15, was twist), head command 51:55 (neck_pitch, head_pitch, head_yaw,
head_roll), body command 55:61, gait clock 61:63 (sin, cos). Mirrored the clock is half
a cycle on (the other foot leads): both components flip.
"""
import torch
from tensordict import TensorDict

_JOINT_PERM = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
_JOINT_SIGN = [1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1]

# 2026-10-29 (V2): stijl-one-hot 63:63+n (spiegelt niet), en het commando hangt af van de taak:
# lopen = twist (vx, vy, wz -> vx, -vy, -wz), volgen = doel (bearing, afstand, zichtbaar -> -b, a, z)
from mjlab_microduck.tasks.duckling_velocity_env_cfg import STYLES, TASK
_N_STYLE = len(STYLES)
_CMD_SIGN = [-1.0, 1.0, 1.0] if TASK == "volgen" else [1.0, -1.0, -1.0]
_OBS_PERM = ([0, 1, 2] + [3, 4, 5]
             + [6 + j for j in _JOINT_PERM] + [20 + j for j in _JOINT_PERM] + [34 + j for j in _JOINT_PERM]
             + [48, 49, 50] + [51, 52, 53, 54] + [55, 56, 57, 58, 59, 60] + [61, 62]
             + [63 + i for i in range(_N_STYLE)])
_OBS_SIGN = ([-1.0, 1.0, -1.0] + [1.0, -1.0, 1.0]
             + _JOINT_SIGN + _JOINT_SIGN + _JOINT_SIGN
             + _CMD_SIGN + [1.0, 1.0, -1.0, -1.0] + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
             + [-1.0, -1.0] + [1.0] * _N_STYLE)
assert len(_OBS_PERM) == len(_OBS_SIGN) == 63 + _N_STYLE

_cache = {}


def _tensors(dev):
    if dev not in _cache:
        _cache[dev] = (torch.tensor(_OBS_PERM, dtype=torch.long, device=dev),
                       torch.tensor(_OBS_SIGN, dtype=torch.float32, device=dev),
                       torch.tensor(_JOINT_PERM, dtype=torch.long, device=dev),
                       torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=dev))
    return _cache[dev]


def duckling_vel_symmetry(env, obs, actions):
    aug_obs = aug_act = None
    if obs is not None:
        a = obs["actor"]
        op, os_, _, _ = _tensors(a.device)
        c = obs["critic"]
        aug_obs = TensorDict({"actor": torch.cat([a, a[:, op] * os_], 0),
                              "critic": torch.cat([c, c], 0)},
                             batch_size=[a.shape[0] * 2], device=a.device)
    if actions is not None:
        _, _, ap, as_ = _tensors(actions.device)
        aug_act = torch.cat([actions, actions[:, ap] * as_], 0)
    return aug_obs, aug_act
