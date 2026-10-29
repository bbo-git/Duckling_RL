"""Duckling: Jairo's ~15 cm SLA-printed duck (differential hips, coreless gearmotors + micro servos).

The walk model (robot/duckling/robot_walk.xml) is generated from the Fusion 360 export with
    python gen_duck_mjcf.py duck_joints.csv duck_links.csv --walk robot/duckling/robot_walk.xml
and follows the Microduck naming conventions (trunk_base, imu, left_foot/right_foot sites,
left_foot_collision/right_foot_collision geoms, knee/ankle joint names, passive_* for the jaw),
so the Microduck velocity rewards and observations work unchanged.

Joint order (14 actuated): left leg (hip_yaw, hip_roll, hip_pitch, knee, ankle),
neck_pitch, head_pitch, head_roll, head_yaw, right leg (same as left).
"knee" = Fusion's ankle_pitch, "ankle" = Fusion's toe_pitch.

The hip and head differentials are trained at JOINT level (pitch/roll, roll/yaw targets);
the firmware mixes them into motor A/B commands: A = out + carrier, B = out - carrier.

Actuators are plain MuJoCo position actuators (no BAM model exists for these motors yet),
with command delay and gain/friction/damping randomization in the task config.
TODO: replace by identified parameters from the test bench.
"""

import math
import os
from pathlib import Path

import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_ROBOT_DIR: Path = Path(os.path.dirname(__file__)) / "duckling"
DUCKLING_WALK_XML: Path = _ROBOT_DIR / "robot_walk.xml"


def get_walk_spec() -> mujoco.MjSpec:
    assert DUCKLING_WALK_XML.exists(), (
        f"{DUCKLING_WALK_XML} not found - generate it with gen_duck_mjcf.py --walk"
    )
    return mujoco.MjSpec.from_file(str(DUCKLING_WALK_XML))


# Standing pose. In the CAD zero pose the CoM sits ~10 mm in front of the soles, so the
# policy had to fight its own HOME. This stance puts the CoM over the sole centres with
# flat soles and slightly bent knees (ankle = hip_pitch + knee keeps the sole level).
# Same values left and right: the pitch-chain axes are not mirrored in the CAD export.
HOME_POS_DEG = {"hip_pitch": -11.8, "knee": -8.6, "ankle": -20.3}
# 2026-10-28 (V2, alles servo's): de CAD-nulstand IS de balansstand (zwaartepunt boven de zolen)
HOME_POS_DEG = {"hip_pitch": 0.0, "knee": 0.0, "ankle": 0.0}
HOME_FRAME = EntityCfg.InitialStateCfg(
    joint_pos={
        r".*_hip_pitch$": math.radians(HOME_POS_DEG["hip_pitch"]),
        r".*_knee$": math.radians(HOME_POS_DEG["knee"]),
        r".*_ankle$": math.radians(HOME_POS_DEG["ankle"]),
        r".*": 0.0,
    },
    joint_vel={".*": 0.0},
)

# Spawn height of trunk_base (m); it settles at ~57 mm in the HOME stance.
SPAWN_Z = (0.0575, 0.0600)
SPAWN_Z = (0.0595, 0.0615)   # V2: trunk_base staat op ~59.2 mm in HOME

FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"],
    # Bits: floor 1, left leg 2, right leg 4. Soles touch the floor and the other leg,
    # shin/ankle/toe hulls only touch the other leg (legs can no longer pass through
    # each other; such contacts are penalised by the self_collisions reward).
    contype={r"^left_foot_collision$": 3, r"^right_foot_collision$": 5,
             r"^left_.*_collision$": 2, r"^right_.*_collision$": 4},
    conaffinity={r"^left_foot_collision$": 5, r"^right_foot_collision$": 3,
                 r"^left_.*_collision$": 4, r"^right_.*_collision$": 2},
    condim={r"^(left|right)_foot_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot_collision$": 1},
    friction={r"^(left|right)_foot_collision$": (0.9,)},
)

# Command latency: 0-2 physics steps (timestep set in the task cfg).
actuators = XmlActuatorCfg(
    target_names_expr=(r"^(?!passive_).*",),
    delay_min_lag=0,
    delay_max_lag=2,
)

DUCKLING_WALK_ROBOT_CFG = EntityCfg(
    spec_fn=get_walk_spec,
    init_state=HOME_FRAME,
    collisions=(FULL_COLLISION,),
    articulation=EntityArticulationInfoCfg(
        actuators=(actuators,),
        soft_joint_pos_limit_factor=0.9,
    ),
)
