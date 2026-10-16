"""Reward ordering check for the Duckling walk task (no training needed).

Teleports the robot through scripted behaviours (kinematically, every 20 ms) while the
velocity command asks for the reference walking speed, and prints the reward per term.
The reference walk cycle must clearly beat standing still, shuffling and splaying;
otherwise the training will find one of those instead.

    uv run python scripts/check_rewards.py            (CPU is fine)
"""
import math
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import duckling_velocity_env_cfg as D

N, SECONDS = 4, 2.0


def main():
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Duckling")
    cfg.scene.num_envs = N
    cfg.events.pop("push_robot", None)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    rob = env.scene["robot"]
    ref = D._ref(env)
    data = np.load(D.WALK_CYCLE_FILE)
    v_ref = float(data["vel_xy"][0])
    rm = env.reward_manager
    twist = env.command_manager.get_term("twist")
    adr = rob.data.indexing.free_joint_q_adr
    home = rob.data.default_joint_pos.clone()
    legs = ref.leg_ids
    names = [str(n) for n in data["joint_names"]]
    roll_cols = [D.LEG_JOINTS.index(f"{s}_hip_roll") for s in ("left", "right")]
    yaw_cols = [D.LEG_JOINTS.index(f"{s}_hip_yaw") for s in ("left", "right")]

    def behaviour(kind, t):
        ph = torch.tensor([(t / D._gait_phase.period) % 1.0])
        q = home[:1].clone()
        x = v_ref * t
        z = D.HOME_TRUNK_HEIGHT
        if kind == "stilstaan":
            x = 0.0
        elif kind in ("referentie", "spreid + uitgedraaid", "ref + eigen zwaai", "pas op de plaats"):
            q[:, legs] = ref.at(ref.q, ph)
            z = float(ref.at(ref.height, ph))
            if kind == "pas op de plaats":
                x = 0.0
            if kind == "ref + eigen zwaai":       # opposite-sign roll (mirrored axes): trunk shifts sideways
                sway = 0.14 * math.sin(2 * math.pi * t / D._gait_phase.period)
                q[0, legs[roll_cols[0]]] += sway; q[0, legs[roll_cols[1]]] -= sway
            elif kind != "referentie":
                q[0, legs[roll_cols[0]]] += 0.25; q[0, legs[roll_cols[1]]] += 0.25   # legs apart
                q[0, legs[yaw_cols[0]]] += 0.30; q[0, legs[yaw_cols[1]]] -= 0.30
        elif kind == "schuifelen":
            # half amplitude around HOME, twice the cadence, trunk 8 mm lower
            ph2 = torch.tensor([(2 * t / D._gait_phase.period) % 1.0])
            q[:, legs] = home[:1, legs] + 0.5 * (ref.at(ref.q, ph2) - ref.q.mean(0))
            z = D.HOME_TRUNK_HEIGHT - 0.008
        return q, x, z

    totals = {}
    for kind in ("referentie", "ref + eigen zwaai", "pas op de plaats", "stilstaan", "schuifelen", "spreid + uitgedraaid"):
        env.reset()
        env._duck_phase_offset[:] = 0.0
        sums = {n: 0.0 for n in rm.active_terms}
        steps = int(SECONDS / env.step_dt)
        prev_q = None
        for k in range(steps):
            t = k * env.step_dt
            env.episode_length_buf[:] = k
            q, x, z = behaviour(kind, t)
            dq = torch.zeros_like(q) if prev_q is None else (q - prev_q) / env.step_dt
            prev_q = q
            rob.write_joint_state_to_sim(q.repeat(N, 1), dq.repeat(N, 1))
            pose = torch.tensor([[x, 0.0, z - 0.0003, 1.0, 0.0, 0.0, 0.0]]).repeat(N, 1)
            vel = torch.zeros(N, 6)
            vel[:, 0] = 0.0 if kind in ("stilstaan", "pas op de plaats") else v_ref
            rob.write_root_link_pose_to_sim(pose)
            rob.write_root_link_velocity_to_sim(vel)
            twist.vel_command_b[:] = torch.tensor([v_ref, 0.0, 0.0])
            env.sim.forward()
            env.scene.update(dt=env.step_dt)
            env.sim.sense()
            rm.compute(dt=env.step_dt)
            for i, n in enumerate(rm.active_terms):
                sums[n] += float(rm._step_reward[:, i].mean()) * env.step_dt
        totals[kind] = sums

    kinds = list(totals)
    print(f"beloning per term over {SECONDS:g} s bij commando vx = {v_ref:.3f} m/s")
    print(f"{'term':24s}" + "".join(f"{k[:17]:>18s}" for k in kinds))
    for n in rm.active_terms:
        row = [totals[k][n] for k in kinds]
        if max(abs(v) for v in row) > 1e-3:
            print(f"{n:24s}" + "".join(f"{v:18.3f}" for v in row))
    tot = {k: sum(totals[k].values()) for k in kinds}
    print(f"{'TOTAAL':24s}" + "".join(f"{tot[k]:18.3f}" for k in kinds))
    best_other = max(tot[k] for k in kinds if k not in ("referentie", "ref + eigen zwaai"))
    margin = (tot["referentie"] - best_other) / abs(tot["referentie"])
    print(f"\nreferentie {'wint' if margin > 0 else 'VERLIEST'}: {margin * 100:+.0f}% t.o.v. het beste alternatief")


if __name__ == "__main__":
    main()
