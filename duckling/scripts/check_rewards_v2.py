"""Reward-volgorde per STIJL (V2, 2026-10-29), zonder training.

Teleporteert de robot (elke 20 ms) door gedragingen terwijl het commando 80% van de
stijl vraagt, en print de beloning per term. Het loopje van de stijl moet duidelijk winnen
van stilstaan, trappelen op de plaats, schuifelen en het loopje van de ANDERE stijl
(anders leert de policy de stijlen niet uit elkaar te houden).

    uv run python scripts/check_rewards_v2.py        (CPU is genoeg)
"""
import math
import numpy as np
import torch

import mjlab_microduck.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import duckling_velocity_env_cfg as D

SECONDS, FRAC = 2.0, 0.8


def main():
    nS = len(D.STYLES)
    cfg = load_env_cfg("Mjlab-Velocity-Flat-Duckling")
    cfg.scene.num_envs = nS
    cfg.events.pop("push_robot", None)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    rob = env.scene["robot"]
    rm = env.reward_manager
    twist = env.command_manager.get_term("twist")
    env.reset()
    ref = D._ref(env)
    home = rob.data.default_joint_pos[:1].clone()
    legs = ref.leg_ids
    nS_, nW, nY = ref.dims
    a = int(np.argmin(np.abs(ref.S.cpu().numpy() - FRAC)))
    b = int(np.argmin(np.abs(ref.W.cpu().numpy())))
    c = int(np.argmin(np.abs(ref.Y.cpu().numpy())))
    cell = lambda st: st * ref.G + (a * nW + b) * nY + c

    def sample(table, g, ph):
        n = int(ref.length[g]); f = ph * n; i0 = int(f) % n; w = f - math.floor(f)
        return (1 - w) * table[g, i0] + w * table[g, (i0 + 1) % n]

    kinds = ["eigen loopje", "loopje andere stijl", "trappelen op de plaats", "stilstaan", "schuifelen"]
    res = {}
    for st in range(nS):
        v = float(ref.v100_s[st]) * FRAC
        for kind in kinds:
            g = cell(st if kind != "loopje andere stijl" else (st + 1) % nS)
            cyc = float(ref.cycle[g])
            env.reset()
            twist.style[:] = st
            sums = {n: 0.0 for n in rm.active_terms}
            prev_q = None
            for k in range(int(SECONDS / env.step_dt)):
                t = k * env.step_dt
                ph = (t / cyc) % 1.0
                q = home.clone(); x = v * t; z = D.HOME_TRUNK_HEIGHT; vx = v
                if kind in ("eigen loopje", "loopje andere stijl", "trappelen op de plaats"):
                    q[0, legs] = sample(ref.q, g, ph); z = float(sample(ref.h, g, ph))
                    if kind == "trappelen op de plaats":
                        x, vx = 0.0, 0.0
                elif kind == "stilstaan":
                    x, vx = 0.0, 0.0
                elif kind == "schuifelen":
                    ph2 = (2 * t / cyc) % 1.0
                    q[0, legs] = home[0, legs] + 0.4 * (sample(ref.q, g, ph2) - ref.q[g, :int(ref.length[g])].mean(0))
                    z = D.HOME_TRUNK_HEIGHT - 0.006; x, vx = 0.3 * x, 0.3 * v
                dq = torch.zeros_like(q) if prev_q is None else (q - prev_q) / env.step_dt
                prev_q = q
                n = env.num_envs
                rob.write_joint_state_to_sim(q.repeat(n, 1), dq.repeat(n, 1))
                pose = torch.tensor([[x, 0.0, z - 0.0003, 1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
                vel = torch.zeros(n, 6); vel[:, 0] = vx
                rob.write_root_link_pose_to_sim(pose)
                rob.write_root_link_velocity_to_sim(vel)
                twist.vel_command_b[:] = torch.tensor([v, 0.0, 0.0])
                twist.style[:] = st
                env.sim.forward(); env.scene.update(dt=env.step_dt); env.sim.sense()
                ref.last_step = -5; ref._update(env); ref.phase[:] = (t / float(ref.cycle[cell(st)])) % 1.0
                rm.compute(dt=env.step_dt)
                for i, nm in enumerate(rm.active_terms):
                    sums[nm] += float(rm._step_reward[0, i]) * env.step_dt
            res[(st, kind)] = sums
        print(f"\n=== stijl {D.STYLES[st][0]}: commando {v * 100:.1f} cm/s ({FRAC * 100:.0f}%), {SECONDS:g} s")
        print(f"{'term':22s}" + "".join(f"{k[:16]:>17s}" for k in kinds))
        for nm in rm.active_terms:
            row = [res[(st, k)][nm] for k in kinds]
            if max(abs(x) for x in row) > 1e-3:
                print(f"{nm:22s}" + "".join(f"{x:17.3f}" for x in row))
        tot = {k: sum(res[(st, k)].values()) for k in kinds}
        print(f"{'TOTAAL':22s}" + "".join(f"{tot[k]:17.3f}" for k in kinds))
        best = max(tot[k] for k in kinds[1:])
        print(f"eigen loopje {'WINT' if tot[kinds[0]] > best else 'VERLIEST'}: "
              f"{(tot[kinds[0]] - best) / abs(tot[kinds[0]]) * 100:+.0f}% t.o.v. het beste alternatief "
              f"({max(kinds[1:], key=lambda k: tot[k])})")


if __name__ == "__main__":
    main()
