# Duckling

https://github.com/user-attachments/assets/a0d4112b-b866-473c-8725-49b2428aca3c

A ~15 cm, ~190 g bipedal duck robot, and the reinforcement-learning setup that teaches it to walk
and to follow a person at 50 cm. It is a scaled-down, redesigned take on Pollen Robotics' Microduck.

This repository is an adaptation of **[microduck_rl](https://github.com/pollen-robotics/microduck_rl)
by Pollen Robotics**, trained with [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) and PPO.

## What changed compared to microduck_rl

- **3D models modified:** new, smaller robot (SLA-printed, ~0.65 scale), with a 3-DOF hip
  (yaw plus a pitch/roll differential driven by 6 mm coreless gear motors), servo-driven ankle and toe
  (Emax ES9051), and a 3-DOF head with a neck.
- **New robot description:** `src/mjlab_microduck/robot/duckling/` (MJCF, meshes, gait grids).
- **New tasks:** `duckling_velocity_env_cfg.py` (walking with imitation of a hand-made walk cycle)
  and target following with camera-like perception.
- **Tools:** gait generator (`tools/gait_gen.py`, `tools/walk_speed.py`) and players
  (`tools/duckling_play_onnx.py`, `tools/duckling_follow.py`).

## Quickstart

```bash
chmod +x duck.sh && ./duck.sh setup
./duck.sh update /duck_volg16b.zip
ENVS=8192 ./duck.sh train 2000
```
### New RunPod pod (global volume at /workspace)
```bash
cd /root && git clone https://github.com/bbo-git/Duckling_RL
tar xf /workspace/duck_env.tar -C /root
ln -s /workspace/duck_logs Duckling_RL/duckling/logs
export PATH="$HOME/.local/bin:$PATH"
cd /root/Duckling_RL && export REPO=$PWD/duckling
./duckling/duck.sh train 600                                            # default grid: gait_grid_stap_loop_vloei_v3.npz (0.5 s per step)
# DUCK_GRID=<file in grids/> ./duckling/duck.sh train 600               # another example walk
# ./duckling/duck.sh resume 300 200                                     # continue from model_200
./duckling/duck.sh export 100 --push                                    # checkpoint -> onnx/ in the repo
```
Before a new run: move old runs out of the way so `resume` picks the right one:
`mkdir -p /workspace/duck_logs/oud && mv /workspace/duck_logs/rsl_rl/duckling_velocity/* /workspace/duck_logs/oud/`

### play iterations
```bash
python duckling_follow.py duckling_walk_<iteration>.onnx --grid gait_grid_v7b.npz
```
### create a custom walk cycle for training
```bash
gait_gen.py --view
```

## Notes

Before updating to 16b, training for 1500 its on duck_basis might me a good idea.

## License and attribution

- **Code:** Apache License 2.0, see [LICENSE](LICENSE). Files from microduck_rl have been modified.
- **3D models:** the original Microduck 3D models are © **Pollen Robotics**
  ([microduck_rl](https://github.com/pollen-robotics/microduck_rl)) and licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
  **I have modified these models.** My adapted models (the Duckling meshes) are therefore also licensed under
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
  ([legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)):
  attribution required, **no commercial use**, and adaptations must be shared under the same license.
  See [LICENSE-MODELS.md](LICENSE-MODELS.md) for which files this covers.

