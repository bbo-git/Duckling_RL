# Duckling robot files

Put here (not in git):
- `duck_joints.csv`, `duck_links.csv` – export from Fusion (export_duck_joints.py)
- `meshes/*.stl` – one STL per printed part, named after the Fusion body, global frame, zero pose

Then generate the walk model:

    python scripts/gen_duck_mjcf.py \
        src/mjlab_microduck/robot/duckling/duck_joints.csv \
        src/mjlab_microduck/robot/duckling/duck_links.csv \
        --walk src/mjlab_microduck/robot/duckling/robot_walk.xml

View it with `python -m mujoco.viewer --mjcf src/mjlab_microduck/robot/duckling/scene_walk.xml`.
