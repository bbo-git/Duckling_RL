import sys, os, math, numpy as np
D = os.path.abspath("tr/microduck_rl/src/mjlab_microduck/robot/duckling"); T = os.path.abspath("tr/microduck_rl/tools")
sys.path.insert(0, T); os.chdir(D)
import duckling_play_onnx as P
robot = P.Duckling("scene_walk.xml"); pol = P.Policy(sys.argv[1])
robot.gait_clock = True; robot.styled = pol.n_in == 63 + len(P.STYLES)
robot.grids = [P.GridClock(p) for _, p in P.STYLES]
names = robot.names; hy = names.index("head_yaw")
def run(style, vx, wz, headyaw, secs=6):
    robot.set_style(style); robot.reset(); tw = np.array([vx, 0, wz]); yaws = []
    for k in range(int(secs / 0.02)):
        h = np.array([0, 0, headyaw(k * 0.02), 0])
        a = pol(robot.obs(tw, h)); a[hy] = h[2] if False else a[hy]
        robot.step(a, tw)
        yaws.append(2 * math.atan2(robot.d.qpos[6], robot.d.qpos[3]))
    y = np.unwrap(yaws); return math.degrees((y[-1] - y[50]) / (secs - 1.0)), robot.fallen
for style in (0, 1):
    for vx in (0.0, 0.04):
        for wz in (0.5, -0.5):
            out = []
            for lab, hf in (("kop 0", lambda t: 0.0), ("kop +40", lambda t: 0.7), ("kop -40", lambda t: -0.7)):
                r, f = run(style, vx, wz, hf); out.append(f"{lab}: {r:+5.0f} gr/s{' VIEL' if f else ''}")
            print(f"stijl {style} vx {vx:.2f} wz {math.degrees(wz):+.0f} gr/s ->  " + "   ".join(out))
