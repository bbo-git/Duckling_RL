import sys, math, numpy as np, duckling_play_onnx as P
pol = P.Policy(sys.argv[1]); G = P.GridClock(sys.argv[2])
H = [5, 6, 7, 8]          # neck_pitch, head_pitch, head_roll, head_yaw (actuator order)
for name, t in (("stil", (0, 0, 0)), ("vooruit 4", (0.04, 0, 0)), ("vooruit 8", (0.08, 0, 0)), ("vooruit 12", (0.12, 0, 0)),
                ("achteruit 2", (-0.02, 0, 0)), ("achteruit 4", (-0.04, 0, 0)), ("draai L", (0.0, 0, 0.5)), ("draai R", (0.0, 0, -0.5)),
                ("zij L", (0, 0.015, 0)), ("zij R", (0, -0.015, 0))):
    t = np.array(t, float)
    r = P.Duckling("scene_walk.xml", grid=G); r.gait_clock = True
    J, Hq, F, la, fell = [], [], [], np.zeros(14), None; WD=[]
    for k in range(500):
        a = pol(r.obs(t, np.zeros(4))); r.step(a, t)
        if k == 100:
            q = r.d.qpos[3:7]; yaw0 = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2)); p0 = r.d.qpos[:2].copy()
        if k > 100:
            J.append(np.degrees(np.abs(a - la).mean())); Hq.append(r.d.qpos[r.qadr[H]].copy())
            F.append(np.degrees(np.abs(a[[4, 13]] - la[[4, 13]]).mean()))
            q_=r.d.qpos[3:7]; yw=math.atan2(2*(q_[0]*q_[3]+q_[1]*q_[2]),1-2*(q_[2]**2+q_[3]**2)); dl=r.d.site('left_foot').xpos-r.d.site('right_foot').xpos; WD.append(abs(-math.sin(yw)*dl[0]+math.cos(yw)*dl[1])*1000)
        la = a
        if r.fallen:
            fell = k * 0.02; break
    if fell is not None and fell < 2.1:
        print(f"{name:12s} VALT na {fell:.1f} s"); continue
    T = (len(J)) * 0.02; d = r.d.qpos[:2] - p0; c, s = math.cos(yaw0), math.sin(yaw0)
    q = r.d.qpos[3:7]; yaw = math.atan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[2]**2+q[3]**2))
    Hq = np.degrees(np.array(Hq)); hr = Hq.max(0) - Hq.min(0)
    print(f"{name:12s} vx {(c*d[0]+s*d[1])/T*1000:5.0f} vy {(-s*d[0]+c*d[1])/T*1000:5.0f} mm/s  draai {math.degrees(math.atan2(math.sin(yaw-yaw0), math.cos(yaw-yaw0)))/T:6.1f} gr/s"
          f"  breedte {np.mean(WD):3.0f}  trillen {np.mean(J):4.1f} enkel {np.mean(F):4.1f}  hoofd bereik knik/pitch/rol/yaw {hr.round(0)}" + (f"  VALT na {fell:.1f}s" if fell else ""))
