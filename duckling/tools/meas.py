import sys, math, numpy as np, mujoco
import duckling_play_onnx as P
pol = P.Policy(sys.argv[1]); GRID = P.GridClock("gait_grid.npz") if len(sys.argv) > 2 else None
for vx, wz in ((0.03,0),(0.06,0),(0.09,0),(0.12,0),(0.0,0),(0.05,0.5),(0.05,-0.5)):
    r = P.Duckling("scene_walk.xml", grid=GRID); r.gait_clock = pol.n_in == 63
    x0 = r.d.qpos[:2].copy(); J=[]; la=np.zeros(14); Z=[]; yaw0=None
    for k in range(500):
        t=np.array([vx,0,wz]); a = pol(r.obs(t, np.zeros(4))); r.step(a, t)
        if k>50: J.append(np.degrees(np.abs(a-la).mean()))
        la=a
        q=r.d.qpos[3:7]; yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        if k==50: yaw0=yaw; x50=r.d.qpos[:2].copy()
        if k>50: Z.append([r.d.site("left_foot").xpos[2], r.d.site("right_foot").xpos[2]])
        if r.fallen: break
    Z=np.array(Z)*1000; lift=np.abs(Z[:,0]-Z[:,1]).max()
    T=(500-50)*P.TIMESTEP*P.DECIMATION; d=r.d.qpos[:2]-x50
    c,s=math.cos(yaw0),math.sin(yaw0); fwd=(c*d[0]+s*d[1])/T
    dyaw=math.degrees(math.atan2(math.sin(yaw-yaw0),math.cos(yaw-yaw0)))/T
    print(f"vx {vx:.2f} wz {wz:.1f}: vooruit {fwd*1000:5.0f} mm/s  draai {dyaw:6.1f} gr/s (gevraagd {math.degrees(wz):.0f})  voet-op max {lift:4.1f} mm  trillen {np.mean(J):4.1f}  {'VALT' if r.fallen else ''}")
