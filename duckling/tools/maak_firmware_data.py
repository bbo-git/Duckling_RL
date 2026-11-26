"""maak_firmware_data.py - zet het loopnetwerk, het loopje en de ijking om naar C-headers
voor firmware/duckling_loop (Arduino, RP2350). 2026-11-02.

    pip install onnx numpy
    python tools/maak_firmware_data.py --onnx onnx/duckling_walk_1600.onnx \
        --grid duckling/src/mjlab_microduck/robot/duckling/grids/gait_grid_normaal_v3.npz \
        --calib duckling/firmware/calib.json --uit duckling/firmware/duckling_loop

Schrijft policy_weights.h (netwerk), duck_data.h (gangritme + loopje voor de animatietest)
en calib.h (servo-ijking uit ijk.py). Na elke nieuwe ijking: opnieuw draaien en uploaden.
"""
import argparse, json, os
import numpy as np

# volgorde van de 14 gewrichten = training (actuator-volgorde in robot_walk.xml)
JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_roll", "head_yaw",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
RANGE_DEG = {"hip_yaw": (-20, 20), "hip_roll": (-20, 20), "hip_pitch": (-60, 60), "knee": (-50, 90),
             "ankle": (-60, 90), "neck_pitch": (-30, 30), "head_pitch": (-30, 30), "head_roll": (-15, 15),
             "head_yaw": (-90, 90)}
# filter en geschatte stand: = training (duckling_velocity_env_cfg.py / duckling_play_onnx.py)
ALPHA, ALPHA_HEAD = 0.4, 0.25
HEAD = ("neck_pitch", "head_pitch", "head_roll", "head_yaw")
EST_VMAX = {"hip_yaw": 9.0, "knee": 9.0, "ankle": 9.0, "neck_pitch": 9.0, "head_pitch": 8.1,
            "hip_pitch": 6.0, "hip_roll": 6.0, "head_yaw": 5.4, "head_roll": 5.4}
EST_SPEED_FRAC = 0.75
# servo's (GPIO, OVERDRACHT.md) en welke gewrichten ze bedienen
SERVOS = ["L_diffA", "L_diffB", "R_diffA", "R_diffB", "kop_diffA", "kop_diffB", "L_yaw", "R_yaw",
          "L_knie", "R_knie", "L_enkel", "R_enkel", "nek", "kop_pitch", "kaak"]   # index = GPIO


def key(name):
    for k in RANGE_DEG:
        if name.endswith(k):
            return k
    raise KeyError(name)


def arr(name, a, fmt="%.8ef"):
    a = np.asarray(a, dtype=np.float32).ravel()
    body = ",".join(fmt % x for x in a)
    return f"static const float {name}[{a.size}] = {{{body}}};\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--uit", default=".")
    ap.add_argument("--anim-snelheid", type=float, default=0.5, help="loopje voor 'anim' (fractie van v100)")
    ap.add_argument("--residu", action="store_true",
                    help="2026-11-26: netwerk met residu-acties (doel = loopje uit het rooster + RES x actie)")
    a = ap.parse_args()
    os.makedirs(a.uit, exist_ok=True)

    # --- netwerk
    import onnx
    from onnx import numpy_helper
    m = onnx.load(a.onnx)
    W = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    std = [v for k, v in W.items() if k.startswith("onnx::Div")][0]
    layers = [(W[f"mlp.{i}.weight"], W[f"mlp.{i}.bias"]) for i in (0, 2, 4, 6)]
    with open(os.path.join(a.uit, "policy_weights.h"), "w") as f:
        f.write(f"// gegenereerd uit {os.path.basename(a.onnx)} door maak_firmware_data.py - niet met de hand aanpassen\n")
        f.write("#pragma once\n")
        f.write(f'#define POLICY_NAAM "{os.path.basename(a.onnx)}"\n')
        f.write(f"#define N_OBS {std.size}\n")
        f.write(f"#define POLICY_RESIDUAL {1 if a.residu else 0}\n#define RES_SCALE_LEG 0.30f\n#define RES_SCALE_HEAD 0.20f\n")
        f.write(arr("OBS_MEAN", W["obs_normalizer._mean"]) + arr("OBS_STD", std))
        for i, (w, b) in enumerate(layers):
            f.write(f"#define L{i}_IN {w.shape[1]}\n#define L{i}_OUT {w.shape[0]}\n")
            f.write(arr(f"L{i}_W", w) + arr(f"L{i}_B", b))

    # --- gangritme (cyclusduur per snelheid x bocht) + een loopje voor de animatietest
    d = np.load(a.grid)
    S, Wt, Y = d["speeds"], d["turns"], d["sides"]
    yi = int(np.argmin(np.abs(Y)))
    cyc = d["cycle_s"][:, :, yi]
    names = list(d["joint_names"])
    si = int(np.argmin(np.abs(S - a.anim_snelheid))); ti = int(np.argmin(np.abs(Wt)))
    n = int(d["length"][si, ti, yi])
    q = d["joint_pos"][si, ti, yi, :n]
    anim = np.stack([q[:, names.index(j)] for j in JOINTS], 1)        # (n, 14) in trainingsvolgorde
    lo = [np.radians(RANGE_DEG[key(j)][0]) for j in JOINTS]
    hi = [np.radians(RANGE_DEG[key(j)][1]) for j in JOINTS]
    with open(os.path.join(a.uit, "duck_data.h"), "w") as f:
        f.write(f"// gegenereerd uit {os.path.basename(a.grid)} door maak_firmware_data.py\n#pragma once\n")
        f.write(f"#define V100 {float(d['v100']):.6f}f\n#define N_SPEEDS {len(S)}\n#define N_TURNS {len(Wt)}\n")
        f.write(arr("GRID_SPEEDS", S) + arr("GRID_TURNS_DEG", Wt) + arr("GRID_CYCLE", cyc))
        f.write(f"#define ANIM_N {n}\n#define ANIM_CYCLE {float(d['cycle_s'][si, ti, yi]):.6f}f\n")
        f.write(f"#define ANIM_SPEED_FRAC {float(S[si]):.3f}f\n")
        f.write(arr("ANIM_Q", anim))
        f.write(arr("J_LO", lo) + arr("J_HI", hi))
        # 2026-11-26: het hele rooster (zijstap 0) voor de residu-basis, in trainingsvolgorde
        L = d["length"][:, :, yi]
        T = int(L.max())
        cols = [names.index(j) for j in JOINTS]
        gq = d["joint_pos"][:, :, yi, :T][..., cols]                    # (nS, nW, T, 14)
        f.write(f"#define GRID_T {T}\n")
        f.write(f"#define GRID_WALK_S0 {float(d['walk_s0']) if 'walk_s0' in d else 0.15:.4f}f   // 2026-11-26: onder deze snelheid mengt naar stilstaan\n")
        f.write("static const unsigned char GRID_LEN[%d] = {%s};\n" % (L.size, ",".join(str(int(x)) for x in L.reshape(-1))))
        f.write(arr("GRID_Q", gq))
        f.write(arr("STAND_Q", np.asarray(d["stand_joint_pos"])[cols]))
        f.write(arr("J_ALPHA", [ALPHA_HEAD if j in HEAD else ALPHA for j in JOINTS]))
        f.write(arr("J_VEST", [EST_VMAX[key(j)] * EST_SPEED_FRAC for j in JOINTS]))
        f.write("static const char* const J_NAMES[14] = {" + ",".join(f'"{j}"' for j in JOINTS) + "};\n")

    # --- ijking
    c = json.load(open(a.calib))
    with open(os.path.join(a.uit, "calib.h"), "w") as f:
        f.write(f"// gegenereerd uit {os.path.basename(a.calib)} ({c.get('_info', {}).get('datum', '?')}) door maak_firmware_data.py\n#pragma once\n")
        f.write("static const float CAL_OFFSET[15] = {" + ",".join("%.2ff" % c[s]["offset"] for s in SERVOS) + "};\n")
        f.write("static const int CAL_TEKEN[15] = {" + ",".join("%d" % c[s]["teken"] for s in SERVOS) + "};\n")
        for dn, tag in (("L_heup", "LH"), ("R_heup", "RH"), ("kop", "KOP")):
            f.write(f"#define {tag}_SP {int(c[dn]['sp'])}\n#define {tag}_SR {int(c[dn]['sr'])}\n")
        nog = c.get("_info", {}).get("nog_niet", [])
        f.write(f'#define CAL_NOG_NIET "{", ".join(nog)}"\n')
    print("klaar:", a.uit, "| netwerk", os.path.basename(a.onnx), "| anim", n, "standen, cyclus",
          round(float(d['cycle_s'][si, ti, yi]), 3), "s | nog niet geijkt:", nog)


if __name__ == "__main__":
    main()
