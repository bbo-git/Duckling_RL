#!/usr/bin/env bash
# duck.sh - Duckling training on RunPod in one command.
#
#   ./duck.sh setup              uv + dependencies (once per new pod)
#   ./duck.sh update file.zip    unpack an update zip from Claude into the repo
#   ./duck.sh library [n]        (re)build gait_library.npz from the gait generator
#   ./duck.sh grid               gait_grid.npz = jouw loopje: snelheid x bocht x zijstap (STYLE=jairo_walk_v1 voor het origineel)
#   ./duck.sh check              reward-volgorde per stijl (eigen loopje moet winnen; V2)
#   DUCK_TASK=volgen ./duck.sh resume 1000   volgen als fine-tune na lopen (standaard DUCK_TASK=lopen)
#   ./duck.sh train [iters]      generate model, check it, (re)start training + tensorboard
#   ./duck.sh resume [iters]     continue the newest run from its last checkpoint
#   ./duck.sh status             progress and the main numbers
#   ./duck.sh export [iter]      checkpoint (newest, or model_<iter>) -> /workspace/duckling_<mode>_<iter>.onnx
#   (checkpoints every 50 iterations; SAVE=250 ./duck.sh train for fewer)
#   ./duck.sh stop               stop training
#
# Put this file in /workspace and run it from anywhere (chmod +x /workspace/duck.sh once).
set -euo pipefail

REPO=${REPO:-/workspace/microduck_rl}   # REPO=/workspace/duck_gisteren ./duck.sh ... voor een andere kopie
ROBOT=$REPO/src/mjlab_microduck/robot/duckling
# Which policy: the steerable walk (default), the universal gait tracker or the old walk.
#   MODE=tracker ./duck.sh train  -> Mjlab-Tracker-Flat-Duckling (random gaits, no steering)
#   MODE=walk ./duck.sh train     -> Mjlab-Velocity-Flat-Duckling
MODE=${MODE:-walk}   # 2026-10-14: walk is de enige taak die echt loopt
if [ "$MODE" = "walk" ]; then
  TASK=Mjlab-Velocity-Flat-Duckling; EXP=duckling_velocity; TAG=walk
elif [ "$MODE" = "tracker" ]; then
  TASK=Mjlab-Tracker-Flat-Duckling; EXP=duckling_tracker; TAG=tracker
else
  TASK=Mjlab-Steer-Flat-Duckling; EXP=duckling_steer; TAG=steer
fi
LOG=/workspace/train.log
RUNS=$REPO/logs/rsl_rl/$EXP
# Parallel environments: from the GPU memory unless ENVS is set (ENVS=4096 ./duck.sh train).
# ~24-32 GB (4090/5090): 4096 | >= 48 GB (A6000, L40S, RTX PRO 6000 96 GB): 8192.
# More envs = more experience per iteration (slower iterations, fewer needed).
if [ -z "${ENVS:-}" ]; then
  mem=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || true)
  if [ "${mem:-0}" -ge 46000 ]; then ENVS=8192; else ENVS=4096; fi
fi
# GPUs: one, unless GPUS is set (GPUS=2 ./duck.sh train; experimental). ENVS is per GPU.
GPUS=${GPUS:-1}
if [ "$GPUS" -gt 1 ]; then GPU_ARGS="--gpu-ids all"; else GPU_ARGS=""; fi
export PATH="$HOME/.local/bin:$PATH"

say() { printf '\n== %s\n' "$*"; }
die() { printf '\n!! %s\n' "$*" >&2; exit 1; }

need_uv() {
  command -v uv >/dev/null 2>&1 && return
  say "uv installeren"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
}

newest_run() { ls -1t "$RUNS" 2>/dev/null | head -1 || true; }
# newest run that has a checkpoint (a crashed start leaves an empty run dir behind)
newest_ckpt_run() {
  local r
  for r in $(ls -1t "$RUNS" 2>/dev/null || true); do
    ls "$RUNS/$r"/model_*.pt >/dev/null 2>&1 && { echo "$r"; return; }
  done
  return 0
}
# With more than one GPU the training output goes to per-GPU files (torchrunx); use GPU 0's.
logfile() {
  local run f
  run=$(newest_run)
  if [ -n "$run" ] && [ -d "$RUNS/$run/torchrunx" ]; then
    f=$(find "$RUNS/$run/torchrunx" -name '*\[0\].log' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    [ -n "$f" ] && { echo "$f"; return; }
  fi
  echo "$LOG"
}
newest_ckpt() {   # newest model_<n>.pt in a run dir, by iteration number
  { ls -1 "$RUNS/$1"/model_*.pt 2>/dev/null || true; } | sed 's/.*model_\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2
}
is_training() { pgrep -f "bin/train $TASK|scripts.train $TASK|train $TASK" >/dev/null 2>&1; }

gen_model() {
  say "walk-model genereren"
  cd "$REPO"
  local missing=""
  for f in duck_joints.csv duck_links.csv; do [ -f "$ROBOT/$f" ] || missing="$missing $f"; done
  [ -d "$ROBOT/meshes" ] || missing="$missing meshes/"
  if [ -n "$missing" ]; then
    echo "in $ROBOT staat:"; ls -la "$ROBOT"
    die "ontbreekt:$missing"
  fi
  uv run python scripts/gen_duck_mjcf.py "$ROBOT/duck_joints.csv" "$ROBOT/duck_links.csv" \
    "$ROBOT/robot_walk.xml" --walk 2>&1 | grep -v "lijkt in m" || true
  local n; n=$(grep -c "_collision" "$ROBOT/robot_walk.xml" || true)
  [ "$n" = "8" ] || die "robot_walk.xml heeft $n botsingsgeoms (verwacht 8) - gegenereerd zonder --walk?"
  if [ "$MODE" = "walk" ]; then
    [ -f "$ROBOT/walk_cycle.npz" ] || die "walk_cycle.npz ontbreekt in $ROBOT"
    echo "ok: 8 botsingsgeoms, walk_cycle.npz aanwezig"
  elif [ "$MODE" = "steer" ]; then
    [ -f "$ROBOT/gait_grid.npz" ] || die "gait_grid.npz ontbreekt in $ROBOT (maak hem met: ./duck.sh grid)"
    [ -f src/mjlab_microduck/tasks/duckling_steer_env_cfg.py ] || die "duckling_steer_env_cfg.py ontbreekt"
    echo "ok: 8 botsingsgeoms, gait_grid.npz aanwezig"
  else
    [ -f "$ROBOT/gait_library.npz" ] || die "gait_library.npz ontbreekt in $ROBOT (maak hem met: ./duck.sh library)"
    [ -f src/mjlab_microduck/tasks/duckling_tracker_env_cfg.py ] || die "duckling_tracker_env_cfg.py ontbreekt"
    echo "ok: 8 botsingsgeoms, gait_library.npz aanwezig"
  fi
}

start_tensorboard() {
  if ss -ltn 2>/dev/null | grep -q ":6006 "; then return; fi
  cd "$REPO"
  nohup uv run tensorboard --logdir "$RUNS" --bind_all --port 6006 >/dev/null 2>&1 &
  echo "tensorboard gestart op poort 6006"
}

stop_training() {
  if is_training; then
    pkill -f "train $TASK" || true
    sleep 3
    echo "training gestopt"
  fi
}

launch() {   # launch "<extra args>"
  cd "$REPO"
  # shellcheck disable=SC2086
  local before; before=$(newest_run)
  nohup uv run train "$TASK" --env.scene.num-envs "$ENVS" --agent.logger tensorboard --agent.save-interval "${SAVE:-50}" $GPU_ARGS $1 > "$LOG" 2>&1 &
  say "training gestart op $GPUS GPU('s) met $ENVS omgevingen per GPU (log: $LOG)"
  local L="$LOG"
  for _ in $(seq 1 120); do
    sleep 2
    if [ "$GPUS" -gt 1 ] && [ "$(newest_run)" = "$before" ]; then continue; fi   # new run dir not there yet
    L=$(logfile)
    if grep -q "Learning iteration" "$L" 2>/dev/null; then break; fi
    if ! is_training; then tail -30 "$LOG"; [ "$L" != "$LOG" ] && tail -30 "$L"; die "training is gestopt tijdens het opstarten (zie hierboven)"; fi
  done
  [ "$L" != "$LOG" ] && echo "log van GPU 0: $L"
  grep -m1 -E "(walk_imitation|joint_tracking) +\|" "$L" || true
  grep -E "Learning iteration" "$L" | tail -1 || echo "(nog aan het opstarten)"
  start_tensorboard
  echo
  echo "Meekijken:  ./duck.sh status    of    tail -f $(logfile)"
}

case "${1:-}" in
  setup)
    need_uv
    cd "$REPO"
    say "dependencies (uv sync)"
    uv sync
    echo "klaar"
    ;;

  update)
    zip="${2:-}"; [ -f "$zip" ] || die "gebruik: ./duck.sh update /workspace/<bestand>.zip"
    say "uitpakken: $zip"
    python3 - "$zip" "$REPO" <<'EOF'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
for n in z.namelist():
    if not n.endswith("/"):
        print("  ", n)
z.extractall(sys.argv[2])
EOF
    echo "klaar - start opnieuw met ./duck.sh train"
    ;;

  train)
    need_uv
    iters="${2:-3000}"
    stop_training
    gen_model
    launch "--agent.max-iterations $iters"
    ;;

  library)
    need_uv
    cd "$REPO"
    say "gangenbibliotheek bouwen (${2:-700} kandidaten)"
    cd "$ROBOT" && uv run --project "$REPO" python "$REPO/scripts/gait_lib.py" --n "${2:-700}" --out gait_library.npz
    ;;

  grid)
    need_uv
    cd "$ROBOT" && uv run --project "$REPO" python "$REPO/scripts/walk_speed.py" \
      --style "favorites/${STYLE:-jairo_walk_v2}.npz" --grid gait_grid.npz --check
    ;;

  check)
    need_uv
    cd "$REPO"
    uv run python scripts/check_rewards_v2.py 2>&1 | sed -n '/=== stijl/,$p'
    ;;

  resume)
    need_uv
    iters="${2:-1500}"
    run=$(newest_ckpt_run); [ -n "$run" ] || die "geen run met een checkpoint gevonden in $RUNS"
    ck=$(newest_ckpt "$run"); [ -n "$ck" ] || die "geen checkpoint in $run"
    stop_training
    say "hervatten: $run / $(basename "$ck")"
    launch "--agent.max-iterations $iters --agent.resume True --agent.load-run $run --agent.load-checkpoint $(basename "$ck")"
    ;;

  status)
    if is_training; then echo "training: loopt"; else echo "training: gestopt"; fi
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/GPU /'
    L=$(logfile)
    [ -f "$L" ] || die "geen $L"
    python3 - "$L" <<'EOF'
import re, sys
txt = open(sys.argv[1], errors="ignore").read()
txt = re.sub(r"\x1b\[[0-9;]*m", "", txt)
blocks = txt.split("Learning iteration")
if len(blocks) < 2:
    print("nog geen iteraties"); sys.exit()
last = blocks[-1]
it = re.match(r"\s*(\d+)/(\d+)", last)
print(f"iteratie {it.group(1)} van {it.group(2)}")
def val(key):
    m = re.search(re.escape(key) + r":\s*(-?[\d.]+)", last)
    return float(m.group(1)) if m else None
rows = [("Mean episode length", "episodelengte (stappen)"),
        ("Mean action std", "actieruis"),
        ("Episode_Reward/alive", "alive (0..1, overeind blijven)"),
        ("Episode_Reward/walk_imitation", "walk_imitation (max ~5)"),
        ("Episode_Reward/joint_tracking", "joint_tracking (max ~5)"),
        ("Episode_Reward/contact_tracking", "contact_tracking (max ~1.5)"),
        ("Episode_Reward/speed_tracking", "speed_tracking (max ~2)"),
        ("Episode_Reward/heading_hold", "heading_hold (max ~1.5)"),
        ("Episode_Reward/velocity_tracking", "velocity_tracking (max ~2)"),
        ("Episode_Reward/yaw_rate_tracking", "yaw_rate_tracking (max ~1.5)"),
        ("Episode_Reward/contact_schedule", "contact_schedule (max ~2)"),
        ("Episode_Reward/trunk_height", "trunk_height (max ~1)"),
        ("Episode_Reward/track_linear_velocity", "track_linear_velocity (max ~2)"),
        ("Metrics/twist/error_vel_xy", "snelheidsfout m/s (lager = beter)"),
        ("Episode_Reward/stance_width", "stance_width (richting 0)"),
        ("Episode_Reward/self_collisions", "self_collisions (richting 0)"),
        ("ETA", None)]
alive = val("Episode_Reward/alive") or 0
for key, label in rows:
    if key == "ETA":
        m = re.search(r"ETA:\s*(\S+)", last)
        if m: print(f"  {'nog te gaan':38s} {m.group(1)}")
        continue
    v = val(key)
    if v is None: continue
    extra = ""
    if key == "Episode_Reward/track_linear_velocity" and alive > 0.05:
        extra = f"   (per seconde overeind: {v / alive:.2f})"
    print(f"  {label:38s} {v:9.4f}{extra}")
EOF
    run=$(newest_ckpt_run); [ -n "$run" ] && echo "laatste checkpoint: $run/$(basename "$(newest_ckpt "$run")")" || true
    ;;

  export)
    need_uv
    run=$(newest_ckpt_run); [ -n "$run" ] || die "geen run met een checkpoint gevonden"
    if [ -n "${2:-}" ]; then ck="$RUNS/$run/model_$2.pt"; [ -f "$ck" ] || die "geen $ck"
    else ck=$(newest_ckpt "$run"); [ -n "$ck" ] || die "geen checkpoint in $run"; fi
    it=$(basename "$ck" .pt | sed 's/model_//')
    out=/workspace/duckling_${TAG}_${it}.onnx
    say "exporteren: $run / $(basename "$ck")"
    cd "$REPO"
    uv run scripts/export.py "$TASK" --checkpoint-file "$ck" --onnx-file "$out" 2>&1 | tail -1
    cp "$out" /workspace/duckling_${TAG}_latest.onnx
    echo "klaar: $out  (en /workspace/duckling_${TAG}_latest.onnx)"
    echo "download via VS Code: rechtsklik op het bestand in /workspace -> Download"
    ;;

  stop)
    stop_training
    ;;

  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
