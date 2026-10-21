#!/bin/bash
# Train and evaluate every experiment, keeping new artifacts under a disk budget.
#
#   ./run_experiments.sh            # resumable: finished runs are skipped
#
# Needs the 150-demo front-camera dataset first:
#   uv run python sim/collect_data.py --task pick_place --num_episodes 150
#
# Each run writes checkpoints/<run>/ (policy_best.ckpt, run_config.json,
# dataset_stats.pkl, eval_video.mp4) and results/<run>.json. The wrist-camera
# runs need a separate 320x240 front+wrist dataset (~3.6 GB). It is collected,
# used, then deleted to stay within budget; collection is seeded, so re-running
# this script regenerates it identically if those runs need repeating.
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl

BUDGET_BYTES=$((5 * 1024 * 1024 * 1024))
EPOCHS=2000
WRIST_DATA=pick_place_wrist320
mkdir -p results logs

# Bytes used by everything this script creates (the pre-existing
# checkpoints/pick_place_chunk50_tagg run is not counted).
artifact_bytes() {
    shopt -s nullglob
    local paths=(results logs "data/$WRIST_DATA") d
    for d in checkpoints/pick_place_*; do
        [[ $d == checkpoints/pick_place_chunk50_tagg ]] || paths+=("$d")
    done
    { du -sbc "${paths[@]}" 2>/dev/null || true; } | awk '/total$/ {t=$1} END {print t+0}'
}

# Abort before a step if it would push artifacts over budget.
#   require_budget <bytes the next step will add>
require_budget() {
    local used; used=$(artifact_bytes)
    if (( used + $1 > BUDGET_BYTES )); then
        echo "!!! Disk budget: $((used >> 20)) MiB used + $(($1 >> 20)) MiB needed > $((BUDGET_BYTES >> 20)) MiB" >&2
        exit 1
    fi
    echo "--- disk: $((used >> 20)) MiB used of $((BUDGET_BYTES >> 20)) MiB budget"
}

CKPT_BYTES=$((340 * 1024 * 1024))

# train_and_eval <dataset> <run suffix> <train.py args...>
train_and_eval() {
    local task=$1 suffix=$2; shift 2
    local run="${task}${suffix}"
    if [[ -f "results/$run.json" ]]; then
        echo "=== $run: already done, skipping"; return
    fi
    require_budget "$CKPT_BYTES"
    echo "=== $run: training ($(date))"
    uv run python train.py --task "$task" --suffix "$suffix" --num_epochs "$EPOCHS" "$@" \
        > "logs/$run.log" 2>&1
    echo "=== $run: evaluating ($(date))"
    uv run python sim/evaluate_sim.py --ckpt "checkpoints/$run/policy_best.ckpt" \
        --num_episodes 50 --video --output "results/$run.json" >> "logs/$run.log" 2>&1
    grep '^Evaluation:' "logs/$run.log" | tail -1 || true
}

# noise_sweep <run>
noise_sweep() {
    local run=$1
    if [[ -f "results/noise_$run.json" ]]; then return; fi
    echo "=== $run: noise sweep ($(date))"
    uv run python scripts/eval_noise.py --ckpt "checkpoints/$run/policy_best.ckpt" \
        --noise_levels 0.0,0.3,0.5,0.7,1.0 --num_episodes 20 \
        --output "results/noise_$run.json" > "logs/noise_$run.log" 2>&1
}

ACT_ARGS=(--policy_class ACT --chunk_size 50 --temporal_agg)

# --- Camera ablation (320x240, front vs front+wrist), run first so its dataset can be freed
if [[ ! -f "results/${WRIST_DATA}_front.json" || ! -f "results/${WRIST_DATA}_front_wrist.json" ]]; then
    if [[ ! -d "data/$WRIST_DATA" ]]; then
        require_budget $((3700 * 1024 * 1024))
        echo "=== collecting $WRIST_DATA ($(date))"
        uv run python sim/collect_data.py --task "$WRIST_DATA" --num_episodes 150 \
            --camera_names front,wrist --cam_size 320x240 > "logs/collect_$WRIST_DATA.log" 2>&1
    fi
    train_and_eval "$WRIST_DATA" _front "${ACT_ARGS[@]}" --cameras front
    train_and_eval "$WRIST_DATA" _front_wrist "${ACT_ARGS[@]}" --cameras front,wrist
fi
rm -rf "data/$WRIST_DATA"

# --- Main comparisons on the 640x480 front-camera dataset
train_and_eval pick_place _act "${ACT_ARGS[@]}"
train_and_eval pick_place _diffusion --policy_class Diffusion --chunk_size 16 --lr 1e-4 \
    --diffusion_steps 100 --ddim_steps 10
train_and_eval pick_place _act_mobilenetv2 "${ACT_ARGS[@]}" --backbone mobilenetv2
train_and_eval pick_place _act_noise0.3 "${ACT_ARGS[@]}" --noise_level 0.3
train_and_eval pick_place _act_noise0.7 "${ACT_ARGS[@]}" --noise_level 0.7

# --- Robustness: how each ACT variant degrades as observations get noisier
for run in pick_place_act pick_place_act_noise0.3 pick_place_act_noise0.7; do
    noise_sweep "$run"
done

echo "=== All experiments complete ($(date)); artifacts: $(( $(artifact_bytes) >> 20 )) MiB"
