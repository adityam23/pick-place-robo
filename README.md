# pick-place-robo

Imitation learning on a simulated **Franka Emika Panda** 7-DoF arm doing pick-and-place. The arm sees a single front camera and its own joint state, and outputs joint targets. Policies are trained end-to-end from scripted-expert demonstrations in MuJoCo.

The current baseline is **ACT** (Action Chunking with Transformers, Zhao et al. 2023). Next experiments — Diffusion Policy and sim-to-real noise robustness — are sketched in the [Roadmap](#roadmap) below.

> **Quick eval (no training):**
> 1. Set up the environment (see [Setup](#setup)).
> 2. Download `policy_best.ckpt` and `dataset_stats.pkl` from the [v0.1 release](#) into `checkpoints/pick_place/`.
> 3. `uv run python sim/evaluate_sim.py --ckpt checkpoints/pick_place/policy_best.ckpt --chunk_size 50 --temporal_agg --video`

## Results

Single ACT policy trained on **150 scripted demonstrations**, chunk size 50, temporal aggregation on, 2000 epochs. Headline numbers come from a 50-episode eval in MuJoCo with fresh random cube/target placements (seeds 5000–5049):

| Metric | Value |
|---|---|
| Success rate (cube within 5 cm of target) | **78 %** (39/50) |
| Final cube-to-target distance — successful trials | 1.9 cm mean |
| Final cube-to-target distance — failed trials | 20.7 cm mean |
| Best validation loss | 0.032 (epoch 1462) |
| Training epochs | 2000 |
| Demonstrations | 150 |

When the policy succeeds, it places the cube within ~2 cm of the target — well inside the 5 cm threshold. The 11 failures cluster around ~20 cm, which corresponds to either dropped grasps or the cube being knocked away during transport. Full per-episode numbers in [`results/pick_place_150ep.json`](results/pick_place_150ep.json).

Demo video: [v0.1 release asset](#).

## Setup

Python 3.11 + [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install git+https://github.com/Shaka-Labs/detr.git
uv run python scripts/patch_detr.py
```

That's the whole setup. The Franka MJCF model is bundled at `mujoco_menagerie/franka_emika_panda/` — no external clone needed.

### What the patch script does

The Shaka-Labs DETR fork hardcodes `state_dim=5` (for the ALOHA bimanual rig it was written for). This project uses `state_dim=8` (7 Panda joints + 1 gripper), so `scripts/patch_detr.py` rewrites two installed-package files in place. Re-running is safe — it detects an already-patched file and skips. Vendoring the patched DETR fork into the repo is on the [roadmap](#roadmap).

## Reproducing the result

Two paths.

### Fastest: use the released checkpoint

```bash
mkdir -p checkpoints/pick_place
# Download policy_best.ckpt + dataset_stats.pkl from the v0.1 release page,
# then place them under checkpoints/pick_place/

uv run python sim/evaluate_sim.py \
    --task pick_place \
    --ckpt checkpoints/pick_place/policy_best.ckpt \
    --chunk_size 50 --temporal_agg \
    --num_episodes 50 --video
```

### From scratch (~half a day on a GPU)

```bash
# 1. Collect 150 expert demos (HDF5 episodes in data/pick_place/)
uv run python sim/collect_data.py --task pick_place --num_episodes 150

# 2. Train ACT — 2000 epochs, chunk 50, temporal aggregation on
uv run python train.py --task pick_place \
    --chunk_size 50 --temporal_agg \
    --suffix _chunk50_tagg \
    --eval_every 500 --eval_episodes 5

# 3. Evaluate
uv run python sim/evaluate_sim.py --task pick_place \
    --ckpt checkpoints/pick_place_chunk50_tagg/policy_best.ckpt \
    --chunk_size 50 --temporal_agg --num_episodes 50 --video
```

Seeds are fixed (training seed 42; eval seeds 5000–5049). The scripted expert is itself ~100 % successful — check that first with `uv run python sim/scripted_expert.py`.

## Architecture

```
                 ┌──────── front camera (640×480 RGB) ───────┐
                 │                                            ▼
qpos (8) ──► transformer encoder (CVAE) ◄── ResNet18 image features
                 │
                 ▼
   transformer decoder (action queries)
                 │
                 ▼
   action chunk: (chunk_size, 8) joint targets
```

- **Observation:** 1 fixed front camera + 8-dim joint state (7 arm joints + 1 gripper finger).
- **Action:** 8-dim joint position targets, predicted in chunks of 50 timesteps.
- **Policy:** ACT (Action Chunking with Transformers) — a CVAE-conditioned transformer encoder-decoder. ResNet18 vision backbone, hidden dim 512, 4 encoder + 7 decoder layers, 8 heads.
- **Inference:** temporal aggregation smooths overlapping action chunks across timesteps.
- **Expert:** damped-least-squares IK driving a 9-phase state machine (`APPROACH → DESCEND → GRASP → LIFT → TRANSPORT → LOWER → RELEASE → RETREAT → DONE`).
- **Simulator:** MuJoCo. Scene = Franka Panda + table + 2 cm red cube (free body) + green target marker. Cube and target are randomly placed at each reset within `x ∈ [0.35, 0.65]`, `y ∈ [-0.2, 0.2]`. Episode length is 300 steps; success = cube within 5 cm of target at episode end.

See [`config/config.py`](config/config.py) for all hyperparameters and [`sim/env.py`](sim/env.py) for the scene.

## Roadmap

The repo is intentionally named `pick-place-robo` rather than `act-pick-place` because the next two experiments swap out the policy or attack a different axis.

### 1. Diffusion Policy baseline

Diffusion Policy (Chi et al. 2023) has overtaken CVAE-based methods on a number of bimanual manipulation benchmarks because its action-space denoising captures multi-modality better than a Gaussian-parameterized CVAE. The plan: implement a `DiffusionPolicy` class alongside `ACTPolicy`, share the ResNet18 vision encoder, train on the same 150-episode dataset, and compare success rate, inference latency, and parameter count head-to-head. The interesting question is whether the multi-modality argument shows up at all for a deterministic scripted expert, or only when human teleop demos are used.

### 2. Sim-to-real noise robustness

The current ACT was trained on perfectly clean MuJoCo observations. Realistic deployment imposes camera noise, joint-encoder jitter, occasional gripper failures, and lighting variation. Plan: add a noise-injection layer to the dataloader (Gaussian image noise + blur, joint position/velocity jitter, randomized gripper failure rate), train at several noise levels, and plot success-rate degradation curves. Useful both as a robustness story and a precursor to actual hardware deployment.

### Smaller items

- **Vendor the patched DETR fork** so `scripts/patch_detr.py` becomes unnecessary.
- **Multi-camera ablation** (front-only vs. wrist-only vs. both) — `policy.py` already takes a `camera_names` list.
- **Switch backbone** from ResNet18 to MobileNetV2 for an inference-latency comparison.

## References

- Zhao et al. (2023). *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
- Chi et al. (2023). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
- Base codebase: [Shaka-Labs/ACT](https://github.com/Shaka-Labs/ACT)
- Robot model: [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Franka Emika Panda, MIT)
