# Dream On

Diffusion-generated visual subgoals and pixel-space reward shaping for PPO in sparse-reward robotic manipulation.

This Machine Learning II project builds on *A Dream Is All You Need* by Devin Gupta, Ali Ahmad and Annie Lee (Stanford CS224R, 2025). It compares state-only PPO and SAC with PPO conditioned on diffusion-generated subgoals, using L2 or SSIM shaping on the FrankaKitchen kettle task.

The project adds DDIM sampling, a unified RL training wrapper, subgoal-quality evaluation and multi-seed comparisons. The low-level policy receives numerical robot/object observations as well as an image in the image-conditioned conditions; this is not a purely vision-based agent.

## Environment and installation

The archived experiments were run with Python 3.12 and PyTorch 2.7.0 on a MacBook Air with Apple Silicon and 8 GB RAM. Diffusion inference used the MPS backend. Other platforms and accelerators may produce different results and require platform-specific MuJoCo rendering configuration.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) on macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/evazink/dream-on-dunno.git
cd dream-on-dunno
uv sync --locked --python 3.12
```

If you downloaded the repository as a ZIP, extract it and open a terminal in its root directory instead of running `git clone`. All commands below assume the repository root is the working directory. Commands use `uv run`, so activating the virtual environment separately is unnecessary.

The report's audited code and archived results correspond to commit `efa2608bf202c8a0dc3697f74456cc4410c42f05`. For that exact source snapshot, check it out before installing dependencies:

```bash
git checkout efa2608bf202c8a0dc3697f74456cc4410c42f05
```

That historical snapshot predates this expanded README; keep these instructions separately if checking it out.

## Repository contents

| Path | Contents |
| --- | --- |
| `dream_on/common.py` | Frame preprocessing, DDIM/legacy predictors, image distances and environment wrapper |
| `dream_on/train_rl.py` | PPO/SAC training and periodic sparse-reward evaluation |
| `dream_on/eval_diffusion.py` | Future-frame reconstruction evaluation |
| `dream_on/export_results.py` | Full-run aggregation and plots |
| `dream_on/run_experiments.sh` | Experiment queue with configurable seeds and budgets |
| `src/data_utils/dataloader.py` | Generation and caching of rendered training pairs |
| `src/diffusion/` | Conditional U-Net and DDPM training |
| `data/diffusion/ckpt/best_model.pth` | Archived trained diffusion checkpoint |
| `runs/` | Archived per-run evaluation arrays, monitor logs and metadata |
| `results/` | Saved aggregate results and diffusion-quality measurements |

## Inspect the saved results without retraining

The archived outputs can be inspected directly:

- `runs/<run_name>/evaluations.npz`: evaluation checkpoints and episode returns.
- `runs/<run_name>/train.monitor.csv`: training episode rewards, lengths and success indicators.
- `runs/<run_name>/eval.monitor.csv`: evaluation episode records.
- `runs/<run_name>/meta.json`: arguments, software metadata and recorded elapsed training time.
- `results/diffusion_eval/diffusion_eval.json`: validation reconstruction metrics on 292 frame pairs.
- `results/diffusion_eval_train/diffusion_eval.json`: reconstruction metrics on 300 sampled cached training pairs.

The dataset cache and all trained RL policy files are not guaranteed to be included in the repository. Reading the archived evaluation arrays does not require these files.

## Quick training check

Run a short state-only PPO experiment to check the training setup:

```bash
uv run python dream_on/train_rl.py \
  --algo ppo --image none --reward sparse \
  --seed 0 --timesteps 4096 --no_gif \
  --logdir runs_smoke
```

This checks whether the environment and training loop run; it is not expected to solve the task. It does not exercise image rendering or diffusion inference.

A run with a `DONE` marker is skipped. Use a new `--logdir` or `--tag` for a new experiment. The RL script does not resume a partially trained policy automatically; do not reuse an interrupted run directory for a clean rerun.

## Data and diffusion checkpoint

Source demonstrations are accessed through [Minari](https://minari.farama.org/) and their recorded actions are replayed in Gymnasium-Robotics FrankaKitchen to obtain images.

- Generator training source: `D4RL/kitchen/complete-v2` (19 source episodes, 4,209 transitions).
- Generator validation source: the first five episodes of `D4RL/kitchen/mixed-v2`.
- Frames: RGB render resized to 120 Г— 120, converted to grayscale, then normalised to [в€’1, 1].
- Nominal prediction horizon and subgoal refresh interval: five environment steps.

The archived generator checkpoint is used by default:

```text
data/diffusion/ckpt/best_model.pth
```

RL runs with `--image diffusion` require this checkpoint, but do not require the cached training image pairs. Validation-pair construction downloads the relevant Minari dataset if necessary and requires working MuJoCo rendering.

### Optional generator retraining

To train a new generator without replacing the archived checkpoint:

```bash
uv run python src/diffusion/train.py \
  --k_step_future 5 --img_resolution 120 \
  --batch_size 8 --epochs 100 --learning_rate 1e-4 \
  --save_best_model --checkpoint_freq 10 \
  --dataset_dir data/kitchen_pairs \
  --checkpoint_dir data/diffusion/retrained_ckpt \
  --tensorboard_log_dir data/diffusion/retrained_tb \
  --resume_from_checkpoint none
```

The loader uses an existing `data/kitchen_pairs/pairs_k5_120x120.pkl` cache if present, or creates it otherwise. A new generator checkpoint is saved under `data/diffusion/retrained_ckpt/`; pass its path explicitly with `--ckpt` to use it in subsequent commands. A freshly generated cache and a new training run need not reproduce the archived weights or metrics exactly.

## Evaluate the archived generator

Compare copying the current frame, legacy inference and DDIM with 5, 10, 20 and 50 sampling steps:

```bash
uv run python dream_on/eval_diffusion.py \
  --ckpt data/diffusion/ckpt/best_model.pth \
  --dataset D4RL/kitchen/mixed-v2 \
  --episodes 5 --k 5 --stride 5 --max_pairs 300 \
  --ddim 5,10,20,50 --seed 0 \
  --out results_reproduction/diffusion_eval
```

If the training-pair cache exists, evaluate that cache separately:

```bash
uv run python dream_on/eval_diffusion.py \
  --ckpt data/diffusion/ckpt/best_model.pth \
  --pairs_pkl data/kitchen_pairs/pairs_k5_120x120.pkl \
  --stride 5 --max_pairs 300 --ddim 5,10,20,50 --seed 0 \
  --out results_reproduction/diffusion_eval_train
```

Each command writes `diffusion_eval.json` and a qualitative comparison image. Metrics are RMSE, SSIM and RMSE on pixels whose absolute current-to-future intensity change exceeds 0.05. Runtime is amortised per image in batches of 16, not single-image online latency. The mixed-v2 pairs were used to select the DDIM step count, so they serve as validation data rather than an untouched final test set.

## RL experiment settings

All conditions use the kettle task, a 50-step episode limit, evaluation every 2,000 training transitions and five deterministic-policy evaluation episodes. Image-conditioned PPO uses 20 DDIM steps, refreshes its subgoal every five transitions and uses shaping weight 0.1 where applicable.

| Condition | Algorithm | Image input | Training reward | Seeds | Requested transitions per seed |
| --- | --- | --- | --- | --- | --- |
| C1 | PPO | None | Sparse | 0вЂ“7 | 150,000 |
| C2 | SAC | None | Sparse | 0 | 150,000 |
| C3 | PPO | Generated subgoal | Sparse в€’ 0.1 Г— RMSE | 0вЂ“2 | 100,000 |
| C4 | PPO | Generated subgoal | Sparse в€’ 0.1 Г— (1 в€’ SSIM) | 0вЂ“2 | 100,000 |
| C5 | PPO | Current frame | Sparse | 0 | 100,000 |
| C6 | PPO | Generated subgoal | Sparse | 0 | 100,000 |

Evaluation always uses the sparse task reward, even when training uses shaping. Observation statistics are synchronised from training and frozen for evaluation, and evaluation reward normalisation is disabled. The evaluation environment is initially seeded with the training seed plus 1,000. PPO may collect slightly more transitions than requested to complete its rollout.

### Run the report's RL conditions

The following Bash commands write new runs to `runs_reproduction/`, preserving the archived `runs/` directory. They may take many hours on a laptop. Use another fresh directory for a further independent rerun.

```bash
# C1: state-only PPO, eight seeds
for seed in 0 1 2 3 4 5 6 7; do
  uv run python dream_on/train_rl.py \
    --algo ppo --image none --reward sparse \
    --seed "$seed" --timesteps 150000 --logdir runs_reproduction
done

# C2: state-only SAC
uv run python dream_on/train_rl.py \
  --algo sac --image none --reward sparse \
  --seed 0 --timesteps 150000 --logdir runs_reproduction

# C3 and C4: diffusion subgoals with L2 or SSIM shaping
for seed in 0 1 2; do
  for reward in l2 ssim; do
    uv run python dream_on/train_rl.py \
      --algo ppo --image diffusion --reward "$reward" \
      --ckpt data/diffusion/ckpt/best_model.pth \
      --lam 0.1 --ddim_steps 20 --goal_every 5 \
      --seed "$seed" --timesteps 100000 --logdir runs_reproduction
  done
done

# C5: current-frame observation, sparse reward
uv run python dream_on/train_rl.py \
  --algo ppo --image current --reward sparse \
  --seed 0 --timesteps 100000 --logdir runs_reproduction

# C6: generated-subgoal observation, sparse reward
uv run python dream_on/train_rl.py \
  --algo ppo --image diffusion --reward sparse \
  --ckpt data/diffusion/ckpt/best_model.pth \
  --ddim_steps 20 --goal_every 5 \
  --seed 0 --timesteps 100000 --logdir runs_reproduction
```

`--device auto` selects MPS, CUDA or CPU for the diffusion sampler, in that order of availability. This flag does not explicitly select the SB3 policy-training device. The trained policy, normalisation statistics, evaluation logs and optional rollout GIF are saved in each new run directory.

## Recompute the report's common-horizon RL metrics

All learning comparisons in the report use evaluation checkpoints at or below **100,000 transitions**, including for the 150,000-transition baselines.

- Success rate: mean undiscounted sparse return across the five evaluation episodes.
- Threshold reached: a trailing three-evaluation mean success rate of at least 0.8.
- Steps to threshold: the last checkpoint of the first qualifying window; averaged only across qualifying seeds.
- Final success: mean of the last five evaluations within the common horizon.
- AUC: trapezoidal integration of the **unsmoothed** success rate, divided by the interval from the first to the last included checkpoint.
- Seed dispersion: population standard deviation (`ddof=0`).

The archived `dream_on/export_results.py` aggregates the entire available run. Therefore its full-run PPO/SAC summaries differ from the report's common-horizon metrics. To recompute the report values from the archived arrays without retraining, run:

```bash
uv run python - <<'PY'
from pathlib import Path
import numpy as np

ROOT = Path("runs")  # Change to runs_reproduction for your new runs.
conditions = [
    ("C1", "ppo_none_sparse"),
    ("C2", "sac_none_sparse"),
    ("C3", "ppo_diffusion_l2_lam0.1"),
    ("C4", "ppo_diffusion_ssim_lam0.1"),
    ("C5", "ppo_current_sparse"),
    ("C6", "ppo_diffusion_sparse"),
]
for label, prefix in conditions:
    records = []
    for path in sorted(ROOT.glob(prefix + "_s*/evaluations.npz")):
        with np.load(path) as data:
            keep = data["timesteps"] <= 100_000
            steps = data["timesteps"][keep]
            success = data["results"][keep].mean(axis=1)
        if len(steps) < 5 or steps[-1] != 100_000:
            raise ValueError(f"Incomplete common-horizon evaluations: {path}")
        smooth = np.convolve(success, np.ones(3) / 3, mode="valid")
        hits = np.flatnonzero(smooth >= 0.8)
        threshold = int(steps[hits[0] + 2]) if len(hits) else None
        auc = np.trapezoid(success, steps) / (steps[-1] - steps[0])
        records.append((threshold, float(success[-5:].mean()), float(auc)))
    if not records:
        raise ValueError(f"No evaluations found for {label} in {ROOT}")
    hits = [r[0] for r in records if r[0] is not None]
    final = np.array([r[1] for r in records])
    auc = np.array([r[2] for r in records])
    threshold_text = (
        f"{np.mean(hits):.0f} +/- {np.std(hits):.0f}" if hits else "not reached"
    )
    print(
        f"{label}: seeds={len(records)}, reached={len(hits)}/{len(records)}, "
        f"steps={threshold_text}, "
        f"final={final.mean():.3f} +/- {final.std():.3f}, "
        f"AUC={auc.mean():.3f} +/- {auc.std():.3f}"
    )
PY
```

For full-run summaries and plots in a separate output directory:

```bash
uv run python dream_on/export_results.py --runs runs --out results_full_run
```

This second command does not apply the report's 100,000-transition filter.

## Results and interpretation

Within the common horizon, state-only PPO reaches the threshold in 4/8 seeds, L2-shaped PPO in 2/3, SSIM-shaped PPO in 0/3, current-frame PPO in 1/1 and sparse-subgoal PPO in 0/1. The single SAC baseline does not reach the threshold. These observations do not establish a statistically reliable improvement from generated subgoals or shaping.

Recorded `wall_clock_sec` includes periodic evaluation and cooling pauses. Training was paused when the laptop overheated; one SSIM run contains an approximately 408-minute interval between adjacent episode records. These elapsed times are not controlled active-compute benchmarks.

## Reproducibility limitations

- The training loader samples episodes separately in batches, allowing repeats across batches; the 19 source episodes are not guaranteed to be represented uniquely.
- Frame-pair targets are clipped at memory-chunk boundaries as well as episode ends, shortening some nominal five-step targets.
- Recorded actions are replayed from resets without explicitly restoring the recorded simulator states.
- Training and evaluation share a diffusion predictor and its random-number generator in the archived implementation; evaluation affects the subsequent training noise sequence.
- SSIM is calculated directly on [в€’1, 1] intensities. Computing it after conversion to [0, 1] is a different variant and was not tested in the archived RL experiments.
- Each shaped-reward condition has three seeds, and each ablation has only one. Evaluation has five episodes per checkpoint.
- Matching seeds does not guarantee identical results across hardware, software versions or newly generated training caches.

The commands above describe the archived implementation and its settings; they do not silently correct these limitations or guarantee recovery of identical trained policies.

## Experiment queue

`dream_on/run_experiments.sh` provides `diffusion`, `eval_diffusion`, `baselines`, `dreamon`, `ablation`, `export` and `all` stages. Its defaults are three seeds and 150,000 transitions. Consequently, `all` is not an exact reproduction of the report's experiment matrix. The `eval_diffusion` stage also expects the cached training pairs, and completed runs are skipped. Prefer the explicit commands above when matching the report.

## References and attribution

- Gupta, D., Ahmad, A., and Lee, A. (2025). [A Dream Is All You Need](https://cs224r.stanford.edu/projects/pdfs/CS224R_Final_Paper12.pdf). Stanford CS224R final project report.
- Schulman et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Ho, Jain, and Abbeel (2020). [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).
- Song, Meng, and Ermon (2021). [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502).
- Wang et al. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. IEEE Transactions on Image Processing, 13(4), 600вЂ“612.

The reused diffusion architecture and data pipeline are under `src/`. New sampling, RL training and evaluation components are under `dream_on/`; the dataset pipeline also includes the frame-cloning storage fix. This project should be interpreted as a reimplementation and experimental extension, rather than an independent origin of the full method.
