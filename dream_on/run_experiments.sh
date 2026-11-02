#!/usr/bin/env bash
# Dream On v2 — experiment queue. Run from the repository root:
#   caffeinate -i ./dream_on/run_experiments.sh <stage>
# Stages: diffusion | eval_diffusion | baselines | dreamon | ablation | export | all
# Finished runs (runs/<name>/DONE) are skipped, so any stage can be safely re-run after a crash.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CKPT=data/diffusion/ckpt/best_model.pth
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-150000}

stage_diffusion() {
  # 1) build (I_t, I_{t+5}) pairs from D4RL/kitchen/complete-v2 and 2) train the conditional DDPM
  $PY src/diffusion/train.py --k_step_future 5 --img_resolution 120 --batch_size 8 --epochs 100 \
      --learning_rate 1e-4 --save_best_model --checkpoint_freq 10 \
      --dataset_dir data/kitchen_pairs --checkpoint_dir data/diffusion/ckpt \
      --tensorboard_log_dir data/diffusion/tb --resume_from_checkpoint latest
}

stage_eval_diffusion() {
  $PY dream_on/eval_diffusion.py --ckpt $CKPT --dataset D4RL/kitchen/mixed-v2 --episodes 5 \
      --out results/diffusion_eval
  $PY dream_on/eval_diffusion.py --ckpt $CKPT --pairs_pkl data/kitchen_pairs/pairs_k5_120x120.pkl \
      --out results/diffusion_eval_train
}

stage_baselines() {
  for s in $SEEDS; do $PY dream_on/train_rl.py --algo ppo --image none --seed $s --timesteps $STEPS; done
  $PY dream_on/train_rl.py --algo sac --image none --seed 0 --timesteps $STEPS
}

stage_dreamon() {
  for s in $SEEDS; do
    $PY dream_on/train_rl.py --algo ppo --image diffusion --reward l2   --ckpt $CKPT --seed $s --timesteps $STEPS
    $PY dream_on/train_rl.py --algo ppo --image diffusion --reward ssim --ckpt $CKPT --seed $s --timesteps $STEPS
  done
}

stage_ablation() {
  # separates the effect of the extra observation from the effect of the dense reward
  $PY dream_on/train_rl.py --algo ppo --image current   --seed 0 --timesteps $STEPS
  $PY dream_on/train_rl.py --algo ppo --image diffusion --reward sparse --ckpt $CKPT --seed 0 --timesteps $STEPS
}

stage_export() {
  $PY dream_on/export_results.py --runs runs --out results --diffusion_tb data/diffusion/tb
  # small archive with everything needed for the report (no model weights)
  zip -qr results_for_report.zip results runs -i '*.npz' '*.json' '*.csv' '*.md' '*.png' '*.gif' '*tfevents*' \
      && echo "-> results_for_report.zip ($(du -h results_for_report.zip | cut -f1))"
}

case "${1:-}" in
  diffusion) stage_diffusion ;;
  eval_diffusion) stage_eval_diffusion ;;
  baselines) stage_baselines ;;
  dreamon) stage_dreamon ;;
  ablation) stage_ablation ;;
  export) stage_export ;;
  all) stage_diffusion; stage_eval_diffusion; stage_baselines; stage_dreamon; stage_ablation; stage_export ;;
  *) echo "usage: $0 {diffusion|eval_diffusion|baselines|dreamon|ablation|export|all}"; exit 1 ;;
esac
