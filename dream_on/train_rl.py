"""
Dream On v2 — unified RL training script (PPO / SAC) for FrankaKitchen-v1.

Examples
--------
# state-only PPO baseline
python dream_on/train_rl.py --algo ppo --image none --seed 0
# PPO + current frame (authors' "custom env" condition)
python dream_on/train_rl.py --algo ppo --image current --seed 0
# Dream On: PPO + diffusion subgoal + dense L2 reward
python dream_on/train_rl.py --algo ppo --image diffusion --reward l2 --ckpt data/diffusion/ckpt/best_model.pth --seed 0
# Dream On + SSIM reward (our modification)
python dream_on/train_rl.py --algo ppo --image diffusion --reward ssim --ckpt data/diffusion/ckpt/best_model.pth --seed 0

Evaluation always uses the environment's sparse reward, so eval/mean_reward == success rate
for every condition and curves are directly comparable.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dream_on.common import DreamOnWrapper, SubgoalPredictor  # noqa: E402

gym.register_envs(gymnasium_robotics)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--image", choices=["none", "current", "diffusion"], default="none")
    p.add_argument("--reward", choices=["sparse", "l2", "ssim"], default="sparse")
    p.add_argument("--lam", type=float, default=0.1, help="weight of the pixel-space shaping term")
    p.add_argument("--goal_every", type=int, default=5, help="re-sample subgoal every N steps (= k)")
    p.add_argument("--ddim_steps", type=int, default=20)
    p.add_argument("--ckpt", type=str, default="data/diffusion/ckpt/best_model.pth")
    p.add_argument("--task", type=str, default="kettle")
    p.add_argument("--max_episode_steps", type=int, default=50)
    p.add_argument("--timesteps", type=int, default=150_000)
    p.add_argument("--eval_freq", type=int, default=2_000)
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", help="device for the diffusion sampler")
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--n_steps", type=int, default=2048, help="PPO rollout length (keep 2048 for real runs)")
    p.add_argument("--no_gif", action="store_true")
    return p.parse_args()


def make_env_fn(a, reward_mode, predictor, log_path=None):
    def _f():
        env = gym.make("FrankaKitchen-v1", tasks_to_complete=[a.task],
                       render_mode="rgb_array", max_episode_steps=a.max_episode_steps)
        env = DreamOnWrapper(env, image_mode=a.image, reward_mode=reward_mode, lam=a.lam,
                             goal_every=a.goal_every, predictor=predictor)
        return Monitor(env, log_path, info_keywords=("is_success",))
    return _f


def save_rollout_gif(model, a, predictor, venv_norm, path):
    """One deterministic episode; frames = [rendered scene | current 120px | diffusion subgoal]."""
    import imageio
    from dream_on.common import preprocess_frame
    env = make_env_fn(a, "sparse", predictor)()
    obs, _ = env.reset(seed=a.seed + 7)
    frames, done = [], False
    inner = env.env  # DreamOnWrapper
    while not done:
        o = venv_norm.normalize_obs(obs[None])
        act, _ = model.predict(o, deterministic=True)
        obs, r, term, trunc, info = env.step(act[0])
        done = term or trunc
        rgb = inner.env.render()
        cur = preprocess_frame(rgb)
        panels = [np.asarray(rgb[::4, ::4])]
        to_rgb = lambda t: np.repeat(((t[0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)[..., None], 3, -1)
        panels.append(to_rgb(cur))
        if a.image == "diffusion":
            panels.append(to_rgb(inner.goal_image()))
        frames.append(np.concatenate(panels, axis=1))
    imageio.mimsave(path, frames, duration=1 / 15, loop=0)
    env.close()
    return float(info.get("is_success", 0.0))


def main():
    a = parse()
    if a.reward != "sparse" and a.image != "diffusion":
        sys.exit("--reward l2/ssim requires --image diffusion")
    name = f"{a.algo}_{a.image}_{a.reward}" + (f"_lam{a.lam}" if a.reward != "sparse" else "") \
        + (f"_{a.tag}" if a.tag else "") + f"_s{a.seed}"
    run_dir = Path(a.logdir) / name
    if (run_dir / "DONE").exists():
        print(f"[skip] {name} already finished")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    predictor = None
    if a.image == "diffusion":
        predictor = SubgoalPredictor(a.ckpt, ddim_steps=a.ddim_steps, device=a.device, seed=a.seed)
        print(f"Diffusion sampler on {predictor.device}, DDIM steps={a.ddim_steps}, "
              f"trained k={predictor.k_step_future}, goal_every={a.goal_every}")

    env = DummyVecEnv([make_env_fn(a, a.reward, predictor, str(run_dir / "train"))])
    env.seed(a.seed)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    eval_env = DummyVecEnv([make_env_fn(a, "sparse", predictor, str(run_dir / "eval"))])
    eval_env.seed(a.seed + 1000)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    common = dict(verbose=1, seed=a.seed, tensorboard_log=str(run_dir), learning_rate=3e-4, gamma=0.99)
    if a.algo == "ppo":
        model = PPO("MlpPolicy", env, n_steps=a.n_steps, batch_size=64, n_epochs=10, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), **common)
    else:
        model = SAC("MlpPolicy", env, buffer_size=min(1_000_000, a.timesteps), learning_starts=10_000,
                    batch_size=256, tau=0.005, train_freq=1, gradient_steps=1, ent_coef="auto",
                    policy_kwargs=dict(net_arch=[256, 256]), **common)

    eval_cb = EvalCallback(eval_env, n_eval_episodes=a.eval_episodes, eval_freq=a.eval_freq,
                           log_path=str(run_dir), best_model_save_path=str(run_dir / "best_model"),
                           deterministic=True, render=False)
    t0 = time.time()
    model.learn(total_timesteps=a.timesteps, callback=eval_cb, tb_log_name="tb")
    wall = time.time() - t0
    model.save(run_dir / "final_model")
    env.save(str(run_dir / "vec_normalize.pkl"))

    final_success = None
    if not a.no_gif:
        try:
            final_success = save_rollout_gif(model, a, predictor, env, run_dir / "rollout.gif")
        except Exception as e:  # GIF is a nice-to-have, never fail the run on it
            print(f"GIF failed: {e}")
    meta = dict(vars(a), run=name, wall_clock_sec=round(wall, 1), steps_per_sec=round(a.timesteps / wall, 1),
                platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                gif_episode_success=final_success)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "DONE").write_text("ok\n")
    env.close()
    eval_env.close()
    print(f"[done] {name} in {wall/60:.1f} min")


if __name__ == "__main__":
    main()
