import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import imageio
import cv2

# Register the environment
gym.register_envs(gymnasium_robotics)

# Create logs directory
log_dir = "0_logs/"
os.makedirs(log_dir, exist_ok=True)

# Create a timestamp for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"ppo_franka_{timestamp}"

class SaveVideoCallback(BaseCallback):
    """
    Callback for saving a rollout as a GIF.
    """
    def __init__(self, eval_env, save_freq=10000, n_eval_episodes=1, deterministic=True):
        super().__init__()
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.video_dir = os.path.join(log_dir, run_name, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Save a rollout
            frames = []
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, _, done, _ = self.eval_env.step(action)
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)
            
            # Save as GIF
            video_path = os.path.join(self.video_dir, f"rollout_{self.n_calls}.gif")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved rollout to {video_path}")
        return True

class FlattenObservationWrapper(gym.Wrapper):
    """
    A wrapper that flattens dictionary observations into a single array.
    """
    def __init__(self, env):
        super().__init__(env)
        # Get the observation space
        obs_space = env.observation_space
        if isinstance(obs_space, gym.spaces.Dict):
            # Calculate total dimension recursively
            total_dim = self._get_space_dim(obs_space)
            
            # Create new observation space
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
            )
        else:
            self.observation_space = obs_space

    def _get_space_dim(self, space):
        if isinstance(space, gym.spaces.Box):
            return np.prod(space.shape)
        elif isinstance(space, gym.spaces.Discrete):
            return 1
        elif isinstance(space, gym.spaces.Dict):
            return sum(self._get_space_dim(subspace) for subspace in space.values())
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

    def _flatten_obs(self, obs):
        if isinstance(obs, dict):
            # Recursively flatten nested dictionaries
            flattened = []
            for key in sorted(obs.keys()):
                if isinstance(obs[key], dict):
                    flattened.append(self._flatten_obs(obs[key]))
                elif isinstance(obs[key], np.ndarray):
                    flattened.append(obs[key].flatten())
                elif isinstance(obs[key], (int, float)):
                    flattened.append(np.array([obs[key]]))
                else:
                    raise ValueError(f"Unsupported observation type: {type(obs[key])}")
            return np.concatenate(flattened)
        return obs

# Create the environment
def make_env():
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='rgb_array', max_episode_steps=50)
    env = FlattenObservationWrapper(env)
    env = Monitor(env, log_dir)
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# Create evaluation environment
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)

# Create evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{log_dir}/{run_name}/best_model",
    log_path=f"{log_dir}/{run_name}/eval_logs",
    eval_freq=1000,
    deterministic=True,
    render=False
)

# Create video saving callback
video_callback = SaveVideoCallback(
    eval_env,
    save_freq=10000,  # Save a video every 10k steps
    n_eval_episodes=1,
    deterministic=True
)

# Initialize the PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=f"{log_dir}/{run_name}",
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    policy_kwargs=dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )
)

# Train the model
total_timesteps = 500_000
model.learn(
    total_timesteps=total_timesteps,
    callback=[eval_callback, video_callback],
    tb_log_name=run_name
)

# Save the final model and normalization stats
model.save(f"{log_dir}/{run_name}/final_model")
env.save(f"{log_dir}/{run_name}/vec_normalize.pkl")

# Cleanup
env.close()
eval_env.close()