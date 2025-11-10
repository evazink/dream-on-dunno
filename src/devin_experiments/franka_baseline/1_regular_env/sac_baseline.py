import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
from stable_baselines3 import SAC
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
run_name = f"sac_franka_{timestamp}"

class SaveVideoCallback(BaseCallback):
    """
    Callback for saving a rollout as a GIF.
    """
    def __init__(self, eval_env, save_freq=10000, n_eval_episodes=1, deterministic=True, max_steps=100):
        super().__init__()
        self.eval_env = eval_env
        self.save_freq = save_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.max_steps = max_steps
        self.video_dir = os.path.join(log_dir, run_name, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            # Save a rollout
            frames = []
            obs = self.eval_env.reset()
            done = False
            step_count = 0
            
            while not done and step_count < self.max_steps:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, rewards, dones, infos = self.eval_env.step(action)
                done = dones[0]  # Get done flag for first (and only) environment
                frame = self.eval_env.render()
                if frame is not None:
                    frames.append(frame)
                step_count += 1
            
            # Save as GIF
            video_path = os.path.join(self.video_dir, f"rollout_{self.n_calls}.gif")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"Saved rollout to {video_path} (length: {len(frames)} steps)")
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
    deterministic=True,
    max_steps=100  # Run for up to 100 steps in evaluation
)

# Initialize the SAC model
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=f"{log_dir}/{run_name}",
    learning_rate=3e-4,
    buffer_size=1000000,  # Larger buffer for better sample efficiency
    learning_starts=10000,  # Start learning after collecting some samples
    batch_size=256,  # Larger batch size for SAC
    tau=0.005,  # Target network update rate
    gamma=0.99,
    train_freq=1,  # Update every step
    gradient_steps=1,  # Number of gradient steps per update
    action_noise=None,  # SAC uses stochastic policy, no need for action noise
    optimize_memory_usage=False,
    ent_coef='auto',  # Automatically tune entropy coefficient
    target_update_interval=1,  # Update target network every step
    target_entropy='auto',  # Automatically tune target entropy
    use_sde=False,  # Don't use State Dependent Exploration
    sde_sample_freq=-1,
    use_sde_at_warmup=False,
    policy_kwargs=dict(
        net_arch=[256, 256],  # Simpler architecture for SAC
        log_std_init=0.0  # Initialize log_std to 0
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