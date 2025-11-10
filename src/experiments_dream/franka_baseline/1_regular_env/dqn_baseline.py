import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
from stable_baselines3 import DQN
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
run_name = f"dqn_franka_{timestamp}"

class DiscretizeActionWrapper(gym.Wrapper):
    """
    A wrapper that discretizes continuous actions into a discrete action space.
    Each continuous action dimension is discretized into n_bins bins.
    """
    def __init__(self, env, n_bins=5):
        super().__init__(env)
        self.n_bins = n_bins
        self.action_space = env.action_space
        self.n_actions = n_bins ** self.action_space.shape[0]
        self.action_space = gym.spaces.Discrete(self.n_actions)
        
        # Create action mapping
        self.action_mapping = self._create_action_mapping()
        
    def _create_action_mapping(self):
        # Create a grid of actions
        action_ranges = []
        for i in range(self.action_space.shape[0]):
            action_ranges.append(np.linspace(
                self.action_space.low[i],
                self.action_space.high[i],
                self.n_bins
            ))
        
        # Create all possible combinations
        actions = []
        for i in range(self.n_actions):
            # Convert index to base-n_bins number
            digits = []
            temp = i
            for _ in range(self.action_space.shape[0]):
                digits.append(temp % self.n_bins)
                temp //= self.n_bins
            
            # Map to actual action values
            action = np.array([action_ranges[j][digits[j]] for j in range(len(digits))])
            actions.append(action)
            
        return np.array(actions)
    
    def step(self, action):
        # Convert discrete action to continuous
        continuous_action = self.action_mapping[action]
        return self.env.step(continuous_action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

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
    env = DiscretizeActionWrapper(env, n_bins=5)  # Discretize actions into 5 bins per dimension
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

# Initialize the DQN model
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=f"{log_dir}/{run_name}",
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=500,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(
        net_arch=[256, 256]
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