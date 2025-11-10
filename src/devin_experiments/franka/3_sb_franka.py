from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import gymnasium_robotics
from utils.custom_franka_env import CustomFrankaEnv
import torch
from PIL import Image

from stable_baselines3 import HerReplayBuffer, DQN, PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env

gym.register_envs(gymnasium_robotics)

def make_env():
    return CustomFrankaEnv(gym.make(
        'FrankaKitchen-v1', 
        object_noise_ratio=0.0, 
        robot_noise_ratio=0.0, 
        tasks_to_complete=['kettle'], 
        render_mode='rgb_array'
    ))

check_env(make_env())

env = make_vec_env(
  lambda: make_env(), n_envs=2
)

model = SAC(
   'MultiInputPolicy', 
   env, 
   verbose=1,
   buffer_size=50_000,
).learn(5000)

# Test the trained agent
obs = env.reset()
n_steps = 20
for step in range(n_steps):
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='console')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break