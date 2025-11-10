import gymnasium as gym
import gymnasium_robotics
from utils.custom_franka_env import CustomFrankaEnv
from utils.simple_net import SimpleNet
import torch
from PIL import Image

gym.register_envs(gymnasium_robotics)

env = CustomFrankaEnv(gym.make(
    'FrankaKitchen-v1', 
    object_noise_ratio=0.0, 
    robot_noise_ratio=0.0, 
    tasks_to_complete=['kettle'], 
    render_mode='rgb_array',
    max_episode_steps=50,
))

print('observation space: ', env.observation_space)
print('action space: ', env.action_space)

actor = SimpleNet()

obs, info = env.reset()
for _ in range(50):
    # action = actor(torch.tensor(obs))[0]
    action = env.action_space.sample()
    # obs, reward, terminated, truncated, info = env.step(action.detach().numpy())  # Take the action in the environment
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs), 'Observation is not in the observation space'
    # env.render()  # Render the environment

    if terminated:
        break

env.close()