import gymnasium as gym
import gymnasium_robotics
import torch
from PIL import Image

gym.register_envs(gymnasium_robotics)

env = gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='human', max_episode_steps=50)

print('observation space: ', env.observation_space)
print('action space: ', env.action_space)

obs, info = env.reset()
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(obs), 'Observation is not in the observation space'
    env.render()  # Render the environment

    if terminated:
        break

env.close()