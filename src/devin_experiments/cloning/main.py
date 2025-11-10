import gymnasium as gym
import gymnasium_robotics
import numpy as np
import matplotlib.pyplot as plt
from utils.custom_franka_env import CustomFrankaEnv 
from utils.franka_viewer import FrankaKitchenViewer

# Initialize the environment
env = CustomFrankaEnv(gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='rgb_array'))
viewer = FrankaKitchenViewer(title="Franka Kitchen Environment with Goal Overlay")

# Reset the environment
obs, info = env.reset()

print("Starting simulation...")
for i in range(50): # Increased steps for a longer animation
    action = env.action_space.sample() # Take a random action
    obs, reward, terminated, truncated, info = env.step(action)

    animated_frame = env.render()
    viewer.update(animated_frame)

    print(f"Step {i+1}: Reward = {reward:.2f}")

    if terminated or truncated:
        print("Episode finished.")
        break

env.close()
viewer.close()
print("Environment closed.")