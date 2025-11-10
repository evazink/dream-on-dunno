import gymnasium as gym
import gymnasium_robotics
from stable_baselines3 import PPO
from utils.custom_franka_env import CustomFrankaEnv
from utils.franka_viewer import FrankaKitchenViewer

# Load environment and model
env = CustomFrankaEnv(gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='rgb_array'))
model = PPO.load("ppo_franka_kitchen")

viewer = FrankaKitchenViewer(title="Franka Kitchen Evaluation")
obs, _ = env.reset()

for i in range(50):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    frame = env.render()
    viewer.update(frame)

    print(f"Step {i+1}: Reward = {reward:.2f}")
    if terminated or truncated:
        break

env.close()
viewer.close()
