import minari
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
from rl.helper import get_reward
from copy import deepcopy
import os

# Create output directory for images if it doesn't exist
os.makedirs('trajectory_images', exist_ok=True)

# Load dataset and environment
# https://minari.farama.org/datasets/D4RL/kitchen/mixed-v2/
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
env = dataset.recover_environment(render_mode='rgb_array')
dataset.set_seed(seed=1)
# print("Observation space:", dataset.observation_space)
# print("Action space:", dataset.action_space)
# print("Total episodes:", dataset.total_episodes)
# print("Total steps:", dataset.total_steps)

env  = dataset.recover_environment(render_mode='rgb_array')
goal_env = dataset.recover_environment(render_mode='rgb_array')

# Sample an episode
episode = dataset.sample_episodes(n_episodes=1)[0]
env.reset()

rewards_v1 = []
rewards_v2 = []
goal_update_timesteps = []

for t in range(100):
    # Get current state image
    current_img = Image.fromarray(env.render())

    if t % 20 == 0:
    
        # Get goal state image
        goal_env.reset()
        goal_img = get_goal_img(goal_env, t, episode, num_steps_ahead=20)
        goal_update_timesteps.append(t)
    
    visualization = True
    if visualization == True:
        # Create overlay visualization
        plt.figure(figsize=(12, 6))
        
        # Plot current state as base image
        plt.imshow(current_img)
        
        # Overlay goal state with transparency
        plt.imshow(goal_img, alpha=0.5)
        
        plt.title(f'Current State (t={t}) with Goal Overlay')
        plt.axis('off')
        
        # Save the visualization
        plt.savefig(f'trajectory_images/step_{t:03d}.png')
        plt.close()
    
    # Take step in environment
    obs, _, terminated, truncated, info = env.step(episode.actions[t])
    # obs, _, terminated, truncated, info = env.step(env.action_space.sample())

    rewards_v1.append(get_reward_v1(current_img, goal_img))
    rewards_v2.append(get_reward_v2(current_img, goal_img))
    
    if terminated or truncated:
        break

env.close()
goal_env.close()

# Plot rewards over time
plt.figure(figsize=(12, 6))
plt.plot(range(len(rewards_v1)), rewards_v1, 'b-', label='Reward V1')
plt.plot(range(len(rewards_v2)), rewards_v2, 'r-', label='Reward V2')

# Add vertical lines for goal updates
for t in goal_update_timesteps:
    plt.axvline(x=t, color='g', linestyle='--', alpha=0.3, label='Goal Update' if t == goal_update_timesteps[0] else "")

plt.xlabel('Time Step (t)')
plt.ylabel('Reward')
plt.title('Rewards Over Time')
plt.grid(True)
plt.legend()
plt.savefig('trajectory_images/rewards_over_time.png')
plt.close()


print(f"Trajectory images have been saved in the 'trajectory_images' directory.")
print("You can view them sequentially to see the progression of states and goals.")