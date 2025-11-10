import minari
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time # For pauses between steps
import imageio # For saving GIFs
import os

# --- 1. Load the Trained Model ---
print("Loading the trained Behavior Cloning model 'bc_kitchen_agent.keras'...")
try:
    model = keras.models.load_model('bc_kitchen_agent.keras')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'bc_kitchen_agent.keras' exists in the same directory.")
    exit()

# --- 2. Initialize the Minari Environment for Rendering ---
print("Initializing Minari environment for rendering...")
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
env = dataset.recover_environment(render_mode='rgb_array')  # Changed to rgb_array for frame capture

# Get state and action dimensions from the environment for correct input shaping
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]
print(f"Observation space dimension: {state_dim}")
print(f"Action space dimension: {action_dim}")

# --- 3. Run and Render Evaluation Episodes ---
num_render_episodes = 3
max_steps_per_episode = 100
render_delay = 0.05

# Create directory for saving frames if it doesn't exist
os.makedirs('trajectory_images', exist_ok=True)

print(f"\nStarting rendering for {num_render_episodes} episodes, {max_steps_per_episode} steps per episode...")

for i in range(num_render_episodes):
    obs, info = env.reset()
    total_reward = 0
    print(f"\n--- Rendering Episode {i+1}/{num_render_episodes} ---")
    
    # List to store frames for this episode
    frames = []

    for t in range(max_steps_per_episode):
        # Ensure the observation is in the correct shape (batch_size, state_dim)
        obs_reshaped = obs['observation']
        obs_input_to_model = np.expand_dims(obs_reshaped, axis=0)

        # Predict action using the loaded BC model
        predicted_action = model.predict(obs_input_to_model, verbose=0)[0]

        # Take the predicted action in the environment
        obs, reward, terminated, truncated, info = env.step(predicted_action)
        total_reward += reward

        # Capture the frame
        frame = env.render()
        frames.append(frame)
        time.sleep(render_delay)

        if terminated or truncated:
            print(f"Episode terminated/truncated at step {t+1}. Total reward: {total_reward:.2f}")
            break
    else:
        print(f"Episode finished max steps ({max_steps_per_episode}). Total reward: {total_reward:.2f}")
    
    # Save the episode as a GIF
    if frames:
        gif_path = f'trajectory_images/episode_{i+1}.gif'
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"Saved episode {i+1} as GIF to {gif_path}")

env.close()
print("\nRendering complete.")