import gymnasium as gym
import gymnasium_robotics
import numpy as np
import os
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env # Import make_vec_env
from gymnasium import spaces

# Assuming utils.custom_franka_env exists and contains CustomFrankaEnv
from utils.custom_franka_env import CustomFrankaEnv


# Register the environment
gym.register_envs(gymnasium_robotics)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Visualize a trained SAC agent for the FrankaKitchen environment.")
parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the .pth model checkpoint file for the diffusion model.")
args = parser.parse_args()


# --- Configuration ---
# Define the path where your model is saved
MODEL_SAVE_PATH = "sac_franka_kitchen_model"

# --- Environment Setup Function (re-defined for this script) ---
# This function now correctly prepares a single environment for make_vec_env
def create_single_franka_env_for_vec(model_checkpoint_path):
    """
    Creates a single FrankaKitchen environment wrapped with CustomFrankaEnv and Monitor,
    configured for human rendering. This function is designed to be passed to make_vec_env.
    """
    base_env = gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='human')
    wrapped_env = CustomFrankaEnv(base_env, model_checkpoint_path=model_checkpoint_path)
    monitored_env = Monitor(wrapped_env)
    return monitored_env

# --- Visualization Script ---
print("\nStarting visualization of trained agent...")

# Use make_vec_env to create the vectorized environment chain for visualization
# This ensures that render_env_normalized is indeed a VecEnv, which VecNormalize expects.
# We set n_envs=1 because we only need one environment for visualization.
render_env_normalized = make_vec_env(lambda: create_single_franka_env_for_vec(args.model_checkpoint_path), n_envs=1)
print(f"CustomFrankaEnv: New observation space shape: {render_env_normalized.observation_space.shape}") # This will print the shape from your CustomFrankaEnv's __init__


# Load the saved VecNormalize statistics for the visualization environment
vec_normalize_path = os.path.join(MODEL_SAVE_PATH, "vec_normalize.pkl")
if os.path.exists(vec_normalize_path):
    print(f"Loading VecNormalize statistics from: {vec_normalize_path}")
    # Load the VecNormalize object itself and assign it back to render_env_normalized
    # This is the most robust way to ensure normalization consistency.
    render_env_normalized = VecNormalize.load(vec_normalize_path, render_env_normalized)
    # Important: Set to False for evaluation/rendering to prevent further updates to normalization stats
    print("--- Visualization Env VecNormalize Stats ---")
    print("Observation Mean:", render_env_normalized.obs_rms.mean)
    print("Observation Variance:", render_env_normalized.obs_rms.var)
    print("Reward Mean (if norm_reward=True):", render_env_normalized.ret_rms.mean) # Only relevant if norm_reward=True
    render_env_normalized.training = False
    render_env_normalized.norm_reward = False # Keep consistent with training setup
else:
    print(f"Error: `vec_normalize.pkl` not found at {MODEL_SAVE_PATH}. Normalization will be off, which may affect performance.")
    # You might want to exit or raise an error here if normalization is critical

# Load the trained model
model_path = os.path.join(MODEL_SAVE_PATH, "best_model.zip")
if not os.path.exists(model_path):
    # Try loading the final model if best_model.zip doesn't exist
    model_path = os.path.join(MODEL_SAVE_PATH, "final_model.zip")

if os.path.exists(model_path):
    print(f"Loading model from: {model_path}")
    # Load the SAC model. Pass the *normalized* environment as 'env' to the loaded model.
    loaded_model = SAC.load(model_path, env=render_env_normalized)
    print("Model loaded successfully.")
else:
    print(f"Error: No trained model found at {MODEL_SAVE_PATH}. Please ensure your training script has run and saved a model.")
    render_env_normalized.close()
    exit()

# --- Run Visualization Episodes ---
NUM_VIZ_EPISODES = 5 # Number of episodes to run for visualization
MAX_VIZ_STEPS_PER_EPISODE = 200 # Maximum steps per episode to prevent endless loops

for episode in range(NUM_VIZ_EPISODES):
    # Reset the normalized environment to get the initial observation
    obs = render_env_normalized.reset() # reset() for VecEnv returns just the obs array
    done = False
    truncated = False
    step_count = 0
    total_reward = 0

    print(f"\n--- Visualization Episode {episode + 1}/{NUM_VIZ_EPISODES} ---")

    while not done and not truncated and step_count < MAX_VIZ_STEPS_PER_EPISODE:
        # Predict action using the loaded model.
        action, _states = loaded_model.predict(obs, deterministic=True)

        # Step the normalized environment
        # VecEnv.step() returns (obs, rewards, dones, infos) for multi-env,
        # but for n_envs=1, it will typically return single numpy arrays for obs and rewards,
        # and boolean arrays for dones and truncated.
        # We need to extract the single values from these arrays.
        obs, rewards, dones, infos = render_env_normalized.step(action)
        # Extract single values for single-environment case
        reward = rewards[0]
        done = dones[0]
        truncated = infos[0].get('truncated', False) # Check truncated from info dictionary

        total_reward += reward

        # Render the environment for visual feedback
        render_env_normalized.render()

        step_count += 1

    print(f"Episode completed. Total Reward: {total_reward:.2f}, Steps: {step_count}")

# Close the environment after visualization
render_env_normalized.close()
print("\nVisualization complete!")