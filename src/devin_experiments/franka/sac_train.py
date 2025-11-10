import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
import os
import datetime
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecVideoRecorder # Import VecVideoRecorder
from utils.custom_franka_env import CustomFrankaEnv
import imageio # Import imageio for GIF creation

# Register the environment
gym.register_envs(gymnasium_robotics)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train an SAC agent for the FrankaKitchen environment.")
parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the .pth model checkpoint file for the diffusion model.")
args = parser.parse_args()

# --- Device Setup ---
# Stable Baselines3 automatically handles device selection based on availability.
# However, you can specify it if you want:
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")

# --- Hyperparameters ---
LEARNING_RATE = 0.0003 # Common learning rate for SAC
GAMMA = 0.99
BATCH_SIZE = 256 # Common batch size for SAC
BUFFER_SIZE = 1_000_000 # Increased buffer size for SAC
TRAIN_TIMESTEPS = 70_000 # Total timesteps for training
EVAL_FREQ = 10000 # Evaluate every N timesteps
SAVE_FREQ = 35_000 # Save model every N timesteps
NUM_VIDEO_ROLLOUTS = 2 # Number of rollout trajectories to save as GIFs

# Define the path where your model will be saved
MODEL_SAVE_PATH = "sac_franka_kitchen_model" # Stable Baselines3 saves as a folder
LOG_DIR_BASE = "runs/sac_franka_kitchen"
VIDEO_SAVE_PATH = "rollout_videos" # Directory to save rollout GIFs

# Create video save path if it doesn't exist
os.makedirs(VIDEO_SAVE_PATH, exist_ok=True)


# --- Environment Setup ---
# Stable Baselines3's SAC expects a continuous action space.
# The `FrankaKitchen-v1` environment by default has a Box action space,
# which is continuous. You don't need DISCRETE_ACTIONS here.

# Define a function to create the environment, essential for make_vec_env and EvalCallback
def make_env(model_checkpoint_path, render_mode=None):
    # Make sure to set render_mode='rgb_array' for training if you want to record videos later,
    # and 'human' for visualization.
    env = CustomFrankaEnv(
        gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode=render_mode),
        model_checkpoint_path=model_checkpoint_path
    )
    # Wrap the environment with Monitor to record rewards and episode lengths
    env = Monitor(env) # Monitor should typically be the outermost wrapper after gym.make()
    return env

# Create a vectorized environment for training
# VecNormalize is often used with continuous action spaces to normalize observations
# and sometimes rewards, which can significantly improve performance.
train_env = make_vec_env(lambda: make_env(model_checkpoint_path=args.model_checkpoint_path, render_mode='rgb_array'), n_envs=1)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False) # Normalize observations

# Create a separate environment for evaluation with rgb_array for video recording
eval_env = make_vec_env(lambda: make_env(model_checkpoint_path=args.model_checkpoint_path, render_mode='rgb_array'), n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False) # Use the same normalization as train_env

# --- TensorBoard Setup ---
log_dir = os.path.join(LOG_DIR_BASE, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
print(f"TensorBoard logs will be saved to: {log_dir}")

# --- Initialize SAC Model ---
# SAC is a good choice for continuous action spaces.
# policy="MlpPolicy" means a multi-layer perceptron policy.
# You can customize the policy network architecture using policy_kwargs if needed.
model = SAC(
    "MlpPolicy",
    train_env,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    buffer_size=BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    verbose=1, # Prints training progress
    tensorboard_log=log_dir,
    # device=device # Uncomment if you want to explicitly set device
)

# --- Load Model (if it exists) ---
# Stable Baselines3 saves models in a directory, not a single .pth file for complex algorithms like SAC.
# We'll save the model within the MODEL_SAVE_PATH directory.
if os.path.exists(os.path.join(MODEL_SAVE_PATH, "best_model.zip")):
    print(f"Loading pre-trained model from {MODEL_SAVE_PATH}/best_model.zip")
    # Load the model
    model = SAC.load(os.path.join(MODEL_SAVE_PATH, "best_model.zip"), env=train_env)

    # Load VecNormalize statistics
    vec_normalize_path = os.path.join(MODEL_SAVE_PATH, "vec_normalize.pkl")
    if os.path.exists(vec_normalize_path):
        train_env = VecNormalize.load(vec_normalize_path, train_env)
        eval_env = VecNormalize.load(vec_normalize_path, eval_env) # Also load for eval_env
        # Ensure the loaded envs are wrapped in the model
        model.set_env(train_env)
        print("VecNormalize stats loaded successfully.")
    else:
        print("Warning: VecNormalize stats not found. Environment normalization might be inconsistent.")
    print("Model loaded successfully.")
else:
    print("No saved model found, starting training from scratch.")

# --- Callbacks ---
# CheckpointCallback: Saves the model periodically
checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODEL_SAVE_PATH,
    name_prefix="sac_franka_kitchen"
)

# EvalCallback: Evaluates the agent periodically and saves the best model
# This is crucial for tracking performance and saving the best performing agent.
# For older SB3 versions, use render=True and ensure make_env provides rgb_array.
# Video saving needs to be handled manually or by wrapping the eval_env with VecVideoRecorder.
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=MODEL_SAVE_PATH,
    log_path=log_dir,
    eval_freq=EVAL_FREQ,
    deterministic=True, # Use deterministic actions during evaluation
    render=False # Set to False here, as we'll use VecVideoRecorder explicitly for video
)

# --- Training Loop ---
print(f"Starting training for {TRAIN_TIMESTEPS} timesteps...")
model.learn(
    total_timesteps=TRAIN_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback]
)

# Save the final model
model.save(os.path.join(MODEL_SAVE_PATH, "final_model"))
train_env.save(os.path.join(MODEL_SAVE_PATH, "vec_normalize.pkl")) # Save VecNormalize stats
print("Training complete!")
print(f"Final model and normalization stats saved to: {MODEL_SAVE_PATH}")

print("--- Training Env VecNormalize Stats ---")
print("Observation Mean:", train_env.obs_rms.mean)
print("Observation Variance:", train_env.obs_rms.var)
print("Reward Mean (if norm_reward=True):", train_env.ret_rms.mean) # Only relevant if norm_reward=True

# --- Save Rollout Trajectories as GIFs ---
print(f"\nGenerating {NUM_VIDEO_ROLLOUTS} rollout GIFs from the trained model...")

# Create a separate *vectorized* environment for video rendering
# It's crucial that this environment is also normalized if the model expects normalized observations.
# We'll create a vectorized environment (even if n_envs=1) so it's compatible with VecNormalize and VecVideoRecorder.
video_eval_env = make_vec_env(lambda: make_env(model_checkpoint_path=args.model_checkpoint_path, render_mode='rgb_array'), n_envs=1)
video_eval_env = VecNormalize(video_eval_env, norm_obs=True, norm_reward=False)

# Load VecNormalize statistics for the video environment
vec_normalize_path = os.path.join(MODEL_SAVE_PATH, "vec_normalize.pkl")
if os.path.exists(vec_normalize_path):
    video_eval_env = VecNormalize.load(vec_normalize_path, video_eval_env)
    print("VecNormalize stats loaded for video environment.")
else:
    print("Warning: VecNormalize stats not found for video environment. Video might not reflect trained behavior accurately.")

# Wrap the video environment with VecVideoRecorder
# This will save MP4 files to the specified directory
video_env_recorder = VecVideoRecorder(
    video_eval_env, # Pass the VecNormalize-wrapped vectorized environment
    VIDEO_SAVE_PATH,
    record_video_trigger=lambda x: x == 0,  # Record only the first episode, we'll loop manually
    video_length=500  # Max length of the video in steps
)

for i in range(NUM_VIDEO_ROLLOUTS):
    # Reset the video recorder for each episode
    obs = video_env_recorder.reset() # reset returns (observations, info) tuple in SB3 v2.0+
    # For older SB3 versions, reset() might return just observations.
    # We'll handle both cases to be safe.
    if isinstance(obs, tuple):
        obs, _ = obs # Unpack if it's a tuple (observations, info)
    
    frames = [] # Collect frames for GIF
    done = False
    episode_reward = 0
    # In a vectorized environment, done is an array. We need to check if ANY environment is done.
    # Since n_envs=1, done[0] will be the relevant one.
    
    # Manually render the initial frame *before* the first step, as reset doesn't render.
    # You need to render the underlying unwrapped env to get the frame for GIF.
    # To get the unwrapped env: video_eval_env.venv.envs[0] if video_eval_env is VecNormalize on VecEnv
    # or access through .envs[0] if it's just a VecEnv.
    # Let's be explicit and get the base env from the VecNormalize -> VecVideoRecorder stack.
    # The structure is VecVideoRecorder -> VecNormalize -> DummyVecEnv -> Monitor -> CustomFrankaEnv
    current_env_for_render = video_env_recorder.venv.unwrapped.envs[0]
    frame = current_env_for_render.render()
    frames.append(frame)

    while not done:
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        # step returns (observations, rewards, dones, infos) for vectorized envs
        obs, reward, done, info = video_env_recorder.step(action)
        
        # 'done' is an array in vectorized environments, check the first element for n_envs=1
        # done = terminated[0] or truncated[0] 
        episode_reward += reward[0] # Reward is also an array

        # Render the current frame for GIF creation
        frame = current_env_for_render.render()
        frames.append(frame)

    # After the episode, the VecVideoRecorder will have saved an MP4.
    # Now, save the collected frames as a GIF.
    gif_filename = os.path.join(VIDEO_SAVE_PATH, f"rollout_trajectory_gif_{i+1}_{datetime.datetime.now().strftime('%H%M%S')}.gif")
    imageio.mimsave(gif_filename, frames, fps=30) # Save as GIF with 30 FPS
    print(f"Saved rollout trajectory {i+1} as GIF to {gif_filename} (Episode Reward: {episode_reward:.2f})")

# Close environments
train_env.close()
eval_env.close()
video_env_recorder.close()
# No need to close video_eval_env or video_env_unwrapped separately, as video_env_recorder.close()
# should handle closing the wrapped environments.