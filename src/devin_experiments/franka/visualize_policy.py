import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
from utils.qnet import QNetwork, ReplayBuffer # Import QNetwork from q_network.py
from utils.dqn_agent import DQNAgent # Import DQNAgent from dqn_agent.py
from utils.custom_franka_env import CustomFrankaEnv # Import the custom wrapper for FrankaKitchen-v1
import os # Import os for path manipulation
import argparse

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Visualize a trained DQN agent for the FrankaKitchen environment.")
parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the .pth model checkpoint file for the diffusion model.")
args = parser.parse_args()

# --- Visualization after training ---
print("\nStarting visualization of trained agent...")
env = CustomFrankaEnv(
    gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='human'),
    model_checkpoint_path=args.model_checkpoint_path
)

# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 100
# Define the path where your model will be saved
MODEL_SAVE_PATH = "dqn_franka_kitchen_model.pth"

# --- Environment Setup ---
DISCRETE_ACTIONS = [
    np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32),
    np.array([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5], dtype=np.float32),
]
NUM_DISCRETE_ACTIONS = len(DISCRETE_ACTIONS)
observation_space_size = env.observation_space['observation'].shape[0]

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for training.")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU for training.")

# Create a *new* agent instance for visualization
# This ensures you're loading the saved model and not accidentally using
# the agent that just finished training with its potentially higher epsilon.
visualization_agent = DQNAgent(
    obs_size=observation_space_size,
    action_size=NUM_DISCRETE_ACTIONS,
    learning_rate=LEARNING_RATE, # These don't affect inference but are needed for init
    gamma=GAMMA,
    epsilon_start=0.0, # Set epsilon to 0 for greedy action selection during visualization
    epsilon_end=0.0,
    epsilon_decay=0.0,
    replay_buffer_capacity=1, # Replay buffer not needed for inference
    batch_size=1,
    target_update_freq=1,
    discrete_actions=DISCRETE_ACTIONS,
    device=device
)

# Load the trained model for visualization
if os.path.exists(MODEL_SAVE_PATH):
    visualization_agent.load_model(MODEL_SAVE_PATH)
    # Ensure epsilon is 0 for greedy visualization, even if loaded epsilon is higher
    visualization_agent.epsilon = 0.0
else:
    print(f"Error: No saved model found at {MODEL_SAVE_PATH} for visualization.")
    env.close()
    exit() # Exit if no model to visualize

for episode in range(5):
    observation, info = env.reset()
    done = False
    truncated = False
    step_count = 0

    while not done and not truncated and step_count < MAX_STEPS_PER_EPISODE:
        # Choose action using the loaded visualization agent's greedy policy
        observation_tensor = torch.from_numpy(observation['observation']).float().to(device)
        discrete_action_idx = visualization_agent.choose_action(observation_tensor)
        continuous_action = DISCRETE_ACTIONS[discrete_action_idx]

        next_observation, reward, done, truncated, info = env.step(continuous_action)
        observation = next_observation

        env.render()
        step_count += 1

    print(f"Visualization Episode {episode + 1}/5 completed in {step_count} steps.")

env.close()
print("Visualization complete!")