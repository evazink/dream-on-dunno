import gymnasium as gym
import gymnasium_robotics
import torch
import numpy as np
from utils.qnet import QNetwork, ReplayBuffer
from utils.dqn_agent import DQNAgent
from utils.custom_franka_env import CustomFrankaEnv
import os
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter
import datetime # For creating unique log directories
import argparse

# Register the environment
gym.register_envs(gymnasium_robotics)

# --- Argument Parser ---
parser = argparse.ArgumentParser(description="Train a DQN agent for the FrankaKitchen environment.")
parser.add_argument("--model_checkpoint_path", type=str, required=True, help="Path to the .pth model checkpoint file for the diffusion model.")
args = parser.parse_args()

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device for training.")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU for training.")


# --- Hyperparameters ---
LEARNING_RATE = 0.001
GAMMA = 0.99
EPSILON_START = 0.5
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 100
# Define the path where your model will be saved
MODEL_SAVE_PATH = "dqn_franka_kitchen_model.pth"
# Define the TensorBoard log directory
LOG_DIR_BASE = "runs/dqn_franka_kitchen" # Base directory for TensorBoard logs

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

env = CustomFrankaEnv(
    gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle']),
    model_checkpoint_path=args.model_checkpoint_path
)
observation_space_size = env.observation_space['observation'].shape[0]

# --- Initialize Agent ---
agent = DQNAgent(
    obs_size=observation_space_size,
    action_size=NUM_DISCRETE_ACTIONS,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
    epsilon_start=EPSILON_START,
    epsilon_end=EPSILON_END,
    epsilon_decay=EPSILON_DECAY,
    replay_buffer_capacity=REPLAY_BUFFER_SIZE,
    batch_size=BATCH_SIZE,
    target_update_freq=TARGET_UPDATE_FREQ,
    discrete_actions=DISCRETE_ACTIONS,
    device=device
)

# --- TensorBoard Setup ---
# Create a unique run directory for each training session (or for resuming)
# This helps in comparing different runs and avoiding overwriting logs.
if os.path.exists(MODEL_SAVE_PATH):
    # If resuming, try to find the latest log directory or create a new one with a suffix
    # A simple approach for resuming: create a new log directory with a timestamp
    # This keeps previous logs intact and starts fresh for the resumed training.
    # Alternatively, you could try to load the exact previous log directory,
    # but that requires saving the log directory name with the model, which adds complexity.
    # For now, starting a new log directory for resumed training is simpler and safe.
    log_dir = os.path.join(LOG_DIR_BASE, "resume_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
else:
    # If starting from scratch, create a new log directory with a timestamp
    log_dir = os.path.join(LOG_DIR_BASE, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to: {log_dir}")


# --- Load Model (if it exists) ---
if os.path.exists(MODEL_SAVE_PATH):
    agent.load_model(MODEL_SAVE_PATH)
    agent.epsilon = EPSILON_START
    print(f"Resuming training from {MODEL_SAVE_PATH} with epsilon: {agent.epsilon:.4f}")
else:
    print("No saved model found, starting training from scratch.")

print(f"Starting training for {NUM_EPISODES} episodes...")

# --- Training Loop ---
for episode in range(NUM_EPISODES):
    observation, info = env.reset()
    observation_tensor = torch.from_numpy(observation['observation']).float().to(device)

    assert observation['observation'].size == observation_space_size, f"Observation size mismatch, expected {observation_space_size}, got {observation.size}"
    episode_reward = 0
    done = False
    truncated = False
    step_count = 0

    # while not done and not truncated and step_count < MAX_STEPS_PER_EPISODE:
    while step_count < MAX_STEPS_PER_EPISODE:
        discrete_action_idx = agent.choose_action(observation_tensor)
        continuous_action = DISCRETE_ACTIONS[discrete_action_idx]

        next_observation, reward, done, truncated, info = env.step(continuous_action)
        next_observation_tensor = torch.from_numpy(next_observation['observation']).float().to(device)

        agent.replay_buffer.push(
            observation_tensor.cpu().numpy(),
            discrete_action_idx,
            reward,
            next_observation_tensor.cpu().numpy(),
            done
        )

        observation = next_observation
        observation_tensor = next_observation_tensor
        episode_reward += reward
        step_count += 1

        loss = agent.learn() # Capture the loss returned by agent.learn()

        if loss is not None: # Only log loss if it's not None (i.e., if learning occurred)
            writer.add_scalar('Loss/DQN_Loss', loss, global_step=episode * MAX_STEPS_PER_EPISODE + step_count)


        if step_count % agent.target_update_freq == 0:
            agent.update_target_network()

    agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)

    # --- TensorBoard Logging per episode ---
    writer.add_scalar('Episode/Reward', episode_reward, global_step=episode)
    writer.add_scalar('Episode/Epsilon', agent.epsilon, global_step=episode)
    writer.add_scalar('Episode/Steps_Taken', step_count, global_step=episode)


    print(f"Episode {episode + 1}/{NUM_EPISODES}, Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Steps: {step_count}")

    # --- Save Model Periodically ---
    # Save the model every 50 episodes or at the end of training
    if (episode + 1) % 50 == 0 or (episode + 1) == NUM_EPISODES:
        agent.save_model(MODEL_SAVE_PATH)
        print(f"Model saved after episode {episode + 1}")

# Close the environment after training
env.close()
writer.close() # Close the TensorBoard writer
print("Training complete!")


# --- Visualization after training ---
print("\nStarting visualization of trained agent...")
env = CustomFrankaEnv(
    gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='human'),
    model_checkpoint_path=args.model_checkpoint_path
)

# Create a *new* agent instance for visualization
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

    # while not done and not truncated and step_count < MAX_STEPS_PER_EPISODE:
    while step_count < MAX_STEPS_PER_EPISODE:
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