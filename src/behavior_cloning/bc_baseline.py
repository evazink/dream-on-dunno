import minari
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tqdm import tqdm # For progress bars

# --- 1. Load the Minari Dataset and Environment ---
print("Loading Minari dataset 'D4RL/kitchen/mixed-v2'...")
dataset = minari.load_dataset('D4RL/kitchen/mixed-v2')
env = dataset.recover_environment(render_mode='rgb_array') # We won't be rendering during training, but good to have.
dataset.set_seed(seed=1)
print("Dataset loaded successfully.")

# --- 2. Extract Expert Data (States and Actions) ---
print("Extracting states and actions from the dataset...")
all_expert_states = []
all_expert_actions = []

# Iterate through all episodes in the dataset
for episode in tqdm(dataset.iterate_episodes(), desc="Processing episodes", total=len(dataset)):
    # Each episode object has .observations and .actions attributes
    all_expert_states.extend(episode.observations['observation'][:-1])
    all_expert_actions.extend(episode.actions)

# Convert lists to NumPy arrays
expert_states = np.array(all_expert_states)
expert_actions = np.array(all_expert_actions)

print(f"Total expert samples extracted: {len(expert_states)}")
print(f"Expert states shape: {expert_states.shape}")
print(f"Expert actions shape: {expert_actions.shape}")

# --- 3. Split Data into Training and Validation Sets ---
X_train, X_val, y_train, y_val = train_test_split(
    expert_states, expert_actions, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Get dimensions from the environment (or dataset)
state_dim = env.observation_space['observation'].shape[0]
action_dim = env.action_space.shape[0]
print(f"Observation space dimension: {state_dim}")
print(f"Action space dimension: {action_dim}")

# --- 4. Define the Behavior Cloning Model Architecture ---
print("Defining the BC model architecture...")
model = keras.Sequential([
    keras.layers.Input(shape=(state_dim,)),
    keras.layers.Dense(256, activation='relu'), # More neurons for complex tasks
    keras.layers.Dropout(0.2), # Dropout to prevent overfitting
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(action_dim, activation='tanh') # Actions are continuous, typically normalized between -1 and 1
])

# Kitchen actions are typically within [-1, 1] range, so tanh is suitable.
# If your environment's actions are in a different range, you might need to scale them
# or use a different activation + scaling layer.

model.summary()

# --- 5. Compile the Model ---
print("Compiling the model...")
initial_learning_rate = 1e-3
lr_schedule_exp = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,  # Number of steps after which learning rate decays
    decay_rate=0.9,     # Factor by which the learning rate will be multiplied
    staircase=True)     # If True, applies decay at discrete intervals

model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule_exp), # Start with a small learning rate
              loss='mse') # Mean Squared Error for continuous actions

# --- 6. Train the Model ---
print("Training the Behavior Cloning model...")
history = model.fit(X_train, y_train,
                    epochs=10,        # Number of training epochs (adjust as needed)
                    batch_size=256,   # Batch size (adjust as needed)
                    validation_data=(X_val, y_val),
                    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]) # Early stopping

print("Model training complete.")

# --- 7. Evaluate the Trained Agent in the Environment ---
print("\nEvaluating the trained agent in the environment...")
num_eval_episodes = 3 # Number of episodes to evaluate
max_steps_per_episode = 100 # Your specified 10-step horizon for evaluation

episode_rewards = []
expert_rewards_total = []

# Optional: Get expert rewards from the dataset for comparison
print("Calculating expert average rewards from dataset...")
for episode in tqdm(dataset.iterate_episodes(), desc="Calculating expert rewards", total=len(dataset)):
    expert_rewards_total.append(np.sum(episode.rewards))
print(f"Dataset Expert Average Reward: {np.mean(expert_rewards_total):.2f}")


for i in tqdm(range(num_eval_episodes), desc="Evaluating agent"):
    obs, info = env.reset()
    total_reward = 0
    
    for t in range(max_steps_per_episode):
        # Predict action using the trained BC model
        # Ensure the observation is in the correct shape (batch_size, state_dim)
        obs_reshaped = obs['observation']
        # print('obs_reshaped.shape (before expand_dims): ', obs_reshaped.shape)

        obs_input_to_model = np.expand_dims(obs_reshaped, axis=0) # This makes it (1, state_dim)
        # print('obs_input_to_model.shape (after expand_dims): ', obs_input_to_model.shape)

        assert len(obs_reshaped.shape) == 1 and obs_reshaped.shape[0] == state_dim, \
            f"Observed state shape from env.reset(): {obs_reshaped.shape}, expected (state_dim,)"

        predicted_action = model.predict(obs_input_to_model, verbose=0)[0] # [0] to get rid of the batch dimension from the output
        assert predicted_action.shape[0] == action_dim, f"Predicted action shape: {predicted_action.shape}, action_dim: {action_dim}"

        # Take the predicted action in the environment
        obs, reward, terminated, truncated, info = env.step(predicted_action)
        total_reward += reward

        # If the environment terminates (e.g., goal reached or failure), stop the episode.
        # However, for a fixed 10-step horizon, we just let it run.
        if terminated or truncated:
            break
            
    episode_rewards.append(total_reward)

env.close() # Close the environment after evaluation

average_reward = np.mean(episode_rewards)
std_reward = np.std(episode_rewards)

print(f"\nEvaluation complete over {num_eval_episodes} episodes, {max_steps_per_episode} steps per episode.")
print(f"Agent Average Reward: {average_reward:.2f} +/- {std_reward:.2f}")

# You can save your trained model if you want
model.save('bc_kitchen_agent.keras')
print("Model saved as 'bc_kitchen_agent.keras'")