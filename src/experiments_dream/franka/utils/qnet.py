import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random

# --- Q-Network Definition ---
class QNetwork(nn.Module):
    def __init__(self, obs_size, action_size): # Add 'device' parameter
        super(QNetwork, self).__init__()
        # Define a simple feed-forward neural network
        self.fc1 = nn.Linear(obs_size, 128) # First fully connected layer
        self.relu1 = nn.ReLU()             # ReLU activation function
        self.fc2 = nn.Linear(128, 128)     # Second fully connected layer
        self.relu2 = nn.ReLU()             # ReLU activation function
        self.fc3 = nn.Linear(128, action_size) # Output layer, size equals number of discrete actions

    def forward(self, x):
        # Forward pass through the network
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x) # Output Q-values for each action

# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        # Initialize a deque (double-ended queue) with a maximum capacity
        self.buffer = deque(maxlen=capacity)
        # We don't store the device here if the buffer only handles NumPy arrays.
        # The device is only relevant when converting to tensors in the agent.
        # self.device = device # Remove this line if not needed here

    def push(self, state, action, reward, next_state, done):
        # Add a new experience tuple to the buffer.
        # state and next_state are expected to be NumPy arrays when pushed.
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Randomly sample a batch of experiences from the buffer
        batch = random.sample(self.buffer, batch_size)

        # Unpack the batch into separate lists of states, actions, etc.
        # These will remain as their original types (NumPy arrays for states/next_states,
        # and Python primitives for action, reward, done).
        states, actions, rewards, next_states, dones = zip(*batch)

        # Return them as lists of NumPy arrays or Python lists of primitives.
        # The conversion to PyTorch tensors and moving to the device
        # will now happen in the DQNAgent.learn() method.
        return list(states), list(actions), list(rewards), list(next_states), list(dones)

    def __len__(self):
        # Return the current size of the buffer
        return len(self.buffer)