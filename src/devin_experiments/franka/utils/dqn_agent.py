import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from utils.qnet import QNetwork, ReplayBuffer

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, obs_size, action_size, learning_rate, gamma,
                 epsilon_start, epsilon_end, epsilon_decay,
                 replay_buffer_capacity, batch_size, target_update_freq,
                 discrete_actions, device): # Added 'device' parameter
        self.device = device # Store the device

        self.q_network = QNetwork(obs_size, action_size).to(device) # No need to pass device inside QNetwork's init if it's handled here
        self.target_q_network = QNetwork(obs_size, action_size).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_capacity) # Assuming ReplayBuffer can handle device

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.num_discrete_actions = action_size
        self.discrete_actions = discrete_actions

    def choose_action(self, observation):
        if random.random() < self.epsilon:
            return random.randrange(self.num_discrete_actions)
        else:
            with torch.no_grad():
                obs_tensor = observation.float().unsqueeze(0).to(self.device)
                self.q_network.eval()
                q_values = self.q_network(obs_tensor)
                self.q_network.train()
                return q_values.argmax().item()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        current_q_values = self.q_network(states).gather(1, actions)

        next_q_values = self.target_q_network(next_states).max(1)[0].detach()

        target_q_values = rewards + self.gamma * next_q_values.unsqueeze(1) * (1 - dones)

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping (optional but recommended for stability)
        for param in self.q_network.parameters():
            if param.grad is not None: # Check if gradients exist
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    # --- New Methods for Saving and Loading ---
    def save_model(self, path):
        """Saves the state dictionaries of the Q-network, target Q-network, optimizer, and epsilon."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Loads the state dictionaries into the Q-network, target Q-network, optimizer, and restores epsilon."""
        # Ensure map_location is specified to correctly load model on current device
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")