import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .replay_buffer import HindsightReplayBuffer

class Actor(nn.Module):
    def __init__(self, observation_shape, action_dim):
        super(Actor, self).__init__()
        # Simplified CNN for image input
        # Input: (Batch, H, W, C) -> permute to (Batch, C, H, W)
        in_channels = observation_shape[2] # Assuming (H, W, C)
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened size after convolutions
        # This requires a dummy forward pass to infer the size
        dummy_input = torch.zeros(1, in_channels, observation_shape[0], observation_shape[1])
        conv_output_size = self._get_conv_output(dummy_input)

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

    def _get_conv_output(self, shape):
        # Helper to calculate output size after convolutions
        o = self.conv1(shape)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        # Normalize and permute for Conv2d (Batch, H, W, C) -> (Batch, C, H, W)
        x = x.permute(0, 3, 1, 2) / 255.0 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2) # Clip for stability
        std = torch.exp(log_std)
        
        return Normal(mean, std)

class Critic(nn.Module):
    def __init__(self, observation_shape):
        super(Critic, self).__init__()
        in_channels = observation_shape[2]

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        dummy_input = torch.zeros(1, in_channels, observation_shape[0], observation_shape[1])
        conv_output_size = self._get_conv_output(dummy_input)

        self.fc1 = nn.Linear(conv_output_size, 256)
        self.value_layer = nn.Linear(256, 1) # Output a single state value

    def _get_conv_output(self, shape):
        o = self.conv1(shape)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2) / 255.0 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.value_layer(x)

class ACAgent:
    def __init__(self, observation_shape, action_dim, action_space_high, actor_lr=1e-4, critic_lr=1e-3, writer=None):
        self.actor = Actor(observation_shape, action_dim)
        self.critic = Critic(observation_shape)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Initialize learning rate schedulers
        self.actor_scheduler = ReduceLROnPlateau(
            self.actor_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        self.critic_scheduler = ReduceLROnPlateau(
            self.critic_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        self.action_space_high = action_space_high
        self.writer = writer
        self.step_count = 0
        
        # Initialize replay buffer
        self.replay_buffer = HindsightReplayBuffer(max_size=1000000, k_future=4)
        self.batch_size = 256

    def select_action(self, state):
        # Make a copy of the state to ensure it has positive strides
        state_copy = state.copy()
        state_tensor = torch.tensor(state_copy, dtype=torch.float32).unsqueeze(0)
        action_dist = self.actor(state_tensor)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action).sum(dim=-1)  # Sum log_probs across action dimensions
        
        action_np = action.squeeze(0).detach().numpy()
        # Clip actions to environment's bounds
        action_np = np.clip(action_np, -self.action_space_high, self.action_space_high)

        return action_np, log_prob

    def store_episode(self, states, actions, rewards, next_states, goals, dones):
        """Store an episode in the replay buffer with hindsight goals"""
        self.replay_buffer.add_episode(states, actions, rewards, next_states, goals, dones)

    def update(self, states, rewards, log_probs, next_states, terminateds, gamma):
        # Store the episode in the replay buffer
        actions = [self.select_action(state)[0] for state in states]
        goals = [next_states[-1]] * len(states)  # Use the final state as the goal
        self.store_episode(states, actions, rewards, next_states, goals, terminateds)

        # Sample a batch from the replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0  # Return dummy losses if buffer is not full enough

        batch_states, batch_actions, batch_rewards, batch_next_states, batch_goals, batch_dones = \
            self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states_tensor = torch.tensor(batch_states, dtype=torch.float32)
        rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32).unsqueeze(1)
        next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32)
        terminateds_tensor = torch.tensor(batch_dones, dtype=torch.float32).unsqueeze(1)

        # Calculate V(s') for next states
        with torch.no_grad():
            next_values = self.critic(next_states_tensor)
            target_values = rewards_tensor + gamma * next_values * (1 - terminateds_tensor)

        # Calculate V(s) for current states
        predicted_values = self.critic(states_tensor)

        # Critic Update
        critic_loss = F.mse_loss(predicted_values, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        action_dist = self.actor(states_tensor)
        actions_tensor = torch.tensor(batch_actions, dtype=torch.float32)
        log_probs = action_dist.log_prob(actions_tensor).sum(dim=-1, keepdim=True)
        advantages = target_values - predicted_values.detach()
        actor_loss = -(log_probs * advantages).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update learning rates based on losses
        self.actor_scheduler.step(actor_loss)
        self.critic_scheduler.step(critic_loss)

        # Log additional metrics if writer is available
        if self.writer is not None:
            self.writer.add_scalar('Metrics/Mean_Value', predicted_values.mean().item(), self.step_count)
            self.writer.add_scalar('Metrics/Mean_Advantage', advantages.mean().item(), self.step_count)
            self.writer.add_scalar('Metrics/Mean_Log_Prob', log_probs.mean().item(), self.step_count)
            self.writer.add_scalar('Metrics/Mean_Reward', rewards_tensor.mean().item(), self.step_count)
            self.writer.add_scalar('Buffer/Size', len(self.replay_buffer), self.step_count)
            
            # Log learning rates
            self.writer.add_scalar('Learning_Rate/Actor', self.actor_optimizer.param_groups[0]['lr'], self.step_count)
            self.writer.add_scalar('Learning_Rate/Critic', self.critic_optimizer.param_groups[0]['lr'], self.step_count)
            
            # Log gradients
            for name, param in self.actor.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/Actor/{name}', param.grad, self.step_count)
            
            for name, param in self.critic.named_parameters():
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/Critic/{name}', param.grad, self.step_count)
            
            self.step_count += 1

        return critic_loss.item(), actor_loss.item()