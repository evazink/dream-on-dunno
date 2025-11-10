import numpy as np
import torch
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state, action, reward, next_state, goal, done):
        experience = (state, action, reward, next_state, goal, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, goals, dones = zip(*[self.buffer[idx] for idx in indices])
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(goals),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self, max_size=1000000, k_future=4):
        super().__init__(max_size)
        self.k_future = k_future  # Number of future states to use as goals
        
    def add_episode(self, states, actions, rewards, next_states, goals, dones):
        """
        Add an entire episode to the buffer, including hindsight goals
        """
        episode_length = len(states)
        
        # Add original experience
        for t in range(episode_length):
            self.add(states[t], actions[t], rewards[t], next_states[t], goals[t], dones[t])
        
        # Add hindsight experience
        for t in range(episode_length):
            # Sample k_future future states as goals
            future_indices = np.random.choice(
                np.arange(t + 1, episode_length),
                size=min(self.k_future, episode_length - t - 1),
                replace=False
            )
            
            for future_idx in future_indices:
                # Use future state as goal
                hindsight_goal = next_states[future_idx]
                
                # Calculate new reward based on hindsight goal
                hindsight_reward = self.compute_reward(next_states[t], hindsight_goal)
                
                # Add hindsight experience
                self.add(states[t], actions[t], hindsight_reward, next_states[t], hindsight_goal, dones[t])
    
    def compute_reward(self, achieved_goal, desired_goal):
        # Reward is negative L2 distance between current and goal image.
        achieved_goal_f = achieved_goal.astype(np.float32)
        desired_goal_f = desired_goal.astype(np.float32)

        diff = achieved_goal_f - desired_goal_f
        
        # Flatten and compute L2 norm. This handles both single and batch images.
        norm_diff = np.linalg.norm(diff.reshape(diff.shape[0], -1) if diff.ndim == 4 else diff.flatten())
        
        # Simple normalization by maximum pixel value.
        reward = (-norm_diff / 255.0 + 70) * 0.1
        
        return float(reward) if np.isscalar(reward) else reward.astype(np.float64)