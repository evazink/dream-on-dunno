import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image
import matplotlib.pyplot as plt
import torch # Required for tensor operations
from torch.utils.tensorboard import SummaryWriter
from utils.custom_franka_env import CustomFrankaEnv # Import your custom environment
from utils.ac_agent import ACAgent # Import your agent
from utils.franka_viewer import FrankaKitchenViewer

def train_agent(env, num_episodes=30, gamma=0.5, actor_lr=1e-4, critic_lr=1e-3, log_dir='runs/franka_training/6'):
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize viewer
    # viewer = FrankaKitchenViewer(title="Franka Kitchen Training")
    
    obs_shape = env.observation_space["observation"].shape
    action_dim = env.action_space.shape[0]
    action_space_high = env.action_space.high[0] # Assuming symmetrical bounds for simplicity

    agent = ACAgent(obs_shape, action_dim, action_space_high, actor_lr, critic_lr, writer)

    episode_rewards = []
    best_reward = float('-inf')

    try:
        for episode in range(num_episodes):
            obs_dict, _ = env.reset()
            state = obs_dict["observation"]
            goal = obs_dict["desired_goal"]
            
            done = False
            rewards_history = []
            log_probs_history = []
            states_history = []
            next_states_history = []
            terminateds_history = []
            goals_history = []
            
            steps = 0
            
            while not done and steps <= 7:
                action, log_prob = agent.select_action(state)
                
                next_obs_dict, reward, terminated, truncated, _ = env.step(action)
                next_state = next_obs_dict["observation"]
                done = terminated or truncated

                rewards_history.append(reward)
                log_probs_history.append(log_prob)
                states_history.append(state)
                next_states_history.append(next_state)
                terminateds_history.append(terminated)
                goals_history.append(goal)

                # Render and display the frame
                # frame = env.render()
                # viewer.update(frame, step=steps, reward=reward)

                steps += 1
                state = next_state

            # Update the agent after collecting an episode
            critic_loss, actor_loss = agent.update(
                states_history, 
                rewards_history, 
                log_probs_history, 
                next_states_history, 
                terminateds_history, 
                gamma
            )

            total_episode_reward = sum(rewards_history)
            episode_rewards.append(total_episode_reward)
            
            # Update best reward
            if total_episode_reward > best_reward:
                best_reward = total_episode_reward
                # Save the best model
                torch.save({
                    'actor_state_dict': agent.actor.state_dict(),
                    'critic_state_dict': agent.critic.state_dict(),
                    'actor_optimizer': agent.actor_optimizer.state_dict(),
                    'critic_optimizer': agent.critic_optimizer.state_dict(),
                }, f'{log_dir}/best_model.pt')
            
            # Log metrics to TensorBoard
            writer.add_scalar('Reward/Episode', total_episode_reward, episode)
            writer.add_scalar('Reward/Best', best_reward, episode)
            writer.add_scalar('Loss/Critic', critic_loss, episode)
            writer.add_scalar('Loss/Actor', actor_loss, episode)
            
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}, Total Reward: {total_episode_reward:.2f}, "
                      f"Best Reward: {best_reward:.2f}, "
                      f"Critic Loss: {critic_loss:.4f}, Actor Loss: {actor_loss:.4f}")
            else:
                print(f"Episode {episode+1}, Total Reward: {total_episode_reward:.2f}")

    finally:
        # Clean up
        writer.close()
        env.close()
        # viewer.close()

    return agent, episode_rewards

if __name__ == '__main__':
    dummy_base_env = gym.make('FrankaKitchen-v1', tasks_to_complete=['kettle'], render_mode='rgb_array')
    env = CustomFrankaEnv(dummy_base_env, goal_step_offset=2)

    print("Starting training...")
    trained_agent, rewards_history = train_agent(env, num_episodes=1000, actor_lr=0.001, critic_lr=0.001)
    print("Training finished.")

    # Plot rewards
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards (Actor-Critic with Hindsight Experience Replay)")
    plt.grid(True)
    plt.show()