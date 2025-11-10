import gymnasium as gym
import gymnasium_robotics
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import sys
import os
from src.diffusion.inference import Inference

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

class CustomFrankaEnv(gym.Wrapper):
    """
    A custom wrapper for the FrankaKitchen-v1 environment to:
    1. Define a custom reward function.
    2. Flatten the observation space to be compatible with Stable Baselines3 (SAC).
    """
    def __init__(self, env, model_checkpoint_path: str):
        super().__init__(env)
        self.base_env = env
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low = 0, high = 255, shape = (1, 120, 120), dtype = np.uint8
            ),
            'achieved_goal': spaces.Box(
                low = 0, high = 255, shape = (1, 120, 120), dtype = np.uint8
            ),
            'desired_goal': spaces.Box(
                low = 0, high = 255, shape = (1, 120, 120), dtype = np.uint8
            ),
        })
        self.action_space = self.base_env.action_space
        self.inference = Inference(model_checkpoint_path)

    def get_curr_image(self):
        img = Image.fromarray(self.base_env.render())
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=1)
        img = torchvision.transforms.functional.resize(img, (120, 120))
        return np.array(img, dtype = np.uint8)
    
    def get_goal_image(self):
        # Get the current state image
        current_image = self.get_curr_image() # Shape: (120, 120), dtype: uint8

        # Add a channel dimension to match the model's expected input shape (1, 120, 120)
        current_image_expanded = np.expand_dims(current_image, axis=0)
        
        # Predict the goal image using the diffusion model
        # The predict method returns a (1, 120, 120) numpy array of dtype uint8
        predicted_goal_image = self.inference.predict(current_image_expanded)

        # Remove the channel dimension to get shape (120, 120) as used in the rest of the class
        goal_img = predicted_goal_image.squeeze(0)
        
        return goal_img
    
    def generate_observation(self):
        curr_img = self.get_curr_image()
        goal_img = self.get_goal_image()
        assert curr_img.shape == goal_img.shape, 'Current image and goal image must have the same shape, but got {} and {}'.format(curr_img.shape, goal_img.shape)
        diff_img = np.expand_dims(np.abs(curr_img - goal_img), axis=0)

        observation = {
            'observation': np.expand_dims(np.abs(curr_img - goal_img), axis=0),
            'achieved_goal': np.expand_dims(np.abs(curr_img), axis=0),
            'desired_goal': np.expand_dims(np.abs(goal_img), axis=0)
        }

        if self.observation_space.contains(observation):
            return observation
        else:
            print('Observation is not in the observation space')

    def compute_reward(self, achieved_goal, desired_goal, info):

        # print('\n Achieved goal shape: ', achieved_goal.shape)

        diff = achieved_goal - desired_goal

        if diff.shape == (1, 120, 120):
            return -np.linalg.norm(diff)
        else:
            return -np.linalg.norm(diff[:, 0, :, :], axis=(1, 2))



    def step(self, action):
        observation_dict, original_reward, terminated, truncated, info = self.base_env.step(action)

        observation = self.generate_observation()

        reward = self.compute_reward(observation['achieved_goal'], observation['desired_goal'], info)

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation_dict, info = self.base_env.reset(**kwargs)

        # store default kettle attributes for later stepping
        self.default_kettle_dist = np.linalg.norm(observation_dict['achieved_goal']['kettle'] - observation_dict['desired_goal']['kettle'])
        self.default_kettle_state = observation_dict['achieved_goal']['kettle']

        return self.generate_observation(), info

    def render(self, vis=False):
        render_img = self.generate_observation()

        if vis:
            plt.imshow(render_img[0], cmap='jet')
            plt.pause(0.01)
        return self.base_env.render()

    def close(self):
        plt.close()
        return self.base_env.close()