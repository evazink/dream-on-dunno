import gymnasium as gym
import numpy as np
import gymnasium_robotics
from gymnasium import spaces
from PIL import Image
import matplotlib.pyplot as plt

class CustomFrankaEnv(gym.Wrapper):
    def __init__(self, env, goal_step_offset=20, overlay_alpha=0.4):
        super().__init__(env)
        self.base_env = env
        self.goal_step_offset = goal_step_offset
        self.overlay_alpha = overlay_alpha

        self.img_shape = (480, 480, 3)
        self.goal_image = np.zeros(self.img_shape, dtype=np.uint8)

        self.observation_space = spaces.Dict({
            "observation": spaces.Box(0, 255, shape=self.img_shape, dtype=np.uint8),
            "achieved_goal": spaces.Box(0, 255, shape=self.img_shape, dtype=np.uint8),
            "desired_goal": spaces.Box(0, 255, shape=self.img_shape, dtype=np.uint8),
        })
        
        self.action_space = self.base_env.action_space

    def reset(self, **kwargs):
        _, info = self.base_env.reset(**kwargs)
        self._simulate_goal_image()
        curr_image = self._get_uint8_image()

        obs = {
            "observation": curr_image,
            "achieved_goal": curr_image,
            "desired_goal": self.goal_image
        }
        return obs, info

    def step(self, action):
        _, _, terminated, truncated, info = self.base_env.step(action)
        curr_image = self._get_uint8_image()
        reward = self.compute_reward(curr_image, self.goal_image, info)

        obs = {
            "observation": curr_image,
            "achieved_goal": curr_image,
            "desired_goal": self.goal_image
        }
        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        # Reward is negative L2 distance between current and goal image.
        achieved_goal_f = achieved_goal.astype(np.float32)
        desired_goal_f = desired_goal.astype(np.float32)

        diff = achieved_goal_f - desired_goal_f
        
        # Flatten and compute L2 norm. This handles both single and batch images.
        norm_diff = np.linalg.norm(diff.reshape(diff.shape[0], -1) if diff.ndim == 4 else diff.flatten())
        
        # Simple normalization by maximum pixel value.
        reward = (-norm_diff / 255.0 + 70) * 0.1
        
        return float(reward) if np.isscalar(reward) else reward.astype(np.float64)

    def _simulate_goal_image(self):
        self.base_env.reset()
        for _ in range(self.goal_step_offset):
            self.base_env.step(self.action_space.sample())
        self.goal_image = self._get_uint8_image()
        self.base_env.reset()

    def _get_uint8_image(self):
        img = self.base_env.render()
        
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        if img_array.dtype != np.uint8:
            img_array = (img_array * 255).astype(np.uint8) if img_array.max() <= 1.01 else img_array.astype(np.uint8)
        
        return img_array

    def render(self, mode='rgb_array'):
        current_image = self._get_uint8_image()

        if self.goal_image is None:
            return current_image

        current_pil = Image.fromarray(current_image).convert('RGBA')
        goal_pil = Image.fromarray(self.goal_image).convert('RGBA')
        blended = Image.blend(current_pil, goal_pil, self.overlay_alpha)

        return np.array(blended.convert('RGB'))

    def close(self):
        plt.close('all')
        self.base_env.close()