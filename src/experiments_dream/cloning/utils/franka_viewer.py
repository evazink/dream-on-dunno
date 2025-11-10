# utils/franka_viewer.py
import matplotlib.pyplot as plt
import numpy as np

class FrankaKitchenViewer:
    def __init__(self, figsize=(8, 8), title="Franka Kitchen Environment"):
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)
        self.im = None
        self.title = title
        self.step_text = None
        self.reward_text = None
        plt.ion() # Turn on interactive mode
        self.fig.show() # Display the figure (it will be empty initially)

    def update(self, frame, step=None, reward=None):
        """
        Updates the displayed frame and information.
        :param frame: A numpy array representing the image to display.
        :param step: Current step number
        :param reward: Current reward value
        """
        if self.im is None:
            self.im = self.ax.imshow(frame)
            self.ax.set_title(self.title)
            self.ax.axis('off') # Turn off axes ticks and labels
            
            # Initialize text objects for step and reward
            self.step_text = self.ax.text(10, 30, '', color='white', 
                                        bbox=dict(facecolor='black', alpha=0.7))
            self.reward_text = self.ax.text(10, 60, '', color='white',
                                          bbox=dict(facecolor='black', alpha=0.7))
        else:
            self.im.set_data(frame)
            
            # Update step and reward text if provided
            if step is not None:
                self.step_text.set_text(f'Step: {step}')
            if reward is not None:
                self.reward_text.set_text(f'Reward: {reward:.2f}')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01) # Small pause to allow the display to update

    def close(self):
        plt.ioff()
        plt.close(self.fig)
        print("Viewer closed.")