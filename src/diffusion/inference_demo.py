"""
inference_demo.py - Diffusion Model Inference Demo

This script loads a trained diffusion model and demonstrates it by:
1. Loading an episode from the Minari dataset
2. For each step, using the current frame to predict the next frame
3. Displaying both the actual and predicted frames side by side
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import minari
from pathlib import Path
from tqdm import tqdm
import time
try:
    import torchvision.transforms.functional as TF
except ImportError:
    print("torchvision not found. Please install it to use rgb_to_grayscale. `pip install torchvision`")
    TF = None

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # Adjusted path for project root

from src.diffusion.model import ConditionalUNet
# Removed: from src.diffusion.utils import add_noise_to_images
# No direct need for load_checkpoint utility here if we load checkpoint directly
# For simplicity, we'll parse the checkpoint dictionary directly.

class DiffusionInference:
    def __init__(self, checkpoint_path, device='mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() and device == 'mps' else 'cpu') # Ensure MPS is available if selected
        print(f"Using device: {self.device}")
        
        # Load model, its config, and training args from checkpoint
        self.model, self.model_config, self.training_args = self._load_model_and_config(checkpoint_path)
        self.model.eval()
        
        # Set parameters from loaded training_args
        self.img_resolution = self.training_args.get('img_resolution', 120) # Default if not in args
        self.timesteps = self.training_args.get('diffusion_timesteps', 1000) # Default if not in args
        
        print(f"Model loaded successfully!")
        print(f"Training arguments: {self.training_args}")
        print(f"Model configuration: {self.model_config}")
        print(f"Model expects {self.img_resolution}x{self.img_resolution} grayscale images.")
        print(f"Diffusion timesteps from training: {self.timesteps}")

    def _load_model_and_config(self, checkpoint_path):
        """Load the trained model, its configuration, and training arguments from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get model configuration from checkpoint
        # Critical: This model_config is used to instantiate the UNet
        model_config = checkpoint.get('model_config')
        if not model_config:
            raise ValueError("Model configuration not found in checkpoint.")
            
        # Get training arguments from checkpoint
        training_args = checkpoint.get('training_args')
        if not training_args:
            # Fallback or error if training_args are essential and not found
            print("Warning: Training arguments not found in checkpoint. Using defaults for img_resolution and timesteps.")
            training_args = {} # Initialize to empty dict to avoid error on .get later

        # Create model
        # Ensure model_config has all necessary keys for ConditionalUNet
        # Example: initial_img_resolution might be needed by model if it's used internally
        # For now, assume model_config is complete as saved by train.py
        model = ConditionalUNet(**model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        loaded_loss = checkpoint.get('loss', 'unknown')
        if isinstance(loaded_loss, float):
            print(f"Original training loss: {loaded_loss:.6f}")
        else:
            print(f"Original training loss: {loaded_loss}")
        
        return model, model_config, training_args
    
    def preprocess_frame(self, frame_rgb_numpy):
        """Preprocess RGB frame (H,W,3) to grayscale tensor (1,1,H,W) for model input."""
        # Convert to tensor: [H, W, 3] -> [3, H, W] -> [1, 3, H, W]
        frame_tensor_rgb = torch.from_numpy(frame_rgb_numpy.copy()).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Convert to grayscale: [1, 3, H, W] -> [1, 1, H, W]
        if TF:
            frame_tensor_gray = TF.rgb_to_grayscale(frame_tensor_rgb, num_output_channels=1)
        else:
            # Fallback to manual conversion if torchvision is not available
            print("Warning: torchvision.transforms.functional.rgb_to_grayscale not available. Using manual conversion.")
            frame_tensor_gray = (
                0.2126 * frame_tensor_rgb[:, 0:1, :, :] +
                0.7152 * frame_tensor_rgb[:, 1:2, :, :] +
                0.0722 * frame_tensor_rgb[:, 2:3, :, :]
            )

        # Resize to model's expected resolution
        frame_resized = F.interpolate(frame_tensor_gray, size=(self.img_resolution, self.img_resolution), 
                                    mode='bilinear', align_corners=False)
        
        # Normalize to [-1, 1] (assuming images were normalized this way during training)
        frame_normalized = frame_resized / 127.5 - 1 # If original uint8 range [0,255]
        # If original range was already [0,1] float, then: frame_normalized = frame_resized * 2.0 - 1.0
        # Given that dataloader.py's _convert_to_tensors does / 255.0 then * 2.0 - 1.0,
        # direct output from env.render() is uint8 [0,255]. So /127.5 - 1 is correct here.
        
        return frame_normalized.clamp(-1.0, 1.0) # Ensure it's in range
    
    def postprocess_frame(self, tensor_gray, target_display_size_hw=(480, 480)):
        """Convert model output (grayscale tensor [1,1,H,W]) back to displayable RGB numpy image."""
        # Denormalize from [-1, 1] to [0, 1]
        tensor_denormalized = (tensor_gray.cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
        
        # Resize to target display size
        if tensor_denormalized.shape[-2:] != target_display_size_hw:
            tensor_resized = F.interpolate(tensor_denormalized, size=target_display_size_hw, 
                                 mode='bilinear', align_corners=False)
        else:
            tensor_resized = tensor_denormalized

        # Convert [1, 1, H, W] to [H, W, 1] numpy array for grayscale
        img_hw1 = tensor_resized.squeeze(0).permute(1, 2, 0).numpy()
        
        # Convert grayscale [H,W,1] to RGB [H,W,3] by repeating the channel for matplotlib
        img_hw3_rgb = np.repeat(img_hw1, 3, axis=2)
        
        return (img_hw3_rgb * 255).astype(np.uint8)
    
    def predict_next_frame(self, current_frame_tensor_gray, num_inference_steps=50):
        """
        Predict the next frame using the diffusion model.
        current_frame_tensor_gray: preprocessed current frame [1, 1, H_model, W_model]
        """
        with torch.no_grad():
            # 1. Define the noise schedule (simplified for demo: single large step)
            # For a proper DDPM/DDIM sampler, you'd iterate `num_inference_steps`
            
            # 2. Start with pure noise for the "next frame to be generated"
            # This noise is what the model will try to denoise into an image.
            # It should match the model's output channel (1 for grayscale).
            img_t = torch.randn(1, 1, self.img_resolution, self.img_resolution, device=self.device)

            # 3. The model input requires concatenation: (noisy_image_to_denoise, condition_image)
            # noisy_image_to_denoise = img_t (our current estimate of the next frame, starting with noise)
            # condition_image = current_frame_tensor_gray
            # Model's `in_img_channels` is 2.
            model_input_concat = torch.cat([img_t, current_frame_tensor_gray], dim=1) # [1, 2, H, W]
            
            # 4. Select timestep for denoising.
            # For a single-step generation (very simplified), we might pick a t near 0 or a fixed medium t.
            # Or, as in original, a random t. Let's use a high t for a single big step from noise.
            # For actual generation, you'd loop from t=T-1 down to 0.
            t_diffusion = torch.full((1,), self.timesteps - 1, device=self.device, dtype=torch.long) # Denoise from near pure noise
            # t_diffusion = torch.randint(0, self.timesteps, (1,), device=self.device).long() # Original random t

            # 5. Predict the noise component added to the (hypothetical clean) next frame.
            # Model output shape will be [1, 1, H, W] as out_img_channels=1.
            predicted_noise_epsilon = self.model(model_input_concat, t_diffusion)
            
            # 6. Estimate the "clean" next frame (x_0_hat) from img_t (x_t) and predicted_noise_epsilon.
            # This is a very simplified version of one step of the reverse diffusion process.
            # A common one-step x0 prediction is: x_0_hat = (x_t - sqrt(1-alpha_cumprod_t) * epsilon) / sqrt(alpha_cumprod_t)
            # For this demo, let's use the highly simplified: x_0_hat = x_t - predicted_noise_epsilon
            # This assumes alpha_cumprod_t is close to 1, and sqrt_one_minus_alpha_cumprod_t is also close to 1
            # Or that the model has learned to directly predict x_0 - x_t if trained that way.
            # Given the training loss (MSE on noise), predicted_noise_epsilon is an estimate of noise.
            # So, img_t - predicted_noise_epsilon aims to remove that noise from img_t.
            predicted_next_frame_tensor = img_t - predicted_noise_epsilon # Simplified denoising step
            
            return predicted_next_frame_tensor
    
    def run_episode_demo(self, dataset_name='D4RL/kitchen/complete-v2', episode_idx=0, max_steps=50):
        """Run inference demo on a specific episode."""
        print(f"\nLoading dataset: {dataset_name}")
        dataset = minari.load_dataset(dataset_name, download=True)
        dataset.set_seed(seed=42)
        
        # Get environment for rendering
        env = dataset.recover_environment(render_mode='rgb_array')
        env.reset(seed=42)
        
        # Get a specific episode
        episodes = dataset.sample_episodes(n_episodes=episode_idx+1)
        episode = episodes[episode_idx]
        
        print(f"Running inference demo on episode {episode_idx}")
        print(f"Episode length: {len(episode.actions)} steps")
        print(f"Will process first {min(max_steps, len(episode.actions))} steps")
        
        # Setup visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].set_title("Current Frame (Environment)")
        axes[1].set_title("Predicted Next Frame (Model)")
        plt.ion()  # Interactive mode for real-time updates
        
        # Reset environment and step through episode
        env.reset()
        
        for step in tqdm(range(min(max_steps, len(episode.actions))), desc="Processing steps"):
            # Get current frame from environment
            current_frame = env.render()  # [H, W, 3] numpy array
            
            # Preprocess for model
            current_frame_tensor = self.preprocess_frame(current_frame)
            
            # Predict next frame
            start_time = time.time()
            predicted_frame_tensor = self.predict_next_frame(current_frame_tensor)
            inference_time = time.time() - start_time
            
            # Postprocess for display
            predicted_frame = self.postprocess_frame(predicted_frame_tensor, target_display_size_hw=(480, 480))
            
            # Display side by side
            axes[0].clear()
            axes[1].clear()
            
            axes[0].imshow(current_frame)
            axes[0].set_title(f"Current Frame - Step {step}")
            axes[0].axis('off')
            
            axes[1].imshow(predicted_frame)
            axes[1].set_title(f"Predicted Next Frame ({inference_time:.3f}s)")
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.pause(0.1)  # Brief pause to see the frame
            
            # Step environment forward
            if step < len(episode.actions):
                action = episode.actions[step]
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    print(f"Episode ended at step {step}")
                    break
        
        plt.ioff()  # Turn off interactive mode
        plt.show(block=True) # Use block=True to keep plot open after demo
        env.close()
        print("Demo completed!")

def main():
    # Configuration
    CHECKPOINT_PATH = "data/diffusion/checkpoints/model_v3/checkpoint_epoch_084.pth" # Old example
    # Default path for checkpoints saved by train.py with default args
    #CHECKPOINT_PATH = "data/diffusion/checkpoints_grayscale/best_model.pth" 
    
    # Optional: Allow checkpoint path to be passed as a command-line argument
    if len(sys.argv) > 1:
        chkpt_arg = Path(sys.argv[1])
        if chkpt_arg.exists() and chkpt_arg.is_file():
            CHECKPOINT_PATH = str(chkpt_arg)
            print(f"Using checkpoint from command line argument: {CHECKPOINT_PATH}")
        else:
            print(f"Warning: Command line argument '{sys.argv[1]}' is not a valid file. Using default: {CHECKPOINT_PATH}")

    DATASET_NAME = 'D4RL/kitchen/complete-v2' # As per original
    EPISODE_IDX = 0
    MAX_STEPS = 30  # Process first 30 steps
    
    try:
        # Create inference demo
        demo = DiffusionInference(CHECKPOINT_PATH, device='mps') # or 'cpu' or 'cuda'
        
        # Run the demo
        demo.run_episode_demo(
            dataset_name=DATASET_NAME,
            episode_idx=EPISODE_IDX,
            max_steps=MAX_STEPS
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you have a trained model checkpoint available.")
        print("You can specify a different checkpoint path in the script.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 