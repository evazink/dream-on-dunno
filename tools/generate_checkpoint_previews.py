import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import re
import os

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffusion.model import ConditionalUNet
from src.diffusion.utils import get_linear_beta_schedule, precompute_diffusion_terms, add_noise_to_images
from src.data_utils.dataloader import KitchenPairDataset

class CheckpointGenerator:
    def __init__(self, checkpoint_path, device='mps'):
        self.device = torch.device(device if torch.backends.mps.is_available() and device == 'mps' else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model, self.model_config, self.training_args = self._load_model_and_config(checkpoint_path)
        self.model.eval()
        
        self.img_resolution = self.training_args.get('img_resolution', 120)
        self.timesteps = self.training_args.get('diffusion_timesteps', 1000)
        
        betas = get_linear_beta_schedule(
            self.timesteps,
            self.training_args.get('beta_start', 0.0001),
            self.training_args.get('beta_end', 0.02)
        )
        self.diffusion_terms = precompute_diffusion_terms(betas, self.device)

        print(f"Model loaded from {checkpoint_path}")

    def _load_model_and_config(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint.get('model_config')
        if not model_config:
            raise ValueError("Model configuration not found in checkpoint.")
        training_args = checkpoint.get('training_args', {})

        model = ConditionalUNet(**model_config).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, model_config, training_args

    def reconstruct_image(self, condition_img_tensor, target_img_tensor):
        """Reconstructs an image based on a condition, mimicking the training validation logic."""
        with torch.no_grad():
            # Ensure tensors are on the correct device and have a batch dimension
            condition_img = condition_img_tensor.to(self.device).unsqueeze(0)
            target_img = target_img_tensor.to(self.device).unsqueeze(0)

            # Define a high timestep 't' to add significant noise, similar to training validation.
            t = torch.full((1,), self.timesteps - 1, device=self.device, dtype=torch.long)

            # Add noise to the target image to create x_t
            noisy_img, added_noise = add_noise_to_images(target_img, t, self.device, self.diffusion_terms)
            
            # The model's input is the noisy image concatenated with the condition
            model_input = torch.cat([noisy_img, condition_img], dim=1)

            # Predict the noise component from the noisy image
            predicted_noise = self.model(model_input, t)

            # Reconstruct the image using the formula from utils.py
            # x_0_hat = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            sqrt_alphas_cumprod_t = self.diffusion_terms["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_cumprod_t = self.diffusion_terms["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
            
            epsilon_div = 1e-8
            reconstructed_img_tensor = (noisy_img - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / \
                                       (sqrt_alphas_cumprod_t + epsilon_div)
            
            # Post-process for display
            img = (reconstructed_img_tensor.cpu().clamp(-1.0, 1.0) + 1.0) / 2.0
            img = img.squeeze(0).permute(1, 2, 0).numpy()
            img_rgb = np.repeat(img, 3, axis=2)
            
            return (img_rgb * 255).astype(np.uint8)

def main():
    checkpoint_dir = Path("data/diffusion/checkpoints/model_v3/")
    output_dir = Path("data/diffusion/checkpoints/model_v3/previews/")
    output_dir.mkdir(exist_ok=True)

    # --- Load Dataset to get a sample for reconstruction ---
    print("Loading dataset to fetch a sample for reconstruction...")
    # These parameters should match what was used in training to find the .pkl file
    # You might need to adjust k_step_future and target_size
    try:
        dataset = KitchenPairDataset(k_step_future=3, data_dir="data/kitchen_pairs", target_size=120)
        # Fetch a consistent sample to use for all checkpoints
        sample_idx = 100 
        current_frame, future_frame, _, _, _ = dataset[sample_idx]
        print(f"Successfully loaded dataset and fetched sample item {sample_idx}.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the dataset exists at 'data/kitchen_pairs' and was generated with k=3 and size=120.")
        return


    checkpoint_paths = sorted(list(checkpoint_dir.glob("checkpoint_epoch_*.pth")))

    for checkpoint_path in checkpoint_paths:
        match = re.search(r"checkpoint_epoch_(\d+)\.pth", checkpoint_path.name)
        if not match:
            continue
        
        epoch_num = int(match.group(1))
        print(f"\nProcessing checkpoint for epoch {epoch_num}...")

        try:
            generator = CheckpointGenerator(checkpoint_path)
            # Use the reconstruction method with the sample data
            reconstructed_image = generator.reconstruct_image(current_frame, future_frame)
            
            # Save the image
            output_path = output_dir / f"reconstruction_epoch_{epoch_num}.png"
            plt.imsave(output_path, reconstructed_image)
            print(f"Saved reconstruction preview to {output_path}")

        except Exception as e:
            print(f"Failed to process checkpoint {checkpoint_path.name}: {e}")

if __name__ == "__main__":
    main() 