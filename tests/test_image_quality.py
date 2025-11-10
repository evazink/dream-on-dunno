#!/usr/bin/env python3
"""
Test script for image downsampling and upsampling effects.
Shows original vs processed images side by side.
Samples fresh renders from the kitchen environment.
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import minari

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def sample_environment_render():
    """Sample a fresh render from the kitchen environment."""
    try:
        print("Loading kitchen environment and sampling render...")
        
        # Load dataset and environment (same as env_test.py)
        dataset = minari.load_dataset('D4RL/kitchen/mixed-v2', download=True)
        env = dataset.recover_environment(render_mode='rgb_array')
        dataset.set_seed(seed=np.random.randint(0, 1000))  # Random seed for variety
        
        # Sample a random episode
        episode = dataset.sample_episodes(n_episodes=1)[0]
        env.reset()
        
        # Take a few random steps to get an interesting state
        num_steps = np.random.randint(5, 50)  # Random number of steps
        print(f"Taking {num_steps} steps in environment...")
        
        for t in range(min(num_steps, len(episode.actions))):
            obs, _, terminated, truncated, info = env.step(episode.actions[t])
            if terminated or truncated:
                break
        
        # Get the current state image
        current_img = env.render()  # Returns numpy array (H, W, 3) with values [0, 255]
        
        # Convert to [0, 1] range
        current_img = current_img.astype(np.float32) / 255.0
        
        print(f"Sampled environment render: {current_img.shape}")
        print(f"Value range: [{current_img.min():.3f}, {current_img.max():.3f}]")
        
        env.close()
        return current_img
        
    except Exception as e:
        print(f"Could not sample from environment: {e}")
        return None

def load_sample_from_dataset():
    """Load a sample image from the kitchen dataset."""
    try:
        from data.dataloader import KitchenPairDataset
        
        print("Loading sample from kitchen dataset...")
        dataset = KitchenPairDataset(
            k_step_future=5,
            data_dir="data/kitchen_pairs", 
            force_rebuild=False
        )
        
        # Get a random sample
        idx = np.random.randint(0, len(dataset))
        current_img, next_img = dataset[idx]
        
        # Convert from tensor [-1, 1] to numpy [0, 1]
        current_img = (current_img + 1.0) / 2.0
        current_img = current_img.permute(1, 2, 0).numpy()  # CHW -> HWC
        
        print(f"Loaded sample {idx} from dataset")
        return current_img
        
    except Exception as e:
        print(f"Could not load from dataset: {e}")
        return None

def load_image_from_file(image_path):
    """Load an image from a file path."""
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((480, 480))  # Resize to standard size
        image = np.array(image) / 255.0  # Convert to [0, 1]
        print(f"Loaded image from: {image_path}")
        return image
    except Exception as e:
        print(f"Could not load image from {image_path}: {e}")
        return None

def create_test_image():
    """Create a test image with patterns to see downsampling effects."""
    print("Creating test pattern image...")
    
    # Create a test image with various patterns
    img = np.zeros((480, 480, 3))
    
    # Add gradient
    for i in range(480):
        img[i, :, 0] = i / 480  # Red gradient
    
    # Add checkerboard pattern
    checker_size = 20
    for i in range(0, 480, checker_size):
        for j in range(0, 480, checker_size):
            if (i // checker_size + j // checker_size) % 2 == 0:
                img[i:i+checker_size, j:j+checker_size, 1] = 1.0  # Green squares
    
    # Add some fine details
    for i in range(0, 480, 5):
        img[i:i+2, :, 2] = 1.0  # Blue lines
    
    return img

def downsample_upsample_image(image, target_size=128):
    """
    Downsample image to target_size x target_size, then upsample back to original size.
    
    Args:
        image: numpy array of shape (H, W, 3) with values in [0, 1]
        target_size: size to downsample to
    
    Returns:
        processed_image: numpy array of same shape as input
    """
    # Convert to torch tensor
    image_tensor = torch.from_numpy(image).float()
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
    
    # Downsample to target size
    downsampled = F.interpolate(
        image_tensor, 
        size=(target_size, target_size), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Upsample back to original size
    upsampled = F.interpolate(
        downsampled, 
        size=(480, 480), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Convert back to numpy
    result = upsampled.squeeze(0).permute(1, 2, 0).numpy()  # BCHW -> HWC
    result = np.clip(result, 0, 1)  # Ensure values are in [0, 1]
    
    return result

def calculate_metrics(original, processed):
    """Calculate quality metrics between original and processed images."""
    # Mean Squared Error
    mse = np.mean((original - processed) ** 2)
    
    # Peak Signal-to-Noise Ratio
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(1.0 / np.sqrt(mse))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(original - processed))
    
    return mse, psnr, mae

def plot_comparison(original, processed, title="Image Quality Comparison"):
    """Plot original and processed images side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original (480x480)')
    axes[0].axis('off')
    
    # Processed image  
    axes[1].imshow(processed)
    axes[1].set_title('Processed (480→target→480)')
    axes[1].axis('off')
    
    # Difference image
    diff = np.abs(original - processed)
    diff_display = np.mean(diff, axis=2)  # Convert to grayscale for better visualization
    im = axes[2].imshow(diff_display, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title('Absolute Difference')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Calculate and display metrics
    mse, psnr, mae = calculate_metrics(original, processed)
    fig.suptitle(f'{title}\nMSE: {mse:.6f} | PSNR: {psnr:.2f} dB | MAE: {mae:.6f}', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to test image quality degradation."""
    print("=== Image Quality Degradation Test ===")
    print("This script tests the effect of downsampling on kitchen environment renders")
    print()
    
    # Try to get an image from different sources
    image = None
    
    # Option 1: Sample fresh render from environment (default)
    if len(sys.argv) == 1:  # No command line arguments
        print("Sampling fresh render from kitchen environment...")
        image = sample_environment_render()
        
        # Fallback to dataset if environment sampling fails
        if image is None:
            print("Environment sampling failed, trying dataset...")
            image = load_sample_from_dataset()
    
    # Option 2: Use dataset sample
    elif len(sys.argv) == 2 and sys.argv[1] == "--dataset":
        print("Using sample from preprocessed dataset...")
        image = load_sample_from_dataset()
    
    # Option 3: Load from file if path provided
    elif len(sys.argv) == 2:
        image_path = sys.argv[1]
        image = load_image_from_file(image_path)
    
    # Option 4: Create test pattern if nothing else works
    if image is None:
        print("Using generated test pattern...")
        image = create_test_image()
    
    # Test different downsampling sizes
    test_sizes = [32, 64, 128, 256]
    
    for size in test_sizes:
        print(f"\nTesting downsampling to {size}x{size}...")
        
        # Process the image
        processed = downsample_upsample_image(image, target_size=size)
        
        # Calculate metrics
        mse, psnr, mae = calculate_metrics(image, processed)
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  MAE: {mae:.6f}")
        
        # Show comparison
        plot_comparison(image, processed, f"Kitchen Environment: 480→{size}→480")
    
    print("\nTest complete!")
    print("\nUsage:")
    print("  python test_image_quality.py                    # Sample fresh render from environment")
    print("  python test_image_quality.py --dataset          # Use sample from preprocessed dataset")
    print("  python test_image_quality.py path/to/image.jpg  # Use specific image file")

if __name__ == "__main__":
    main() 