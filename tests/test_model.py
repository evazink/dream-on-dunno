#!/usr/bin/env python3
"""
Quick test script to verify the diffusion model works correctly.
"""
import sys
import os
sys.path.append('src')

import torch
from diffusion.model import ConditionalUNet
from diffusion.utils import add_noise_to_images

def test_model():
    print("Testing ConditionalUNet model...")
    
    # Test configuration
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = ConditionalUNet(
        in_img_channels=6,
        out_img_channels=3,
        base_channels=64,
        channel_mults=(1, 2, 3, 4),
        num_res_blocks_per_level=2,
        attention_resolutions=(120, 60),
        time_emb_dim=256,
        time_emb_mlp_dim=1024,
        initial_img_resolution=480
    ).to(device)
    
    print(f"Model created successfully!")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Test forward pass
    batch_size = 2
    current_img = torch.randn(batch_size, 3, 480, 480, device=device)
    next_img = torch.randn(batch_size, 3, 480, 480, device=device)
    
    # Test the diffusion process
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Test add_noise_to_images function
    betas = torch.linspace(0.0001, 0.02, 1000)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    precomputed = {
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod).to(device),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1. - alphas_cumprod).to(device),
    }
    
    noisy_next_img, added_noise = add_noise_to_images(next_img, timesteps, device, precomputed)
    print(f"Noise added successfully! Noisy image shape: {noisy_next_img.shape}")
    
    # Test model forward pass
    model_input = torch.cat([noisy_next_img, current_img], dim=1)  # [B, 6, H, W]
    print(f"Model input shape: {model_input.shape}")
    
    model.eval()
    with torch.no_grad():
        predicted_noise = model(model_input, timesteps)
    
    print(f"Model forward pass successful!")
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print(f"Expected shape: {added_noise.shape}")
    print(f"Shapes match: {predicted_noise.shape == added_noise.shape}")
    
    # Test loss calculation
    loss_fn = torch.nn.MSELoss()
    loss = loss_fn(predicted_noise, added_noise)
    print(f"Loss calculation successful! Loss: {loss.item():.6f}")
    
    print("\n✅ All tests passed! Model is ready for training.")
    return True

if __name__ == "__main__":
    try:
        test_model()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 