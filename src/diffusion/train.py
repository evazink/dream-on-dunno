"""
Main training script for the Diffusion Model.
"""
import datetime
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
from tqdm import tqdm
import argparse

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add the src directory to the path so we can import properly
# Ensure this runs from the root of the project or adjust path as needed.
# Assuming launched from root: python src/diffusion/train.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion.model import ConditionalUNet # Keep direct model import
from src.data_utils.dataloader import KitchenPairDataset # Keep direct dataset import
from src.diffusion.utils import (
    get_linear_beta_schedule,
    precompute_diffusion_terms,
    add_noise_to_images,
    create_directory, # Replaces create_checkpoint_dir, create_tensorboard_dir
    init_tensorboard_writer, # Replaces init_tensorboard
    log_images_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint_path, # Replaces find_latest_checkpoint
    log_epoch_end_images_to_tensorboard # Import the new function
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Conditional U-Net Diffusion Model for Grayscale Images.")

    # --- Environment & Data --- 
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", 
                        choices=['cpu', 'mps', 'cuda'], help="Device to use for training.")
    parser.add_argument("--dataset_dir", type=str, default="data/kitchen_pairs", help="Root directory for the dataset.")
    parser.add_argument("--k_step_future", type=int, default=3, help="k-step future for KitchenPairDataset.")
    parser.add_argument("--force_rebuild_dataset", action="store_true", help="Force rebuild of the dataset.")
    parser.add_argument("--num_workers_loader", type=int, default=0, help="Number of DataLoader workers.")

    # --- Training Hyperparameters ---
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--disable_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP) even if on CUDA.")

    # --- Diffusion Process Hyperparameters ---
    parser.add_argument("--diffusion_timesteps", type=int, default=1000, help="Number of diffusion timesteps.")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Beta start value for linear schedule.")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Beta end value for linear schedule.")

    # --- Model Architecture (Grayscale: in_channels=2, out_channels=1 are fixed) ---
    parser.add_argument("--img_resolution", type=int, default=120, help="Image resolution for training and model.")
    parser.add_argument("--model_base_channels", type=int, default=32, help="Base channels for the U-Net.")
    parser.add_argument("--model_channel_mults", type=str, default="1,2,3", help="Channel multipliers (comma-separated string, e.g., '1,2,3').")
    parser.add_argument("--model_num_res_blocks", type=int, default=1, help="Number of residual blocks per U-Net level.")
    parser.add_argument("--model_attention_resolutions_divisor", type=int, default=4, 
                        help="Attention applied at img_resolution // X. E.g., 4 means at res/4, res/8 if deep enough etc. Comma separated for multiple specific divisors e.g. '4,8'. Currently supports one main divisor.")
    parser.add_argument("--model_time_emb_dim", type=int, default=128, help="Dimension of sinusoidal time embedding.")
    parser.add_argument("--model_time_emb_mlp_dim", type=int, default=512, help="Dimension of MLP for time embedding.")
    parser.add_argument("--model_dropout_rate", type=float, default=0.1, help="Dropout rate in U-Net residual blocks.")

    # --- Checkpointing & Resuming ---
    parser.add_argument("--checkpoint_dir", type=str, default="data/diffusion/checkpoints_grayscale", 
                        help="Directory to save checkpoints.")
    parser.add_argument("--checkpoint_freq", type=int, default=5, help="Save checkpoint every N epochs.")
    parser.add_argument("--save_best_model", action="store_true", help="Save the model with the best validation loss.")
    parser.add_argument("--resume_from_checkpoint", type=str, default="latest", 
                        choices=['latest', 'best', 'none'], help="Which checkpoint to resume from ('none' for no resume). Can also be a specific path.")

    # --- TensorBoard Logging ---
    parser.add_argument("--tensorboard_log_dir", type=str, default="data/diffusion/tensorboard_logs", 
                        help="Directory for TensorBoard logs.")
    parser.add_argument("--tensorboard_run_name_prefix", type=str, default="diffusion_grayscale_train", 
                        help="Prefix for TensorBoard run name.")
    parser.add_argument("--log_freq", type=int, default=5, help="Log scalars to TensorBoard every N batches.")
    parser.add_argument("--log_images", action="store_true", help="Log sample images to TensorBoard.")
    parser.add_argument("--log_image_freq_multiplier", type=int, default=10, 
                        help="Log images every (log_freq * M) steps. e.g., if log_freq=10, M=10, log images every 100 steps.")

    args = parser.parse_args()
    
    # Post-process some arguments
    args.model_channel_mults = tuple(map(int, args.model_channel_mults.split(',')))
    
    # Create attention resolutions based on divisor
    # Simple version: one divisor creates one attention resolution (and deeper if applicable)
    # More complex: allow comma-separated divisors for specific levels
    res = args.img_resolution
    attn_resos = set()
    current_res = args.img_resolution
    for _ in args.model_channel_mults:
        if args.model_attention_resolutions_divisor > 0 and current_res % args.model_attention_resolutions_divisor == 0:
             # A common strategy is to apply attention at one or two fixed downsampled sizes relative to input.
             # e.g., if divisor is 4, for a 128px input, it might be at 32px (128/4).
             # If it divides current_res and current_res is one of the feature map sizes, use it.
             # For now, let's use a simpler scheme: apply at img_resolution // divisor.
             # This might need more refinement based on how ConditionalUNet uses attention_resolutions.
             # The original train.py used: (IMG_RESOLUTION // 4,)
             # For now, let's stick to that logic. User can provide multiple via a string later if needed.
             pass # Deferring this logic until model instantiation based on how ConditionalUNet uses it.
    args.model_attention_resolutions = (args.img_resolution // args.model_attention_resolutions_divisor,) if args.model_attention_resolutions_divisor > 0 else ()

    return args

def main(args: argparse.Namespace):
    DEVICE = torch.device(args.device)
    print(f"Using device: {DEVICE}")

    # --- Setup Diffusion Process ---
    betas = get_linear_beta_schedule(args.diffusion_timesteps, args.beta_start, args.beta_end)
    diffusion_terms = precompute_diffusion_terms(betas, DEVICE)

    # --- Prepare Directories ---
    checkpoint_dir_path = Path(args.checkpoint_dir)
    tensorboard_log_dir_path = Path(args.tensorboard_log_dir)
    create_directory(checkpoint_dir_path, "Checkpoint")
    create_directory(tensorboard_log_dir_path, "TensorBoard Log")

    # --- Initialize TensorBoard ---
    tb_writer = init_tensorboard_writer(tensorboard_log_dir_path, args.tensorboard_run_name_prefix)

    # --- Model Definition ---
    # For grayscale: in_img_channels=2 (current_gray + future_gray), out_img_channels=1 (predicted_noise_gray)
    model_config = {
        'in_img_channels': 2,
        'out_img_channels': 1,
        'base_channels': args.model_base_channels,
        'channel_mults': args.model_channel_mults,
        'num_res_blocks_per_level': args.model_num_res_blocks,
        'attention_resolutions': args.model_attention_resolutions, # (args.img_resolution // args.model_attention_resolutions_divisor,)
        'dropout_rate': args.model_dropout_rate,
        'time_emb_dim': args.model_time_emb_dim,
        'time_emb_mlp_dim': args.model_time_emb_mlp_dim,
        'initial_img_resolution': args.img_resolution
    }
    model = ConditionalUNet(**model_config).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in U-Net: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    use_amp_actual = (DEVICE.type == 'cuda' and not args.disable_amp)
    scaler = torch.cuda.amp.GradScaler() if use_amp_actual else None
    if use_amp_actual:
        print("Using Automatic Mixed Precision (AMP) for CUDA.")

    # --- Dataset and DataLoader ---
    print("Initializing dataset...")
    train_dataset = KitchenPairDataset(
        k_step_future=args.k_step_future,
        data_dir=args.dataset_dir,
        force_rebuild=args.force_rebuild_dataset,
        target_size=args.img_resolution
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers_loader,
        pin_memory= (DEVICE.type == 'cuda'), # Pin memory if using CUDA
        persistent_workers= (args.num_workers_loader > 0),
        drop_last=True
    )
    print(f"Training dataset size: {len(train_dataset)} pairs. DataLoader ready.")

    # --- Resume from Checkpoint (if applicable) ---
    start_epoch = 0
    loss_history = []
    best_loss = float('inf')
    global_step = 0

    checkpoint_to_load = None
    if args.resume_from_checkpoint != 'none':
        if Path(args.resume_from_checkpoint).is_file():
            checkpoint_to_load = Path(args.resume_from_checkpoint)
        elif args.resume_from_checkpoint == 'latest':
            checkpoint_to_load = find_latest_checkpoint_path(checkpoint_dir_path)
        elif args.resume_from_checkpoint == 'best':
            checkpoint_to_load = checkpoint_dir_path / "best_model.pth"
            if not checkpoint_to_load.exists(): checkpoint_to_load = None # If best_model.pth doesn't exist

    if checkpoint_to_load and checkpoint_to_load.exists():
        print(f"Attempting to resume from: {checkpoint_to_load}")
        loaded_epoch, loaded_loss_hist, loaded_last_loss, loaded_model_cfg, loaded_train_args = load_checkpoint(
            checkpoint_to_load, model, optimizer, DEVICE
        )
        # Note: We are loading weights, but current script's args (model structure, LR etc.) will be used unless 
        # we explicitly decide to override them with loaded_train_args or loaded_model_cfg.
        # For simplicity, we assume current args define the session, and we just restore state.
        start_epoch = loaded_epoch + 1
        loss_history = loaded_loss_hist
        best_loss = min(loss_history) if loss_history else float('inf')
        # Correctly calculate global_step based on loaded epoch and dataloader length
        global_step = start_epoch * len(train_dataloader) 
        print(f"Resumed training from epoch {start_epoch}. Previous loss: {loaded_last_loss:.6f}. Best loss so far: {best_loss:.6f}")
    else:
        if args.resume_from_checkpoint != 'none':
            print(f"Checkpoint for resume ('{args.resume_from_checkpoint}') not found. Starting from scratch.")
        else:
            print("Starting training from scratch.")

    # --- Log Model Graph & Hyperparameters to TensorBoard ---
    try:
        # For add_graph, input needs to match model's forward signature
        sample_x_concat = torch.randn(1, model_config['in_img_channels'], args.img_resolution, args.img_resolution).to(DEVICE)
        sample_time = torch.randint(0, args.diffusion_timesteps, (1,), device=DEVICE).long()
        tb_writer.add_graph(model, (sample_x_concat, sample_time))
        print("Model graph logged to TensorBoard.")
    except Exception as e:
        print(f"Warning: Could not log model graph to TensorBoard: {e}")

    # Log hyperparameters (converting Path objects and tuples to strings for HParams)
    hparams_to_log = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    hparams_to_log['model_channel_mults'] = str(hparams_to_log['model_channel_mults']) # Ensure tuple is string
    hparams_to_log['model_attention_resolutions'] = str(hparams_to_log['model_attention_resolutions'])
    tb_writer.add_hparams(hparams_to_log, {'hparam/best_loss': best_loss, 'hparam/start_epoch': start_epoch})
    print("Hyperparameters logged to TensorBoard.")
    
    print(f"Starting training for {args.epochs - start_epoch} more epochs (Total {args.epochs})...")
    training_start_time = time.time()
    print(f"Training session started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_epoch_loss = 0.0
        num_batches_in_epoch = 0

        # Variables to store the last batch's data for epoch-end logging
        last_batch_current_img = None
        last_batch_next_img = None
        last_batch_noisy_target_img = None
        last_batch_predicted_noise = None
        last_batch_t_diffusion = None

        batch_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        
        for batch_idx, batch_data in enumerate(batch_pbar):
            # train_dataset yields: current_frame, future_frame, dataset_name, episode_num, current_step
            # All are already tensors after the dataloader.py changes.
            current_img_batch, next_img_batch, _, _, _ = batch_data

            optimizer.zero_grad()

            current_img_batch = current_img_batch.to(DEVICE) # Shape: [B, 1, H, W]
            next_img_batch = next_img_batch.to(DEVICE)       # Shape: [B, 1, H, W] (this is x_0 for the future state)

            t_diffusion = torch.randint(0, args.diffusion_timesteps, (current_img_batch.shape[0],), device=DEVICE).long()

            noisy_target_img, added_noise_epsilon = add_noise_to_images(next_img_batch, t_diffusion, DEVICE, diffusion_terms)

            # Model input: concatenate noisy_target_img and current_img_batch
            # Both are [B, 1, H, W], so cat result is [B, 2, H, W]
            model_input_concat = torch.cat([noisy_target_img, current_img_batch], dim=1)

            if use_amp_actual and scaler:
                with torch.cuda.amp.autocast():
                    predicted_noise = model(model_input_concat, t_diffusion)
                    loss = criterion(predicted_noise, added_noise_epsilon)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predicted_noise = model(model_input_concat, t_diffusion)
                loss = criterion(predicted_noise, added_noise_epsilon)
                loss.backward()
                optimizer.step()

            total_epoch_loss += loss.item()
            num_batches_in_epoch += 1
            
            if batch_idx % args.log_freq == 0:
                gc.collect()
                if DEVICE.type == 'mps': torch.mps.empty_cache()

            if global_step % args.log_freq == 0:
                tb_writer.add_scalar('Loss/Batch_Loss', loss.item(), global_step)
                tb_writer.add_scalar('Loss/Epoch_Avg_So_Far', total_epoch_loss / num_batches_in_epoch, global_step)
                tb_writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)
                
                grad_norm_total = 0
                for p_grad in model.parameters():
                    if p_grad.grad is not None:
                        grad_norm_total += p_grad.grad.data.norm(2).item() ** 2
                grad_norm_total = grad_norm_total ** 0.5
                tb_writer.add_scalar('Gradients/Total_Norm', grad_norm_total, global_step)
                
                if args.log_images and global_step % (args.log_freq * args.log_image_freq_multiplier) == 0:
                    with torch.no_grad():
                        log_images_to_tensorboard(
                            tb_writer, 
                            current_img_batch.detach(), 
                            next_img_batch.detach(), # Target x0 
                            model_input_concat[:, 0:1, :, :].detach(), # Noisy image component from concat
                            predicted_noise.detach(), 
                            added_noise_epsilon.detach(), 
                            global_step,
                            max_images_to_log=min(4, args.batch_size) # Ensure not more than batch size
                        )
            global_step += 1
            batch_pbar.set_postfix({'Loss': f'{loss.item():.6f}', 'Avg Loss': f'{total_epoch_loss/num_batches_in_epoch:.6f}'})

            # Store data from the current batch for potential epoch-end logging
            if args.log_images: # Only store if image logging is enabled
                last_batch_current_img = current_img_batch.detach()
                last_batch_next_img = next_img_batch.detach()
                last_batch_noisy_target_img = noisy_target_img.detach() # model_input_concat[:, 0:1, :, :].detach()
                last_batch_predicted_noise = predicted_noise.detach()
                last_batch_t_diffusion = t_diffusion.detach()

        avg_epoch_loss = total_epoch_loss / num_batches_in_epoch
        loss_history.append(avg_epoch_loss)
        tb_writer.add_scalar('Loss/Epoch_Avg', avg_epoch_loss, epoch)
        
        # Epoch-end image logging
        if args.log_images and last_batch_current_img is not None:
            with torch.no_grad():
                log_epoch_end_images_to_tensorboard(
                    writer=tb_writer,
                    epoch=epoch,
                    current_img=last_batch_current_img,
                    target_next_img=last_batch_next_img,
                    noisy_target_img=last_batch_noisy_target_img,
                    predicted_noise=last_batch_predicted_noise,
                    t_diffusion=last_batch_t_diffusion,
                    diffusion_terms=diffusion_terms,
                    max_images_to_log=min(4, args.batch_size)
                )

        is_best_epoch = args.save_best_model and avg_epoch_loss < best_loss
        if is_best_epoch:
            best_loss = avg_epoch_loss
            tb_writer.add_scalar('Loss/Best_Epoch_Avg', best_loss, epoch)

        status_msg = "*** NEW BEST ***" if is_best_epoch else ""
        tqdm.write(f"Epoch [{epoch+1}/{args.epochs}] - Avg Loss: {avg_epoch_loss:.6f} - Best: {best_loss:.6f} {status_msg}")

        if (epoch + 1) % args.checkpoint_freq == 0 or (epoch + 1) == args.epochs or is_best_epoch:
            # Pass vars(args) for training_args, and the constructed model_config
            save_checkpoint(
                model=model, optimizer=optimizer, epoch=epoch, loss=avg_epoch_loss,
                loss_history=loss_history, args_config=vars(args), model_config=model_config, 
                is_best=is_best_epoch, checkpoint_dir=checkpoint_dir_path
            )

    tb_writer.close()
    total_training_time = time.time() - training_start_time
    print("\nTraining complete.")
    print(f"Total training time: {datetime.timedelta(seconds=int(total_training_time))}")
    print(f"Best average epoch loss achieved: {best_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir_path}")
    print(f"TensorBoard logs: {tensorboard_log_dir_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
