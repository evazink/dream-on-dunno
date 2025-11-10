"""
utils.py
"""
import torch
import torch.nn.functional as F # For F.pad in precomputed values setup
from pathlib import Path
import time
import json
from torch.utils.tensorboard import SummaryWriter # For init_tensorboard and log_images

# --- Diffusion Schedule Helpers ---

def get_linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Returns a linear beta schedule from beta_start to beta_end.
    Args:
        timesteps: Number of diffusion timesteps.
        beta_start: Starting value for beta.
        beta_end: Ending value for beta.
    Returns:
        A 1D tensor of beta values.
    """
    return torch.linspace(beta_start, beta_end, timesteps)

def precompute_diffusion_terms(betas: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Precomputes terms needed for the diffusion process based on the beta schedule.
    Args:
        betas: A 1D tensor of beta values.
        device: The torch device to move precomputed tensors to.
    Returns:
        A dictionary of precomputed tensors, moved to the specified device.
    """
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    # alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Not used in add_noise, but useful for sampling
    
    precomputed = {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        # "alphas_cumprod_prev": alphas_cumprod_prev, # For sampling (p_sample)
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1. - alphas_cumprod),
        # "sqrt_recip_alphas": torch.sqrt(1.0 / alphas), # For sampling
        # "posterior_variance": betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod), # For sampling
    }
    for key in precomputed:
        precomputed[key] = precomputed[key].to(device)
    return precomputed


def add_noise_to_images(original_images: torch.Tensor, t: torch.Tensor, device: torch.device, diffusion_terms: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Adds noise to images x_0 to get x_t based on precomputed diffusion terms."""
    # original_images: [B, C, H, W]
    # t: [B] tensor of timesteps
    batch_size = original_images.shape[0]
    noise = torch.randn_like(original_images, device=device) # epsilon

    # Gather coefficients for each t in the batch
    sqrt_alphas_cumprod_t = diffusion_terms["sqrt_alphas_cumprod"][t].to(device).view(batch_size, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = diffusion_terms["sqrt_one_minus_alphas_cumprod"][t].to(device).view(batch_size, 1, 1, 1)

    noisy_images = sqrt_alphas_cumprod_t * original_images + \
                   sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_images, noise # Return noisy image and the noise that was added (target)

# --- Filesystem & Logging Helpers ---

def create_directory(path: Path, description: str = "directory") -> None:
    """Create a directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    print(f"Ensured {description} exists: {path}")

def init_tensorboard_writer(base_log_dir: Path, run_name_prefix: str = "run", run_name_suffix: str | None = None) -> SummaryWriter:
    """Initialize TensorBoard SummaryWriter with a unique run name."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if run_name_suffix:
        run_name = f"{run_name_prefix}_{run_name_suffix}_{timestamp}"
    else:
        run_name = f"{run_name_prefix}_{timestamp}"
    
    log_dir = base_log_dir / run_name
    create_directory(log_dir, "TensorBoard run")
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard logging to: {log_dir}")
    print(f"Start TensorBoard with: tensorboard --logdir={base_log_dir}") # Corrected: use base_log_dir directly
    return writer

def log_images_to_tensorboard(writer: SummaryWriter, current_img: torch.Tensor, target_next_img: torch.Tensor, 
                              noisy_input_img: torch.Tensor, predicted_noise: torch.Tensor, 
                              actual_noise: torch.Tensor, global_step: int, max_images_to_log: int = 4) -> None:
    """Log sample images to TensorBoard for visual inspection."""
    batch_size = min(current_img.shape[0], max_images_to_log)
    
    def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
        return (img_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0 # Clamp to ensure valid range after ops
    
    if current_img.shape[1] not in [1, 3]: # Channels
        print(f"Warning: log_images_to_tensorboard expects 1 or 3 channels, got {current_img.shape[1]}. Skipping image logging.")
        return

    writer.add_images('Input/1_Current_Image', denormalize(current_img[:batch_size]), global_step)
    writer.add_images('Input/2_Target_Next_Image(x0)', denormalize(target_next_img[:batch_size]), global_step)
    writer.add_images('Input/3_Noisy_Image(xt)', denormalize(noisy_input_img[:batch_size]), global_step)
    
    # Visualize noise (add 0.5 to center around gray for better visualization if noise is approx -1 to 1)
    # Or simply denormalize if noise can be outside that range.
    # For MSE loss, noise can be any float. Let's just denormalize it like an image for viz.
    writer.add_images('Noise/1_Predicted_Noise', denormalize(predicted_noise[:batch_size]), global_step)
    writer.add_images('Noise/2_Actual_Noise', denormalize(actual_noise[:batch_size]), global_step)
    writer.add_images('Noise/3_Difference', denormalize(predicted_noise[:batch_size] - actual_noise[:batch_size]), global_step)

def log_epoch_end_images_to_tensorboard(
    writer: SummaryWriter, 
    epoch: int,
    current_img: torch.Tensor, 
    target_next_img: torch.Tensor, # This is x_0
    noisy_target_img: torch.Tensor, # This is x_t
    predicted_noise: torch.Tensor, # This is epsilon_theta(x_t, t)
    t_diffusion: torch.Tensor, # Timesteps used to generate x_t and for prediction
    diffusion_terms: dict[str, torch.Tensor],
    max_images_to_log: int = 4
) -> None:
    """
    Logs current image, target next image (ground truth x_0), and reconstructed next image (x_0_hat)
    to TensorBoard at the end of an epoch.

    Args:
        writer: TensorBoard SummaryWriter instance.
        epoch: The current epoch number (for tagging).
        current_img: Batch of current images [B, C, H, W].
        target_next_img: Batch of target next images (ground truth x_0) [B, C, H, W].
        noisy_target_img: Batch of noisy target images (x_t) [B, C, H, W].
        predicted_noise: Batch of predicted noise (epsilon_theta) by the model [B, C, H, W].
        t_diffusion: Batch of timesteps t used for these samples [B].
        diffusion_terms: Dictionary of precomputed diffusion terms.
        max_images_to_log: Maximum number of images from the batch to log.
    """
    batch_size = min(current_img.shape[0], max_images_to_log)
    device = current_img.device

    def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
        return (img_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0

    if current_img.shape[1] not in [1, 3]: # Channels
        print(f"Warning: log_epoch_end_images_to_tensorboard expects 1 or 3 channels, got {current_img.shape[1]}. Skipping image logging.")
        return

    # Reconstruct x_0_hat from x_t and predicted_noise
    # x_0_hat = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
    sqrt_alphas_cumprod_t = diffusion_terms["sqrt_alphas_cumprod"][t_diffusion].to(device).view(-1, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = diffusion_terms["sqrt_one_minus_alphas_cumprod"][t_diffusion].to(device).view(-1, 1, 1, 1)

    # Ensure predicted_noise and noisy_target_img are on the correct device and batch size matches for slicing
    predicted_noise_batch = predicted_noise[:batch_size]
    noisy_target_img_batch = noisy_target_img[:batch_size]
    
    # Handle potential division by zero if sqrt_alphas_cumprod_t is zero for some t
    # (though unlikely for typical diffusion schedules unless t is very large and alpha_cumprod is ~0)
    # Add a small epsilon to prevent division by zero.
    epsilon_div = 1e-8 
    
    reconstructed_next_img = (noisy_target_img_batch - sqrt_one_minus_alphas_cumprod_t[:batch_size] * predicted_noise_batch) / \
                             (sqrt_alphas_cumprod_t[:batch_size] + epsilon_div)

    writer.add_images('Epoch_End/1_Current_Image', denormalize(current_img[:batch_size]), epoch)
    writer.add_images('Epoch_End/2_Target_Next_Image(x0)', denormalize(target_next_img[:batch_size]), epoch)
    writer.add_images('Epoch_End/3_Reconstructed_Next_Image(x0_hat)', denormalize(reconstructed_next_img), epoch)

# --- Checkpointing Helpers ---

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, loss: float, 
                    loss_history: list, args_config: dict, model_config: dict, 
                    is_best: bool = False, checkpoint_dir: Path = Path("checkpoints")) -> None:
    """Save model checkpoint with training state and configurations."""
    create_directory(checkpoint_dir, "checkpoint")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'loss_history': loss_history,
        'model_config': model_config,    # From model instantiation
        'training_args': args_config, # From argparse
    }
    
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
    
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model: {best_path}")
    
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    torch.save(checkpoint, latest_path)
    
    # Save combined config as JSON for easy inspection
    combined_config = {
        'model_config': model_config,
        'training_args': args_config,
        'last_epoch': epoch,
        'last_loss': float(loss) if loss is not None else None
    }
    config_path = checkpoint_dir / f"config_epoch_{epoch:03d}.json"
    with open(config_path, 'w') as f:
        json.dump(combined_config, f, indent=4)
    latest_config_path = checkpoint_dir / "latest_config.json"
    with open(latest_config_path, 'w') as f:
        json.dump(combined_config, f, indent=4)

def load_checkpoint(checkpoint_path: Path, model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer | None = None, 
                    device: torch.device = torch.device('cpu')) -> tuple[int, list, float | None, dict | None, dict | None]:
    """Load checkpoint and restore training state. Returns (start_epoch, loss_history, last_loss, model_cfg, training_args)."""
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return 0, [], float('inf'), None, None
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss_history = checkpoint.get('loss_history', [])
    last_loss = checkpoint.get('loss', float('inf'))
    model_config = checkpoint.get('model_config', None)
    training_args = checkpoint.get('training_args', None) # This would be the argparse Namespace dict
    
    print(f"Resumed from epoch {epoch}, last loss: {last_loss if last_loss is not None else 'N/A'}")
    return epoch, loss_history, last_loss, model_config, training_args

def find_latest_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    """Find the path to the latest checkpoint (latest_checkpoint.pth) in the directory."""
    if not checkpoint_dir.exists():
        return None
    
    latest_path = checkpoint_dir / "latest_checkpoint.pth"
    if latest_path.exists():
        return latest_path
    
    # Fallback: find highest epoch number if latest_checkpoint.pth is missing
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return checkpoints[-1] if checkpoints else None

