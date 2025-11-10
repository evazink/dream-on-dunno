# Diffusion Model for Kitchen Environment Frame Prediction

This module implements a conditional diffusion model that learns to predict future frames in the D4RL kitchen environment. The model takes a current frame and predicts what the scene will look like in the next few timesteps.

## Overview

The diffusion model is trained on pairs of frames from kitchen manipulation episodes:
- **Input**: Current frame (128×128×3)
- **Output**: Predicted future frame (128×128×3)
- **Architecture**: U-Net with time embeddings and attention layers
- **Training**: DDPM (Denoising Diffusion Probabilistic Models)

## Files Description

- `train.py` - Main training script with checkpointing and TensorBoard logging
- `model.py` - U-Net architecture implementation
- `utils.py` - Utility functions for noise scheduling and diffusion process
- `inference_demo.py` - Interactive demo showing real-time frame prediction
- `README.md` - This file

## Prerequisites

### Dependencies
```bash
# Core dependencies (should already be installed)
torch
torchvision
numpy
matplotlib
tqdm
minari
gymnasium
gymnasium-robotics
mujoco
tensorboard
```

### Hardware Requirements
- **Apple Silicon Mac**: Uses MPS (Metal Performance Shaders) for acceleration
- **Memory**: ~4-8GB RAM for training with optimized settings
- **Storage**: ~500MB for dataset, ~100MB for model checkpoints

## Quick Start

### 1. Train the Model

```bash
# Navigate to the project root
cd /path/to/self-play

# Start training (will create optimized dataset on first run)
python src/diffusion/train.py
```

**First run will take ~5-10 minutes** to:
1. Download D4RL kitchen dataset
2. Process episodes and create frame pairs
3. Resize images to 128×128 and pre-process to tensors
4. Save optimized dataset for future use

**Subsequent training runs** load the pre-processed dataset instantly.

### 2. Monitor Training

```bash
# In a separate terminal, start TensorBoard
tensorboard --logdir=data/diffusion/tensorboard_logs

# Open browser to: http://localhost:6006
```

TensorBoard shows:
- Training loss curves
- Sample images and predictions
- Gradient norms
- Model architecture

### 3. Run Inference Demo

```bash
# After training completes (or with existing checkpoint)
python src/diffusion/inference_demo.py
```

This opens an interactive window showing:
- **Left**: Current frame from environment
- **Right**: Model's predicted next frame

## Training Configuration

### Key Parameters (in `train.py`)
```python
IMG_RESOLUTION = 128        # Image size (128×128)
BATCH_SIZE = 2             # Batch size (optimized for MPS)
LEARNING_RATE = 1e-4       # Adam learning rate
EPOCHS = 100               # Training epochs
TIMESTEPS = 1000           # Diffusion timesteps
```

### Model Architecture
```python
base_channels = 32         # U-Net base channels
channel_mults = (1, 2, 3)  # Channel multipliers per level
num_res_blocks = 1         # Residual blocks per level
attention_resolutions = (32,) # Where to apply attention
```

### Data Configuration
```python
k_step_future = 5          # Predict 5 steps into future
target_size = 128          # Resize images to 128×128
max_episodes_per_batch = 5 # Episode processing batch size
```

## Training Process

### Phase 1: Dataset Creation (First Run Only)
```
Processing dataset: D4RL/kitchen/complete-v2
Total episodes in dataset: 19
Processing episodes in batches...
Added 611 pairs from batch 0-5
Added 597 pairs from batch 5-10
...
Total pairs created: 3809
Converting to tensors and pre-processing images...
Tensor conversion complete!
```

### Phase 2: Training Loop
```
Training started at: 2025-01-24 15:30:45
Epoch [1/100] - Avg Loss: 0.125000 - Elapsed: 0:02:15
Epoch [2/100] - Avg Loss: 0.098234 - Elapsed: 0:04:32
...
Epoch [50/100] - Avg Loss: 0.012345 - Elapsed: 1:23:10 *** NEW BEST ***
```

### Phase 3: Checkpointing
- Saves checkpoints every 5 epochs
- Saves best model automatically
- Saves latest checkpoint for resuming

## Expected Training Time

| Phase | Time | Description |
|-------|------|-------------|
| **Dataset Creation** | ~5-10 min | One-time setup |
| **Training (100 epochs)** | ~30-60 min | On Apple Silicon |
| **Total First Run** | ~40-70 min | Including setup |
| **Subsequent Runs** | ~30-60 min | Dataset pre-loaded |

## Inference Demo

### Features
- **Real-time prediction**: Shows model output for each frame
- **Side-by-side comparison**: Environment vs. predicted frame
- **Timing information**: Inference speed per frame
- **Interactive visualization**: Matplotlib window updates in real-time

### Configuration (in `inference_demo.py`)
```python
CHECKPOINT_PATH = "data/diffusion/checkpoints/best_model.pth"
DATASET_NAME = 'D4RL/kitchen/complete-v2'
EPISODE_IDX = 0        # Which episode to demo (0-18)
MAX_STEPS = 30         # How many steps to process
```

### Expected Output
```
Using device: mps
Loading checkpoint: data/diffusion/checkpoints/best_model.pth
Loaded model from epoch 99
Training loss: 0.003108
Model loaded successfully!

Loading dataset: D4RL/kitchen/complete-v2
Running inference demo on episode 0
Episode length: 208 steps
Will process first 30 steps
Processing steps: 100%|██████████| 30/30 [00:15<00:00, 1.95it/s]
Demo completed!
```

## File Structure

```
data/
├── diffusion/
│   ├── checkpoints/          # Model checkpoints
│   │   ├── best_model.pth   # Best model (lowest loss)
│   │   ├── latest_checkpoint.pth
│   │   └── checkpoint_epoch_*.pth
│   └── tensorboard_logs/     # TensorBoard logs
├── kitchen_pairs/            # Processed dataset
│   └── pairs_k5_128x128.pkl # Optimized frame pairs
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues
```bash
# Error: Out of memory during training
```
**Solution**: Reduce batch size in `train.py`:
```python
BATCH_SIZE = 1  # Reduce from 2 to 1
```

#### 2. Numpy Stride Error
```bash
# Error: negative strides not supported
```
**Solution**: Already fixed with `.copy()` in preprocessing

#### 3. Missing Checkpoint
```bash
# Error: Checkpoint not found
```
**Solution**: Train model first or update checkpoint path:
```python
CHECKPOINT_PATH = "data/diffusion/checkpoints/checkpoint_epoch_050.pth"
```

#### 4. Dataset Not Found
```bash
# Error: Dataset file not found
```
**Solution**: Delete old dataset and retrain:
```bash
rm data/kitchen_pairs/pairs_k5_*.pkl
python src/diffusion/train.py  # Will recreate dataset
```

### Performance Tips

1. **First Training Run**: Be patient during dataset creation
2. **Subsequent Runs**: Should be much faster with pre-processed data
3. **Memory Optimization**: Use single-threaded DataLoader on Mac
4. **TensorBoard**: Monitor training progress for early stopping

## Advanced Usage

### Resume Training
```python
# Training automatically detects existing checkpoints
# Responds 'y' when prompted to resume from latest checkpoint
```

### Custom Episode Selection
```python
# In inference_demo.py, try different episodes:
EPISODE_IDX = 5  # Try episode 5 instead of 0
MAX_STEPS = 50   # Process more steps
```

### Model Variants
```python
# In train.py, experiment with architecture:
base_channels = 64           # Larger model
channel_mults = (1, 2, 3, 4) # Deeper U-Net
k_step_future = 10           # Predict further ahead
```

## Results

### Expected Training Results
- **Initial Loss**: ~0.5-1.0
- **Final Loss**: ~0.001-0.01 (after 100 epochs)
- **Training Time**: ~30-60 minutes on Apple Silicon

### Expected Inference Results
- **Inference Speed**: ~0.02-0.1 seconds per frame
- **Visual Quality**: Model should predict basic object movements and scene changes
- **Temporal Consistency**: Predictions should be temporally coherent

## Next Steps

1. **Experiment with longer predictions**: Increase `k_step_future`
2. **Try different architectures**: Modify U-Net parameters
3. **Add conditioning**: Condition on actions or task information
4. **Higher resolution**: Train on 256×256 images (requires more memory)
5. **Video generation**: Chain predictions for longer sequences

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your environment has all dependencies
3. Monitor TensorBoard for training diagnostics
4. Try reducing batch size or image resolution for memory issues 

# Diffusion Model Training (`train.py`)

This script trains a Conditional U-Net diffusion model on (grayscale) image pairs from the Kitchen environment dataset.

## Features

- Trains a diffusion model to predict noise.
- Uses grayscale images (1-channel input, 1-channel output for noise).
- Supports command-line arguments for configuring most aspects of training, model, data, and logging.
- Checkpointing with support for resuming from latest or best checkpoint.
- TensorBoard logging for metrics, image samples, and hyperparameters.
- Automatic Mixed Precision (AMP) support for CUDA devices.

## Prerequisites

1.  **Dataset:** Ensure you have generated the grayscale image pair dataset using `src/data_utils/dataloader.py`.
    ```bash
    python src/data_utils/dataloader.py --force_rebuild --k_step_future 5 --target_size 128 --data_dir data/kitchen_pairs_grayscale_k5_128
    ```
    (Adjust `k_step_future`, `target_size`, and `data_dir` to match your desired training configuration. The defaults in `train.py` for dataset loading are `k_step_future=5`, `img_resolution=128`, and `dataset_dir="data/kitchen_pairs"`. You might want to use a more specific dataset directory for grayscale data, e.g., `data/kitchen_pairs_grayscale_k5_128`).

2.  **Dependencies:** Make sure all dependencies from `requirements.txt` are installed.

## Running Training

Execute the script from the root directory of the project:

```bash
python src/diffusion/train.py [OPTIONS]
```

### Example Usage

**Start new training with default parameters (MPS or CPU):**
```bash
python src/diffusion/train.py --img_resolution 128 --batch_size 4 --epochs 50 --k_step_future 5 --dataset_dir data/kitchen_pairs_grayscale_k5_128 --tensorboard_log_dir data/diffusion/tb_logs_gray --checkpoint_dir data/diffusion/ckpt_gray
```

**Train on CUDA with AMP enabled:**
```bash
python src/diffusion/train.py --device cuda --img_resolution 128 --batch_size 8 --epochs 100 --learning_rate 0.0002 --k_step_future 5 --dataset_dir path/to/your/grayscale_dataset --log_images
```

**Resume training from the latest checkpoint:**
```bash
python src/diffusion/train.py --resume_from_checkpoint latest --checkpoint_dir data/diffusion/ckpt_gray
# (Ensure other parameters like model architecture match the checkpoint if not using defaults)
```

**Specify model architecture details:**
```bash
python src/diffusion/train.py \
    --model_base_channels 64 \
    --model_channel_mults "1,2,4,4" \
    --model_num_res_blocks 2 \
    --model_attention_resolutions_divisor 8 \
    --img_resolution 256 \
    # ... other args ...
```

### Command-Line Arguments

**Environment & Data:**
*   `--device {cpu,mps,cuda}`: Device for training. Default: mps or cpu.
*   `--dataset_dir STR`: Root directory for the dataset. Default: `data/kitchen_pairs`.
*   `--k_step_future INT`: k-step future for KitchenPairDataset. Default: 5.
*   `--force_rebuild_dataset`: Force rebuild of the dataset. Default: False.
*   `--num_workers_loader INT`: Number of DataLoader workers. Default: 0.

**Training Hyperparameters:**
*   `--epochs INT`: Number of training epochs. Default: 100.
*   `--batch_size INT`: Batch size. Default: 2.
*   `--learning_rate FLOAT`: Learning rate. Default: 1e-4.
*   `--disable_amp`: Disable AMP even if on CUDA. Default: False (AMP enabled on CUDA).

**Diffusion Process:**
*   `--diffusion_timesteps INT`: Number of diffusion timesteps. Default: 1000.
*   `--beta_start FLOAT`: Beta start for linear schedule. Default: 0.0001.
*   `--beta_end FLOAT`: Beta end for linear schedule. Default: 0.02.

**Model Architecture (Grayscale: `in_img_channels` fixed to 2, `out_img_channels` fixed to 1):**
*   `--img_resolution INT`: Image resolution for training and model. Default: 128.
*   `--model_base_channels INT`: Base channels for U-Net. Default: 32.
*   `--model_channel_mults STR`: Channel multipliers (e.g., '1,2,3'). Default: "1,2,3".
*   `--model_num_res_blocks INT`: Residual blocks per U-Net level. Default: 1.
*   `--model_attention_resolutions_divisor INT`: Attention at `img_resolution // X`. Default: 4.
*   `--model_time_emb_dim INT`: Dimension of sinusoidal time embedding. Default: 128.
*   `--model_time_emb_mlp_dim INT`: Dimension of MLP for time embedding. Default: 512.
*   `--model_dropout_rate FLOAT`: Dropout rate in U-Net residual blocks. Default: 0.1.

**Checkpointing & Resuming:**
*   `--checkpoint_dir STR`: Directory to save checkpoints. Default: `data/diffusion/checkpoints_grayscale`.
*   `--checkpoint_freq INT`: Save checkpoint every N epochs. Default: 5.
*   `--save_best_model`: Save model with best validation loss (currently best training loss). Default: False.
*   `--resume_from_checkpoint {latest,best,none,PATH}`: Checkpoint to resume from. Default: `latest`.

**TensorBoard Logging:**
*   `--tensorboard_log_dir STR`: Directory for TensorBoard logs. Default: `data/diffusion/tensorboard_logs_grayscale`.
*   `--tensorboard_run_name_prefix STR`: Prefix for TensorBoard run name. Default: `diffusion_grayscale_train`.
*   `--log_freq INT`: Log scalars every N batches. Default: 10.
*   `--log_images`: Log sample images. Default: False.
*   `--log_image_freq_multiplier INT`: Log images every (log_freq * M) steps. Default: 10.

## Important Notes for Grayscale Training

- The `ConditionalUNet` model is configured with `in_img_channels=2` (concatenated current and future/noisy grayscale frames) and `out_img_channels=1` (predicting noise for a single grayscale channel).
- Ensure your dataset in `dataset_dir` is generated with grayscale images (1 channel) and normalized tensors, as produced by the updated `src/data_utils/dataloader.py`.
- When resuming from a checkpoint, ensure that the model architecture parameters specified (or their defaults) match the architecture stored in the checkpoint if you expect the weights to load correctly. 