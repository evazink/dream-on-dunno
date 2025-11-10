# Kitchen Environment Dataset

This module provides a PyTorch dataset for loading and processing frame pairs from the D4RL kitchen environment datasets.

## Features

- Loads data from multiple D4RL kitchen datasets:
  - `D4RL/kitchen/complete-v2` (Currently, only this one is processed in `dataloader.py`)
- Generates pairs of frames with configurable future step lookahead (k)
- Supports configurable target image size (e.g., 256x256)
- Automatically downloads and caches datasets
- Supports dataset rebuilding with `force_rebuild=True`
- Returns normalized tensors in CHW format

## Usage

To use `KitchenPairDataset` in your Python scripts:

```python
from src.data_utils.dataloader import KitchenPairDataset

# Load or create dataset with k=10 and target_size=256 (default values if run from CLI)
# When used as a module, you still need to specify these:
dataset = KitchenPairDataset(k_step_future=10, target_size=256)

# Load with different k values and target sizes
dataset_k3_128 = KitchenPairDataset(k_step_future=3, target_size=128)

# Force rebuild dataset
dataset_rebuild = KitchenPairDataset(k_step_future=10, target_size=256, force_rebuild=True)
```

### Running from Command Line

You can also run `dataloader.py` directly from the command line to pre-generate or inspect datasets.

```bash
python src/data_utils/dataloader.py [OPTIONS]
```

**Available Options:**

*   `--k K`: Number of steps to look ahead for the future frame (k_step_future). Default: 10.
*   `--target_size SIZE`: Target size to resize images to (e.g., 256 for 256x256). Default: 256.
*   `--data_dir PATH`: Directory to store/load the dataset. Default: `data/kitchen_pairs`.
*   `--force_rebuild`: If set, rebuilds the dataset even if a pre-existing one is found.

**Examples:**

```bash
# Generate/load dataset with k=5 and default target size (256x256)
python src/data_utils/dataloader.py --k 5

# Generate/load dataset with k=20 and target size 128x128
python src/data_utils/dataloader.py --k 20 --target_size 128

# Force rebuild dataset for k=10, target size 256x256, in a custom directory
python src/data_utils/dataloader.py --k 10 --target_size 256 --force_rebuild --data_dir data/my_custom_kitchen_data
```

## Dataset Structure

Each item in the dataset returns a tuple of:
- `current_frame`: Tensor of shape [3, H, W] normalized to [-1, 1]
- `future_frame`: Tensor of shape [3, H, W] normalized to [-1, 1]
- `dataset`: Name of the source dataset
- `episode_num`: Episode number
- `current_step`: Current timestep

## Data Processing

1. For each dataset:
   - Loads 5 episodes
   - Processes each episode to collect frames and transitions
   - Truncates at the last task transition
2. Creates frame pairs with k-step lookahead and resizes images to `target_size`.
3. Saves processed pairs to pickle files for faster loading.
4. Files are stored in `{data_dir}/pairs_k{k}_{target_size}x{target_size}.pkl`

## Example

```python
# Create DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Iterate over batches
for current_frames, future_frames, datasets, episodes, current_steps in dataloader:
    # current_frames: [B, 3, H, W]
    # future_frames: [B, 3, H, W]
    # datasets: List of dataset names
    # episodes: List of episode numbers
    # current_steps: List of timesteps
    ...
```

## Data Format

- **Input**: RGB images from the kitchen environment (480x480)
- **Output**: Pairs of frames (current, future)
- **Normalization**: Images are normalized to [-1,1] range
- **Format**: PyTorch tensors in CHW format (Channels, Height, Width)

## File Structure

```
data/
  kitchen_pairs/
    pairs_k10_256x256.pkl  # Dataset with k=10, target_size=256x256
    pairs_k5_256x256.pkl   # Dataset with k=5, target_size=256x256
    pairs_k20_128x128.pkl  # Dataset with k=20, target_size=128x128
```

## Implementation Details

- **Progress Tracking**: Uses tqdm to show progress during:
  - Dataset processing
  - Episode processing within each dataset
  - Frame pair creation
- **Episode Processing**:
  - Each episode is truncated at the last task completion
  - Frames are collected until either:
    - All 4 tasks are completed
    - Episode terminates
    - Episode is truncated
- **Frame Pairs**:
  - For each frame t, creates a pair with frame t+k
  - Uses the last frame for padding when t+k exceeds episode length
- **Image Resizing**:
  - Images are resized to `(target_size, target_size)` during the `_generate_pairs` phase before saving.

## Notes

- The dataset automatically handles padding at episode boundaries
- Each episode is truncated at the last task completion
- The number of pairs decreases slightly with larger k values
- Images are resized and then stored as pre-processed tensors (normalized to [-1, 1]) in the pickle file.
- Progress bars show:
  - Percentage complete
  - Number of items processed
  - Processing speed
  - Estimated time remaining 