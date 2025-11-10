"""
sweep_and_train.py

A dedicated script to run hyperparameter sweeps for the diffusion model.
It iterates through a predefined list of configurations, launching a separate
training run for each.
"""

import sys
import argparse
from pathlib import Path
import copy

# Add the src directory to the path so we can import modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the main training function and argument parser from the original script
from src.diffusion.train import main as run_training_session, parse_args

# --- Define Hyperparameter Sweep Configurations ---
# Each dictionary represents one training run.
# Add or remove dictionaries to define your sweep.
# Parameters not specified here will use the defaults from `parse_args`.
SWEEP_CONFIGURATIONS = [
    {
        "description": "Baseline: Smaller model, default LR",
        "model_base_channels": 32,
        "model_num_res_blocks": 1,
        "learning_rate": 1e-4,
    },
    {
        "description": "Sweep 1: Increased model width",
        "model_base_channels": 64,  # Increased width
        "model_num_res_blocks": 1,
        "learning_rate": 1e-4,
    },
    {
        "description": "Sweep 2: Increased model depth",
        "model_base_channels": 32,
        "model_num_res_blocks": 2,  # Increased depth
        "learning_rate": 1e-4,
    },
    {
        "description": "Sweep 3: Increased width and depth",
        "model_base_channels": 64,
        "model_num_res_blocks": 2,
        "learning_rate": 1e-4,
    },
    {
        "description": "Sweep 4: Larger model with higher LR",
        "model_base_channels": 64,
        "model_num_res_blocks": 2,
        "learning_rate": 3e-4, # Higher learning rate
    },
]

# You can override general settings for all sweep runs here
# For example, to run all sweeps for fewer epochs than the default.
COMMON_SWEEP_SETTINGS = {
    "epochs": 25,  # Shorter epochs for quicker sweeps
    "checkpoint_freq": 1,  # Save checkpoint every epoch
    "save_best_model": True,  # Save the best model based on loss
    "log_images": True, # Ensure images are logged to tensorboard
    "resume_from_checkpoint": "none", # Start each sweep from scratch
}


def main():
    print(f"--- Starting Hyperparameter Sweep ---")
    print(f"Found {len(SWEEP_CONFIGURATIONS)} configurations to run.")
    
    # Get the default set of arguments from the main training script
    default_args = parse_args()

    for i, config in enumerate(SWEEP_CONFIGURATIONS):
        print(f"\n\n--- Running Sweep {i+1}/{len(SWEEP_CONFIGURATIONS)} ---")
        
        # Create a fresh copy of the default arguments for this run
        run_args = copy.deepcopy(default_args)
        
        # --- Update arguments with the sweep config ---
        # Update with common settings first
        for key, value in COMMON_SWEEP_SETTINGS.items():
            setattr(run_args, key, value)
        
        # Update with specific settings for this run
        description = "No description"
        for key, value in config.items():
            if key == "description":
                description = value
                continue
            setattr(run_args, key, value)
            
        print(f"Description: {description}")
        
        # --- Create unique directory and run names for this sweep ---
        run_name = f"sweep_{i+1:02d}_ch{run_args.model_base_channels}_res{run_args.model_num_res_blocks}_lr{run_args.learning_rate}"
        
        # Override checkpoint and tensorboard dirs to be sweep-specific
        run_args.checkpoint_dir = str(Path("data/diffusion/sweep_checkpoints") / run_name)
        run_args.tensorboard_run_name_prefix = run_name
        run_args.tensorboard_log_dir = str(Path("data/diffusion/sweep_tensorboard_logs"))

        print("Configuration for this run:")
        for key, value in sorted(vars(run_args).items()):
            print(f"  {key}: {value}")
        
        # --- Launch the training session ---
        try:
            run_training_session(run_args)
            print(f"--- Finished Sweep {i+1}/{len(SWEEP_CONFIGURATIONS)} ---")
        except Exception as e:
            print(f"!!! ERROR during sweep {i+1}: {description} !!!")
            print(f"Error: {e}")
            # Optionally, continue to the next sweep
            # raise e # Or uncomment to stop the whole sweep on error
    
    print("\n\n--- Hyperparameter Sweep Complete ---")

if __name__ == "__main__":
    main() 