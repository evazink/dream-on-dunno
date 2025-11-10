#!/usr/bin/env python3
"""
Helper script to launch TensorBoard for monitoring diffusion training.
"""

import subprocess
import sys
from pathlib import Path

def launch_tensorboard():
    """Launch TensorBoard with the correct log directory."""
    log_dir = Path("data/diffusion/tensorboard_logs")
    
    if not log_dir.exists():
        print(f"TensorBoard log directory does not exist: {log_dir}")
        print("Please run training first to generate logs.")
        return
    
    print(f"Launching TensorBoard with log directory: {log_dir}")
    print("TensorBoard will be available at: http://localhost:6006")
    print("Press Ctrl+C to stop TensorBoard")
    
    try:
        # Launch TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main", 
            "--logdir", str(log_dir),
            "--host", "localhost",
            "--port", "6006"
        ])
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print("TensorBoard not found. Install it with: pip install tensorboard")
    except Exception as e:
        print(f"Error launching TensorBoard: {e}")

if __name__ == "__main__":
    launch_tensorboard() 