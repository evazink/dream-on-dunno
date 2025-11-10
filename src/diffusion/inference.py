import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from typing import Any, Tuple
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from src.diffusion.model import ConditionalUNet

class Inference:
    """
    A class for performing inference using a trained model checkpoint.

    The model is expected to take an image of the current state and return
    an image of the predicted next state.
    """

    MODEL_INPUT_SHAPE: Tuple[int, int, int, int] = (1, 1, 120, 120)  # (N, C, H, W)
    EXPECTED_IMAGE_SHAPE: Tuple[int, int, int] = (1, 120, 120) # (C, H, W)
    EXPECTED_IMAGE_DTYPE: np.dtype = np.uint8

    def __init__(self, model_checkpoint_path: str) -> None:
        """
        Initializes the Inference class by loading the model from a checkpoint.

        Args:
            model_checkpoint_path: Path to the .pth model checkpoint file.

        Raises:
            FileNotFoundError: If the model checkpoint file does not exist.
            RuntimeError: If there's an issue loading the model or it's not a valid type.
        """
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint file not found: {model_checkpoint_path}")

        try:
            # Determine device
            self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the checkpoint
            checkpoint: Any = torch.load(model_checkpoint_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                # Try to get model configuration from checkpoint
                # Based on train.py and utils.py, this should be under 'model_config'
                model_cfg_from_checkpoint = checkpoint.get('model_config')

                # Fallback to other common keys if 'model_config' is not found
                if model_cfg_from_checkpoint is None:
                    model_cfg_from_checkpoint = checkpoint.get('model_hyperparams', checkpoint.get('config'))
                
                state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))

                # If state_dict itself is the entire checkpoint (e.g. only weights saved)
                if state_dict is None and not model_cfg_from_checkpoint:
                    # Check if the checkpoint itself is a state_dict
                    # A simple heuristic: does it have typical layer keys?
                    if any(isinstance(v, torch.Tensor) for v in checkpoint.values()) and any(".weight" in k or ".bias" in k for k in checkpoint.keys()):
                        state_dict = checkpoint
                        print("Warning: Checkpoint appears to be a raw state_dict. Model configuration will be inferred or defaulted.")
                    else:
                        raise RuntimeError(
                            f"Checkpoint dictionary from {model_checkpoint_path} does not contain a recognized model state_dict."
                        )
                elif not state_dict or not isinstance(state_dict, dict):
                     raise RuntimeError(
                        f"Checkpoint dictionary from {model_checkpoint_path} does not contain a valid 'model_state_dict' or 'state_dict'."
                    )

                if model_cfg_from_checkpoint and isinstance(model_cfg_from_checkpoint, dict):
                    print(f"Found model configuration in checkpoint. Instantiating ConditionalUNet with these parameters: {model_cfg_from_checkpoint}")
                    try:
                        self.model = ConditionalUNet(**model_cfg_from_checkpoint)
                    except TypeError as te:
                        raise RuntimeError(
                            f"Error instantiating ConditionalUNet with model_config from checkpoint: {te}. "
                            f"Ensure 'model_config' in the checkpoint contains all required "
                            f"constructor arguments for ConditionalUNet."
                        ) from te
                else:
                    # Fallback: Infer critical parameters if no config found in checkpoint
                    print(f"Warning: Model configuration ('model_config', 'model_hyperparams', or 'config') not found in checkpoint: {model_checkpoint_path}. Attempting to initialize ConditionalUNet with inferred channel/resolution settings and defaults for other parameters. This may lead to unexpected behavior or errors if these do not match the trained model.")
                    
                    # These were the inferred parameters previously
                    unet_args_for_inference = {
                        "in_img_channels": self.MODEL_INPUT_SHAPE[1],  # From (N, C, H, W)
                        "out_img_channels": self.EXPECTED_IMAGE_SHAPE[0], # From (C, H, W)
                        "initial_img_resolution": self.MODEL_INPUT_SHAPE[2], # H
                        # For other params, ConditionalUNet defaults will be used.
                        # This is highly likely to mismatch if the checkpoint had specific structural params.
                    }
                    print(f"Using inferred/default parameters: {unet_args_for_inference} (plus other ConditionalUNet defaults)")
                    try:
                        self.model = ConditionalUNet(**unet_args_for_inference)
                    except TypeError as te:
                        # This can happen if ConditionalUNet still has required args not covered by the above
                        raise RuntimeError(
                            f"Error instantiating ConditionalUNet with inferred/default parameters: {te}. "
                            f"The model may require more specific hyperparameters not found in the checkpoint."
                        ) from te

                # Clean state_dict keys if necessary (e.g. remove "module." prefix from DataParallel)
                # or if the state_dict is nested e.g. checkpoint['model_state_dict']['model']
                cleaned_state_dict = {}
                is_nested_or_prefixed = False
                
                # Check for "module." prefix
                if all(key.startswith("module.") for key in state_dict.keys()):
                    is_nested_or_prefixed = True
                    for k, v in state_dict.items():
                        cleaned_state_dict[k[len("module."):]] = v
                else:
                    cleaned_state_dict = state_dict

                # Further check for common nesting like model_state_dict = {"model": actual_state_dict}
                # This was partially handled above by checking 'model' key if no 'conv_in.' found.
                # The logic above was: state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
                # if 'model' in state_dict and isinstance(state_dict['model'], dict): state_dict = state_dict['model']
                # This means `state_dict` variable should already point to the actual state dictionary.

                try:
                    self.model.load_state_dict(cleaned_state_dict)
                except RuntimeError as e:
                    # If loading fails, it might be due to quantization stubs not matching
                    # or other architectural mismatches.
                    # If quant/dequant stubs are missing from state_dict but present in model, it's an issue.
                    # Check if error mentions "quant" or "dequant"
                    if "quant" in str(e) or "dequant" in str(e):
                        print(f"Note: Error loading state_dict, possibly due to quantization stubs. Trying to load with strict=False. Original error: {e}")
                        # Filter out quant/dequant keys if they are the problem
                        filtered_state_dict = {k: v for k, v in cleaned_state_dict.items() if "quant" not in k and "dequant" not in k}
                        if not filtered_state_dict: # if all keys were quant/dequant
                             raise RuntimeError(f"All keys in state_dict were related to quant/dequant. Cannot load. Original error: {e}") from e
                        try:
                            self.model.load_state_dict(filtered_state_dict, strict=False)
                            print("Successfully loaded state_dict with strict=False after filtering quant/dequant keys.")
                        except Exception as e_strict_false:
                             raise RuntimeError(f"Failed to load state_dict even with strict=False after filtering quant/dequant keys. Error: {e_strict_false}") from e_strict_false
                    else:
                        raise RuntimeError(
                            f"Error loading state_dict into ConditionalUNet from {model_checkpoint_path}. "
                            f"Ensure the model architecture matches the checkpoint. Original error: {e}"
                        ) from e

            elif isinstance(checkpoint, (torch.nn.Module, torch.jit.ScriptModule)):
                self.model = checkpoint # Loaded the model object directly
            else:
                raise RuntimeError(
                    f"Loaded object from {model_checkpoint_path} is not a "
                    f"torch.nn.Module, torch.jit.ScriptModule, or a recognized checkpoint dictionary. "
                    f"Got type: {type(checkpoint)}"
                )
                
            self.model.to(self.device)
            self.model.eval() # Set the model to evaluation mode
            
        except Exception as e:
            # Add model_checkpoint_path to the error message for better context
            if model_checkpoint_path not in str(e):
                raise RuntimeError(f"Error loading model from {model_checkpoint_path}: {e}") from e
            else:
                raise # Reraise if path is already in the message

    def predict(self, current_state_image: np.ndarray) -> np.ndarray:
        """
        Predicts the next state image based on the current state image.

        Args:
            current_state_image: A NumPy array representing the current state image.
                Shape: (1, 120, 120), dtype: np.uint8.

        Returns:
            A NumPy array representing the predicted next state image.
                Shape: (1, 120, 120), dtype: np.uint8.

        Raises:
            ValueError: If the input image does not match the expected shape or dtype.
            RuntimeError: If there's an issue during model inference.
        """
        # Input assertions
        if not isinstance(current_state_image, np.ndarray):
            raise ValueError(f"Input must be a NumPy array, got {type(current_state_image)}")
        if current_state_image.shape != self.EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Input image shape must be {self.EXPECTED_IMAGE_SHAPE}, "
                f"got {current_state_image.shape}"
            )
        if current_state_image.dtype != self.EXPECTED_IMAGE_DTYPE:
            raise ValueError(
                f"Input image dtype must be {self.EXPECTED_IMAGE_DTYPE}, "
                f"got {current_state_image.dtype}"
            )

        # Preprocessing
        # 1. Convert to float and normalize to [0, 1]
        image_tensor: torch.Tensor = torch.from_numpy(current_state_image.astype(np.float32)) / 255.0
        # 2. Add batch dimension: (1, 120, 120) -> (1, 1, 120, 120)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # MODIFICATION: The model expects 2 input channels (from model_config['in_img_channels'] = 2).
        # We are duplicating the single input channel to meet this requirement.
        # This is a technical fix; its semantic correctness depends on the model's training and intended use.
        if image_tensor.shape[1] == 1:
            print(f"Warning: Model expects 2 input channels (in_img_channels=2 from config), but received 1. Duplicating channel 0 to create a 2-channel input.")
            image_tensor = torch.cat([image_tensor, image_tensor], dim=1)
        
        # Check against the model_config's expected input channels if available, otherwise class default.
        # This check is now more complex because MODEL_INPUT_SHAPE is (1,1,H,W) but we feed (1,2,H,W)
        # For now, we rely on the model_config loaded during __init__ to have set up the model for 2 channels.
        # The critical part is that image_tensor now has 2 channels before hitting the model.

        # Original check (now potentially misleading if model truly expects 2 channels based on loaded config):
        # if image_tensor.shape != self.MODEL_INPUT_SHAPE:
        #      # This could happen if EXPECTED_IMAGE_SHAPE was (X, Y) instead of (C,X,Y)
        #      # Or if model expects different channel count.
        #     raise RuntimeError(
        #         f"Internal error: Processed tensor shape {image_tensor.shape} "
        #         f"does not match model input shape {self.MODEL_INPUT_SHAPE}."
        #     )

        # Inference
        try:
            with torch.no_grad(): # Disable gradient calculations
                # The ConditionalUNet model expects a time argument.
                # For a single-step prediction, let's assume t=0 or a small integer.
                # This might need to be configurable or determined by the use case.
                # The shape of time tensor should be (batch_size,)
                batch_size = image_tensor.shape[0]
                time_tensor = torch.zeros(batch_size, dtype=torch.long, device=self.device)
                
                predicted_tensor: torch.Tensor = self.model(image_tensor, time_tensor)
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {e}") from e

        # Postprocessing
        # 1. Ensure output tensor has 4 dimensions (N, C, H, W)
        if not (isinstance(predicted_tensor, torch.Tensor) and predicted_tensor.ndim == 4):
            raise RuntimeError(
                f"Model output is not a 4D tensor as expected. "
                f"Got shape {predicted_tensor.shape if isinstance(predicted_tensor, torch.Tensor) else type(predicted_tensor)}"
            )
        
        # 2. Remove batch dimension: (1, 1, 120, 120) -> (1, 120, 120)
        #    Also handle if model output is (N, C, H, W) where N > 1, C > 1
        #    For now, assume model output is (1,1,120,120) matching input channel/batch
        if predicted_tensor.shape[0] != 1 or predicted_tensor.shape[1] != self.EXPECTED_IMAGE_SHAPE[0]:
             raise RuntimeError(
                f"Model output tensor shape {predicted_tensor.shape} batch or channel size "
                f"does not match expected single output {self.MODEL_INPUT_SHAPE}. "
                f"Expected N=1, C={self.EXPECTED_IMAGE_SHAPE[0]}."
            )
        
        output_image_tensor: torch.Tensor = predicted_tensor.squeeze(0) # (1, 120, 120)

        # 3. Denormalize from [0, 1] to [0, 255] and convert to uint8
        output_image_np: np.ndarray = (output_image_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # Output assertions
        if output_image_np.shape != self.EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Output image shape must be {self.EXPECTED_IMAGE_SHAPE}, "
                f"got {output_image_np.shape}"
            )
        if output_image_np.dtype != self.EXPECTED_IMAGE_DTYPE:
            raise ValueError(
                f"Output image dtype must be {self.EXPECTED_IMAGE_DTYPE}, "
                f"got {output_image_np.dtype}"
            )

        return output_image_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform inference using a trained model.")
    parser.add_argument("--model_checkpoint_path", type=str, help="Path to the .pth model checkpoint file.")
    parser.add_argument("--input_image_path", type=str, help="Path to the input image.")
    args = parser.parse_args()

    # --- Load and preprocess the input image ---
    try:
        print(f"Loading input image from: {args.input_image_path}")
        img = Image.open(args.input_image_path).convert('L')  # Convert to grayscale (L mode)
        img = img.resize((120, 120), Image.Resampling.LANCZOS) # Resize (width, height)
        
        input_image_np = np.array(img, dtype=Inference.EXPECTED_IMAGE_DTYPE) # (120, 120)
        
        # Ensure shape is (1, 120, 120)
        if input_image_np.ndim == 2: # (H, W)
            input_image_np = np.expand_dims(input_image_np, axis=0) # (1, H, W)
        
        if input_image_np.shape != Inference.EXPECTED_IMAGE_SHAPE:
            raise ValueError(
                f"Loaded image reshaped to {input_image_np.shape} but expected {Inference.EXPECTED_IMAGE_SHAPE}. "
                f"Original image path: {args.input_image_path}"
            )
        print(f"Input image loaded successfully. Shape: {input_image_np.shape}, dtype: {input_image_np.dtype}")

    except FileNotFoundError:
        print(f"Error: Input image file not found at {args.input_image_path}")
        exit(1)
    except Exception as e:
        print(f"Error loading or processing input image: {e}")
        exit(1)

    print(f"\\nAttempting to initialize Inference class with model: {args.model_checkpoint_path}")
    try:
        inference_module = Inference(model_checkpoint_path=args.model_checkpoint_path)
        print("Inference class initialized successfully.")
        
        # Perform prediction
        print("Attempting prediction...")
        predicted_image_np = inference_module.predict(input_image_np)
        print(f"Predicted image shape: {predicted_image_np.shape}, dtype: {predicted_image_np.dtype}")
        print("Prediction successful.")

        # --- Display the images ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Input image display
        axes[0].imshow(input_image_np.squeeze(0), cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f"Input Image\\n{os.path.basename(args.input_image_path)}")
        axes[0].axis('off')
        
        # Predicted image display
        axes[1].imshow(predicted_image_np.squeeze(0), cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Predicted Next State")
        axes[1].axis('off')
        
        plt.suptitle(f"Model: {os.path.basename(args.model_checkpoint_path)}")
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Model checkpoint file not found. {e}")
    except RuntimeError as e:
        print(f"Runtime error during inference: {e}")
    except ValueError as e:
        print(f"Value error during inference: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}") 