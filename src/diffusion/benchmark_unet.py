import argparse
import time
import torch
import os
from pathlib import Path
import torch.profiler # Added for profiling
from torch.cuda.amp import autocast # Added for AMP
import torch.quantization # Added for PTQ

# Assuming this script is in src/diffusion/, and model.py is in the same directory
# and dataloader.py is in src/data_utils/
from .model import ConditionalUNet
from ..data_utils.dataloader import KitchenPairDataset
from torch.utils.data import DataLoader

def get_model_size_info(model: torch.nn.Module) -> tuple[float, int]:
    """Calculates model size in MB and number of parameters.

    Args:
        model: The PyTorch model.

    Returns:
        A tuple containing:
            - size_all_mb (float): Total model size in megabytes.
            - num_params (int): Total number of parameters.
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Count only trainable parameters
    return size_all_mb, num_params

def main(args: argparse.Namespace) -> None:
    """Main function to run the benchmark."""
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Initialize Model
    print("Initializing model...")
    # Ensure channel_mults and attention_resolutions are correctly parsed
    channel_mults_tuple = tuple(map(int, args.model_channel_mults.split(',')))
    attention_resolutions_tuple = tuple(map(int, args.model_attention_resolutions.split(',')))

    model = ConditionalUNet(
        in_img_channels=args.model_in_channels,
        out_img_channels=args.model_out_channels,
        base_channels=args.model_base_channels,
        channel_mults=channel_mults_tuple,
        num_res_blocks_per_level=args.model_num_res_blocks,
        attention_resolutions=attention_resolutions_tuple,
        dropout_rate=args.model_dropout_rate, # Added dropout_rate
        time_emb_dim=args.model_time_emb_dim,
        time_emb_mlp_dim=args.model_time_emb_mlp_dim,
        initial_img_resolution=args.target_size_dataloader # Critical: must match data
    ).to(device)
    model.eval()

    model_size_mb, num_params = get_model_size_info(model)
    print(f"Model initialized: {num_params:,} trainable parameters, Size: {model_size_mb:.2f} MB (FP32)")

    if args.model_checkpoint_path and not args.quantize_ptq_static: # Dont load checkpoint if we are about to quantize from scratch effectively
        checkpoint_path = Path(args.model_checkpoint_path)
        if checkpoint_path.exists():
            try:
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                print(f"Loaded model weights from {args.model_checkpoint_path}")
            except Exception as e:
                print(f"Error loading checkpoint {args.model_checkpoint_path}: {e}. Using random weights.")
        else:
            print(f"Warning: Checkpoint path {args.model_checkpoint_path} not found. Using random weights.")

    # 2. Initialize DataLoader
    print(f"Loading dataset with k={args.k_dataloader}, target_size={args.target_size_dataloader}x{args.target_size_dataloader} from {args.data_dir_dataloader}")
    try:
        dataset = KitchenPairDataset(
            k_step_future=args.k_dataloader,
            data_dir=args.data_dir_dataloader,
            target_size=args.target_size_dataloader,
            force_rebuild=args.force_rebuild_dataloader 
        )
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        print("Please ensure the dataset is pre-generated using src/data_utils/dataloader.py")
        print(f"Example: python src/data_utils/dataloader.py --k {args.k_dataloader} --target_size {args.target_size_dataloader} --data_dir {args.data_dir_dataloader}")
        return

    if len(dataset) == 0:
        print("Dataset is empty. Please generate data first using src/data_utils/dataloader.py.")
        return
    
    if args.num_batches > (len(dataset) // args.batch_size) and (len(dataset) % args.batch_size !=0) :
         print(f"Warning: num_batches ({args.num_batches}) is greater than the number of available batches in the dataset ({len(dataset)//args.batch_size}).")
         print(f"Will run for a maximum of {len(dataset)//args.batch_size} batches.")


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False, # No need to shuffle for benchmarking
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True # Drop last incomplete batch for consistent batch sizes in benchmark
    )
    
    if len(dataloader) == 0:
        print(f"Dataloader is empty. This might be due to batch_size ({args.batch_size}) being larger than dataset size ({len(dataset)}).")
        return

    # 3. Benchmarking and Profiling loop
    total_time_seconds = 0.0
    total_items_processed = 0
    actual_batches_run_for_timing = 0

    # Determine activities for profiler based on device
    profiler_activities = [torch.profiler.ProfilerActivity.CPU]
    if 'cuda' in args.device:
        profiler_activities.append(torch.profiler.ProfilerActivity.CUDA)

    # Setup profiler schedule if profiling is enabled
    prof_schedule = None
    amp_enabled_for_cuda = args.amp and 'cuda' in args.device and not args.quantize_ptq_static
    if amp_enabled_for_cuda:
        print("Automatic Mixed Precision (AMP) enabled for CUDA.")

    run_profiler = args.profile and not args.quantize_ptq_static
    if run_profiler:
        prof_schedule = torch.profiler.schedule(
            wait=args.profile_schedule_wait,
            warmup=args.profile_schedule_warmup,
            active=args.profile_schedule_active,
            repeat=args.profile_schedule_repeat
        )
        print(f"Profiling enabled. Output will be saved to {args.profile_output_dir}")
        Path(args.profile_output_dir).mkdir(parents=True, exist_ok=True)

    def trace_handler(p):
        output_file = Path(args.profile_output_dir) / f"unet_benchmark_trace_{p.step_num}.json"
        p.export_chrome_trace(str(output_file))
        print(f"Profiler trace saved to {output_file}")
        print(f"To view: Open Chrome browser and go to chrome://tracing, then load the .json file.")
        # Or use TensorBoard: tensorboard --logdir {args.profile_output_dir}
        # To export for TensorBoard:
        # p.export_stacks(f"{args.profile_output_dir}/profiler_stacks_{p.step_num}.txt", "self_cuda_time_total")
        # print(f"Stack info saved to {args.profile_output_dir}/profiler_stacks_{p.step_num}.txt")

    # --- PTQ Static Quantization Steps ---
    if args.quantize_ptq_static:
        print("\n--- Applying Post-Training Static Quantization (PTQ) ---")
        
        # For CPU-based backends like fbgemm/qnnpack, quantization should happen on CPU.
        # The benchmark for the PTQ model will also run on CPU.
        original_device_for_benchmark = device # Save the original device for final report if not CPU
        model_on_cpu_for_ptq = False
        if device.type != 'cpu':
            print(f"Warning: PTQ static quantization with backend '{args.ptq_backend}' is CPU-focused. Temporarily moving model to CPU for PTQ steps.")
            model.to('cpu')
            model_on_cpu_for_ptq = True # Flag that model was moved
        
        # Ensure model is in eval mode for quantization
        model.eval()

        # Specify quantization configuration
        # For x86, 'fbgemm' is common. For ARM, 'qnnpack'.
        # We'll default to fbgemm. User might need to change this based on target.
        qconfig_backend = 'fbgemm' # or 'qnnpack'
        # Check if 'qnnpack' is available and preferred (e.g., if on ARM or explicitly chosen)
        # For simplicity, we stick to fbgemm. A more robust script might choose based on platform.
        if args.ptq_backend:
            qconfig_backend = args.ptq_backend

        if torch.backends.quantized.engine != qconfig_backend:
            print(f"Warning: Default PyTorch quantized engine is {torch.backends.quantized.engine}. Attempting to set to {qconfig_backend}.")
            try:
                torch.backends.quantized.engine = qconfig_backend
                print(f"Successfully set quantized engine to {qconfig_backend}.")
            except Exception as e:
                print(f"Error setting quantized engine to {qconfig_backend}: {e}. Using default {torch.backends.quantized.engine}.")
                qconfig_backend = torch.backends.quantized.engine
        else:
            print(f"Using PyTorch default quantized engine: {qconfig_backend}.")
        
        model.qconfig = torch.quantization.get_default_qconfig(qconfig_backend)
        print(f"Using qconfig: {model.qconfig} with backend {qconfig_backend}")

        # Prepare the model for static quantization. This inserts observers.
        torch.quantization.prepare(model, inplace=True)
        print("Model prepared for static quantization (observers inserted).")

        # Calibration step
        print(f"Running calibration with {args.ptq_num_calibration_batches} batches...")
        with torch.no_grad():
            for i, batch_data in enumerate(dataloader):
                if i >= args.ptq_num_calibration_batches:
                    break
                current_frames, future_frames, _, episode_current_steps, _ = batch_data
                # Ensure data is on CPU if model was moved to CPU for PTQ
                ptq_device = 'cpu' if model_on_cpu_for_ptq else device
                x_concat = torch.cat((future_frames, current_frames), dim=1).to(ptq_device) 
                time_input = episode_current_steps.float().to(ptq_device)
                model(x_concat, time_input)
                if (i+1) % 1 == 0:
                    print(f"  Calibration batch {i+1}/{args.ptq_num_calibration_batches} completed.")
        print("Calibration complete.")

        # Convert the model to a quantized version
        torch.quantization.convert(model, inplace=True)
        print("Model converted to quantized version (INT8).")
        
        # Restore model to original device if it was moved for PTQ, unless benchmark is forced to CPU
        if model_on_cpu_for_ptq:
            if args.benchmark_ptq_on_cpu:
                print(f"Benchmarking PTQ model on CPU as requested by --benchmark_ptq_on_cpu.")
                device = torch.device('cpu') # Ensure device for benchmark is CPU
            else:
                print(f"Moving PTQ model back to original device: {original_device_for_benchmark} for benchmarking.")
                model.to(original_device_for_benchmark)
                device = original_device_for_benchmark # Ensure device for benchmark is the original
        elif args.benchmark_ptq_on_cpu and device.type != 'cpu': # Model was not moved, but user wants CPU benchmark
             print(f"Warning: Model was not on CPU for PTQ, but --benchmark_ptq_on_cpu is set. Moving to CPU for benchmark.")
             model.to('cpu')
             device = torch.device('cpu')

        # Re-calculate model size for the quantized model
        # Note: get_model_size_info might not be perfectly accurate for quantized models
        # as it sums parameter element_size which might still report FP32 for packed params.
        # True size reduction is best seen by saving the model and checking file size.
        model_size_mb, num_params = get_model_size_info(model)
        print(f"Quantized model: {num_params:,} effective parameters, Approx. Size: {model_size_mb:.2f} MB")
        print("Note: Reported size for quantized model is an estimate. True size reduction visible after saving.")

    # Main loop for benchmarking and profiling
    with torch.no_grad(), \
         (torch.profiler.profile(
             activities=profiler_activities,
             schedule=prof_schedule,
             on_trace_ready=trace_handler if run_profiler else None,
             record_shapes=run_profiler, 
             profile_memory=run_profiler,    
             with_stack=run_profiler        
         ) if run_profiler else torch.profiler.ExecutionTraceObserver()) as prof:

        # Warm-up runs (distinct from profiler warmup)
        if args.num_warmup_batches > 0 and not (run_profiler and prof_schedule and args.profile_schedule_warmup > 0):
            print(f"Running {args.num_warmup_batches} warm-up batches (manual)...")
            warmup_batches_done = 0
            for batch_idx, batch_data in enumerate(dataloader):
                if warmup_batches_done >= args.num_warmup_batches:
                    break
                current_frames, future_frames, _, episode_current_steps, _ = batch_data
                x_concat = torch.cat((future_frames, current_frames), dim=1).to(device)
                time_input = episode_current_steps.float().to(device)
                _ = model(x_concat, time_input)
                warmup_batches_done += 1
            if 'cuda' in args.device: torch.cuda.synchronize()
            print(f"Manual warm-up complete ({warmup_batches_done} batches run).")

        print(f"Running benchmark for up to {args.num_batches} batches...")
        for batch_idx, batch_data in enumerate(dataloader):
            if actual_batches_run_for_timing >= args.num_batches and not run_profiler: # If not profiling, respect num_batches for timing
                break
            if run_profiler and prof.step_num >= (args.profile_schedule_wait + args.profile_schedule_warmup + args.profile_schedule_active) * args.profile_schedule_repeat:
                 break # Stop if profiling cycle is complete

            current_frames, future_frames, _, episode_current_steps, _ = batch_data
            x_concat = torch.cat((future_frames, current_frames), dim=1).to(device)
            time_input = episode_current_steps.float().to(device)

            # Timing for non-profiling active steps, or all steps if not profiling
            should_time_batch = not run_profiler or \
                                (run_profiler and prof_schedule and 
                                 (args.profile_schedule_wait + args.profile_schedule_warmup) * prof.repeat_n < prof.step_num <= (args.profile_schedule_wait + args.profile_schedule_warmup + args.profile_schedule_active) * prof.repeat_n)
            
            if should_time_batch and actual_batches_run_for_timing >= args.num_batches:
                 # Ensure we don't exceed num_batches for timing purposes even if profiler runs longer
                 if not run_profiler: break
                 # If profiling, let it continue for its schedule but stop timing here for benchmark stats

            start_event, end_event = None, None
            if 'cuda' in args.device:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                cpu_start_time = time.perf_counter()

            # Apply autocast if AMP is enabled for CUDA
            if amp_enabled_for_cuda:
                with autocast():
                    _ = model(x_concat, time_input)
            else:
                _ = model(x_concat, time_input)

            batch_time_seconds = 0
            if 'cuda' in args.device:
                end_event.record()
                torch.cuda.synchronize()
                batch_time_seconds = start_event.elapsed_time(end_event) / 1000.0
            else:
                cpu_end_time = time.perf_counter()
                batch_time_seconds = cpu_end_time - cpu_start_time
            
            if should_time_batch and actual_batches_run_for_timing < args.num_batches:
                total_time_seconds += batch_time_seconds
                total_items_processed += x_concat.size(0)
                actual_batches_run_for_timing += 1
            
            if (batch_idx + 1) % args.print_freq == 0:
                print_batch_num = actual_batches_run_for_timing if should_time_batch and actual_batches_run_for_timing <= args.num_batches else batch_idx + 1
                print_num_batches_total = args.num_batches if not run_profiler else "profiler-steps"
                print(f"Batch {print_batch_num}/{print_num_batches_total}: Time {batch_time_seconds:.4f}s")

            if run_profiler:
                prof.step() # Signal profiler for next step
    
    if actual_batches_run_for_timing == 0 and not run_profiler:
        print("No batches were timed for benchmarking. Check dataset, num_batches, or batch_size arguments.")
        # If profiling was on, it might have completed without error, so don't return early if traces were generated.
        if not (run_profiler and (Path(args.profile_output_dir)).glob('*.json')):
             return

    avg_time_per_batch = total_time_seconds / actual_batches_run_for_timing if actual_batches_run_for_timing > 0 else 0
    images_per_second = total_items_processed / total_time_seconds if total_time_seconds > 0 else 0

    benchmark_type = "FP32"
    if amp_enabled_for_cuda:
        benchmark_type = "AMP FP16/BF16 (CUDA)"
    elif args.quantize_ptq_static:
        benchmark_type = f"PTQ Static INT8 ({qconfig_backend} backend on CPU)" # Clarify backend and CPU execution

    print(f"\n--- Model Benchmark Results ({benchmark_type}) ---")
    # For PTQ, device was changed to CPU. For others, it's the original model device.
    final_benchmark_device = device if args.quantize_ptq_static else next(model.parameters()).device
    print(f"Device used for benchmark: {final_benchmark_device}")
    print(f"Model Trainable Parameters: {num_params:,}")
    print(f"Model Size (parameters + buffers): {model_size_mb:.2f} MB")
    print(f"Input Image Resolution: {args.target_size_dataloader}x{args.target_size_dataloader}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Batches Timed for Benchmark: {actual_batches_run_for_timing}")
    print(f"Average Inference Time per Batch: {avg_time_per_batch:.4f} seconds")
    print(f"Throughput: {images_per_second:.2f} images/second")
    print("------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ConditionalUNet FP32 Performance")
    
    # --- Dataset Arguments ---
    parser.add_argument("--k_dataloader", type=int, default=10, help="k_step_future for KitchenPairDataset.")
    parser.add_argument("--target_size_dataloader", type=int, default=256, help="Target image size for KitchenPairDataset (e.g., 256 for 256x256). This also sets the model's initial_img_resolution.")
    parser.add_argument("--data_dir_dataloader", type=str, default="data/kitchen_pairs", help="Data directory for KitchenPairDataset.")
    parser.add_argument("--force_rebuild_dataloader", action="store_true", help="Force rebuild the dataset for the dataloader.")
    
    # --- Model Structure Arguments ---
    # These must match the architecture of the model defined in model.py or loaded from a checkpoint.
    # Defaults are based on src/diffusion/model.py ConditionalUNet defaults or sensible values for 256px.
    parser.add_argument("--model_base_channels", type=int, default=64, help="Base channels for the U-Net.")
    parser.add_argument("--model_channel_mults", type=str, default="1,2,3,4", help="Channel multipliers for U-Net levels (comma-separated string, e.g., '1,2,4,8').")
    parser.add_argument("--model_num_res_blocks", type=int, default=2, help="Number of residual blocks per U-Net level.")
    parser.add_argument("--model_attention_resolutions", type=str, default="32,16", help="Resolutions at which to apply attention (comma-separated string, e.g., '32,16' for a 256px input -> 256,128,64,32,16). Model default is (120,60) for 480px input.")
    parser.add_argument("--model_dropout_rate", type=float, default=0.1, help="Dropout rate for ResidualBlocks.")
    parser.add_argument("--model_time_emb_dim", type=int, default=256, help="Dimension of sinusoidal time embedding.")
    parser.add_argument("--model_time_emb_mlp_dim", type=int, default=1024, help="Dimension of MLP for time embedding.")
    parser.add_argument("--model_checkpoint_path", type=str, default=None, help="Optional path to model checkpoint (.pth or .pt) for loading pre-trained weights.")

    # --- Benchmark Execution Arguments ---
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for benchmarking.")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches to run for benchmark. If more than available in dataset, runs on all available.")
    parser.add_argument("--num_warmup_batches", type=int, default=10, help="Number of warm-up batches before starting timing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu", help="Device to use: 'cpu', 'cuda', or 'mps'.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers. Set to 0 for simplicity/debugging, >0 for potential speedup if I/O is a bottleneck.")
    parser.add_argument("--print_freq", type=int, default=10, help="Frequency of printing batch processing time (e.g., every 10 batches).")

    # --- Profiler Arguments ---
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler.")
    parser.add_argument("--profile_output_dir", type=str, default="./profiler_traces", help="Directory to save profiler traces.")
    parser.add_argument("--profile_schedule_wait", type=int, default=1, help="Profiler schedule: wait steps.")
    parser.add_argument("--profile_schedule_warmup", type=int, default=1, help="Profiler schedule: warmup steps.")
    parser.add_argument("--profile_schedule_active", type=int, default=3, help="Profiler schedule: active (measurement) steps.")
    parser.add_argument("--profile_schedule_repeat", type=int, default=1, help="Profiler schedule: number of wait, warmup, active cycles.")

    # --- AMP Arguments ---
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) for CUDA inference. Disabled if --quantize_ptq_static is used.")

    # --- PTQ Static Quantization Arguments ---
    parser.add_argument("--quantize_ptq_static", action="store_true", help="Enable Post-Training Static Quantization (INT8). This will override AMP and Profiler for the benchmark run, and force PTQ benchmark on CPU.")
    parser.add_argument("--ptq_num_calibration_batches", type=int, default=10, help="Number of batches to use for PTQ static calibration.")
    parser.add_argument("--ptq_backend", type=str, default="fbgemm", choices=['fbgemm', 'qnnpack'], help="Quantization backend for PTQ static (e.g., 'fbgemm' for x86, 'qnnpack' for ARM). Runs on CPU.")

    parsed_args = parser.parse_args()
    
    # Validate attention resolutions based on target_size and channel_mults
    try:
        parsed_attention_resolutions = tuple(map(int, parsed_args.model_attention_resolutions.split(',')))
        current_res = parsed_args.target_size_dataloader
        possible_resolutions = {current_res}
        for _ in range(len(parsed_args.model_channel_mults.split(',')) -1): # num downsamples
            current_res //= 2
            possible_resolutions.add(current_res)
        
        for attn_res in parsed_attention_resolutions:
            if attn_res not in possible_resolutions:
                print(f"Warning: Attention resolution {attn_res} is not a valid downsampled resolution for initial size {parsed_args.target_size_dataloader}.")
                print(f"Possible resolutions are: {sorted(list(possible_resolutions))}")
    except ValueError:
        print("Error: --model_attention_resolutions must be a comma-separated list of integers.")
        exit(1)

    main(parsed_args) 