import torch
import numpy as np
from numba import cuda
from PIL import Image
import os
import math

# --- Project Imports ---
# Assuming these files contain the necessary definitions
from unet_lbm import UNetLBM
from diffusion_reversal import LBMDiffusionReversalSolver
from diffusion import compute_density_kernel, idx # Need GPU idx
from kernel_gpu import DTYPE, w_const, threads_per_block, blocks_per_grid # Assuming constants/helpers are here

# --- Configuration ---
IMG_SIZE = 32
N_CHANNELS_MODEL = 27 # 3 RGB * 9 LBM distributions
N_OUT_CHANNELS_MODEL = 27
NUM_CHANNELS_IMG = 3
NUM_DISTRIBUTIONS = 9
TOTAL_STEPS = 1000 # T used during training
OMEGA = 0.01 # Must match training omega if LBM reversal depends on it

# Inference Steps Configuration (Ensure (a + b) * c == TOTAL_STEPS)
STEPS_A_UNET = 5       # Number of U-Net steps per block
STEPS_B_LBM = 5        # Number of LBM reversal steps per block
NUM_BLOCKS_C = 100     # Number of times to repeat the a+b block

if (STEPS_A_UNET + STEPS_B_LBM) * NUM_BLOCKS_C != TOTAL_STEPS:
    raise ValueError(f"Configuration error: (a + b) * c = ({STEPS_A_UNET} + {STEPS_B_LBM}) * {NUM_BLOCKS_C} != {TOTAL_STEPS}")

MODEL_PATH = "unet_lbm_model_32.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_FILENAME = f"generated_image_a{STEPS_A_UNET}_b{STEPS_B_LBM}_c{NUM_BLOCKS_C}.png"

# --- Helper Functions ---

@cuda.jit(device=True)
def idx_flat_to_coords(flat_idx, nx, ny):
    """ Convert flat index back to (i, j, k) """
    k = flat_idx // (nx * ny)
    remainder = flat_idx % (nx * ny)
    j = remainder // nx
    i = remainder % nx
    return i, j, k

@cuda.jit
def init_equilibrium_kernel(f_out, rho_init, nx, ny):
    """Initializes f to equilibrium distribution for a uniform density rho_init."""
    flat_idx = cuda.grid(1)
    if flat_idx < nx * ny * NUM_DISTRIBUTIONS:
        i, j, k = idx_flat_to_coords(flat_idx, nx, ny)
        f_out[flat_idx] = w_const[k] * rho_init

def prepare_unet_input(f_gpu_list, nx, ny, device):
    """ Converts a list of 3 flat GPU arrays [nx*ny*9] to a single tensor [1, 27, ny, nx] """
    batch_tensor = torch.empty((1, N_CHANNELS_MODEL, ny, nx), dtype=torch.float32, device=device)
    # This requires copying data potentially, might be slow if done every step.
    # Consider if U-Net can take flat input or if GPU-side reshape is feasible.
    # Simple approach: copy to host, reshape, copy back.
    temp_host = [f.copy_to_host() for f in f_gpu_list]
    reshaped_host = [h.reshape((NUM_DISTRIBUTIONS, ny, nx)) for h in temp_host]
    stacked_host = np.stack(reshaped_host, axis=0) # Shape (3, 9, ny, nx)
    combined_host = stacked_host.reshape(N_CHANNELS_MODEL, ny, nx) # Shape (27, ny, nx)
    batch_tensor[0] = torch.from_numpy(combined_host).to(device)
    return batch_tensor

def process_unet_output(delta_f_tensor, nx, ny, num_distributions):
    """ Converts the U-Net output tensor [1, 27, ny, nx] back to a list of 3 flat GPU arrays. """
    delta_f_tensor_cpu = delta_f_tensor.squeeze(0).cpu().numpy() # Shape (27, ny, nx)
    delta_f_reshaped = delta_f_tensor_cpu.reshape(NUM_CHANNELS_IMG, num_distributions, ny, nx) # (3, 9, ny, nx)

    delta_f_gpu_list = []
    for ch in range(NUM_CHANNELS_IMG):
        # Reshape to flat (k slowest, j middle, i fastest) and copy to GPU
        flat_channel_data = delta_f_reshaped[ch].reshape(-1) # Order should match idx_host/idx
        delta_f_gpu_list.append(cuda.to_device(flat_channel_data))

    return delta_f_gpu_list

@cuda.jit
def subtract_arrays_kernel(a, b, result):
    """Performs element-wise subtraction: result = a - b"""
    idx = cuda.grid(1)
    if idx < a.shape[0]: # Assuming flat arrays
        result[idx] = a[idx] - b[idx]

# --- Main Inference ---
def main_inference():
    print(f"Using device: {DEVICE}")
    nx, ny = IMG_SIZE, IMG_SIZE

    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = UNetLBM(n_channels=N_CHANNELS_MODEL, n_out_channels=N_OUT_CHANNELS_MODEL).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        return
    model.eval() # Set model to evaluation mode
    print("Model loaded.")

    # 2. Initialize LBM Solver (needed for reverse_step and density calculation)
    # Note: Omega might not be strictly needed if reverse_step doesn't use it,
    # but density calculation might assume equilibrium based on it.
    solver = LBMDiffusionReversalSolver(nx, ny, OMEGA)

    # 3. Generate Initial State f_T (Equilibrium for uniform density)
    print(f"Generating initial state f_{TOTAL_STEPS} (equilibrium)...")
    initial_rho = DTYPE(0.5) # Start with mid-gray uniform density
    f_current_gpu = []
    total_size = nx * ny * NUM_DISTRIBUTIONS
    threads = 32 # Example thread count
    blocks = (total_size + (threads - 1)) // threads

    for _ in range(NUM_CHANNELS_IMG):
        f_channel_gpu = cuda.device_array(total_size, dtype=DTYPE)
        init_equilibrium_kernel[blocks, threads](f_channel_gpu, initial_rho, nx, ny)
        f_current_gpu.append(f_channel_gpu)
    print("Initial state generated on GPU.")

    # 4. Iterative Reversal Loop
    current_t = TOTAL_STEPS
    print(f"Starting inference loop for {TOTAL_STEPS} steps...")
    for c_block in range(NUM_BLOCKS_C):
        print(f"  Block {c_block + 1}/{NUM_BLOCKS_C} (T = {current_t} -> {current_t - STEPS_A_UNET - STEPS_B_LBM})")

        # --- 4a. U-Net Steps ---
        if STEPS_A_UNET > 0:
            print(f"    Applying {STEPS_A_UNET} U-Net steps...")
            # Pre-calculate launch bounds for subtraction kernel (assuming flat arrays)
            sub_threads = 32 # Or another suitable number
            sub_blocks = (total_size + (sub_threads - 1)) // sub_threads

            for a_step in range(STEPS_A_UNET):
                if current_t <= 0: break

                # Prepare input
                f_input_tensor = prepare_unet_input(f_current_gpu, nx, ny, DEVICE)
                t_tensor = torch.tensor([current_t], dtype=torch.float32).to(DEVICE)

                # U-Net prediction
                with torch.no_grad():
                    delta_f_pred_tensor = model(f_input_tensor, t_tensor)

                # Process output and update f
                delta_f_pred_gpu = process_unet_output(delta_f_pred_tensor, nx, ny, NUM_DISTRIBUTIONS)

                # Calculate f_{t-1} = f_t - delta_f_pred using the kernel
                f_next_gpu = []
                for i in range(NUM_CHANNELS_IMG):
                    # Allocate result array for the next step
                    f_next_channel_gpu = cuda.device_array_like(f_current_gpu[i])
                    # Launch subtraction kernel
                    subtract_arrays_kernel[sub_blocks, sub_threads](
                        f_current_gpu[i], delta_f_pred_gpu[i], f_next_channel_gpu
                    )
                    f_next_gpu.append(f_next_channel_gpu)
                    # Optional: Clean up intermediate delta_f array if memory is tight
                    # del delta_f_pred_gpu[i]

                f_current_gpu = f_next_gpu
                current_t -= 1
                # print(f"      U-Net step done. t = {current_t}") # Verbose

        # --- 4b. LBM Reversal Steps ---
        if STEPS_B_LBM > 0:
            print(f"    Applying {STEPS_B_LBM} LBM reversal steps...")
            # Set the solver's state. Assumes solver uses self.f and self.f_streamed
            solver.f = f_current_gpu
            # Correctly copy device arrays for f_streamed
            solver.f_streamed = []
            for f in f_current_gpu:
                new_f_streamed = cuda.device_array_like(f) # Create new device array
                new_f_streamed.copy_to_device(f)          # Copy data from f to new array
                solver.f_streamed.append(new_f_streamed)

            for b_step in range(STEPS_B_LBM):
                 if current_t <= 0: break # Should not happen if config is correct
                 solver.reverse_step() # Assumes this updates solver.f and solver.f_streamed
                 current_t -= 1
                 # print(f"      LBM reverse step done. t = {current_t}") # Verbose

            # Update f_current_gpu from solver state after LBM steps
            f_current_gpu = solver.f # Or solver.f_streamed depending on which holds the result

        if current_t <= 0: break

    print("Inference loop finished.")

    # 5. Get Final Image (Compute Density from f_0)
    print("Computing final image from f_0...")
    density_gpu = [cuda.device_array((ny, nx), dtype=DTYPE) for _ in range(NUM_CHANNELS_IMG)]
    bpg = blocks_per_grid(nx, ny) # Use helper from kernel_gpu.py
    tpb = threads_per_block()

    for ch in range(NUM_CHANNELS_IMG):
        compute_density_kernel[bpg, tpb](f_current_gpu[ch], density_gpu[ch], nx, ny)

    density_host = [d.copy_to_host() for d in density_gpu]

    # Stack channels (ny, nx, 3) - density kernel outputs density[j, i]
    img_array = np.stack(density_host, axis=-1)

    # Clip, scale to [0, 255], and convert to uint8
    img_array_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)

    # Create PIL Image
    img = Image.fromarray(img_array_uint8, 'RGB')

    # 6. Display and Save
    print(f"Saving generated image to {OUTPUT_FILENAME}...")
    img.save(OUTPUT_FILENAME)
    img.show() # Display the image
    print("Done.")


if __name__ == "__main__":
    main_inference()