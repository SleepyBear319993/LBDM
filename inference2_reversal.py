import torch
import numpy as np
from numba import cuda
from PIL import Image
import os
import math

# --- Project Imports ---
from unet_lbm import UNetLBM
from diffusion_reversal import LBMDiffusionReversalSolver # Assumes this has forward_step and initialize_from_image
from diffusion import compute_density_kernel, idx
from kernel_gpu import DTYPE, w_const, threads_per_block, blocks_per_grid

# --- Configuration ---
IMG_SIZE = 32
N_CHANNELS_MODEL = 27
N_OUT_CHANNELS_MODEL = 27
NUM_CHANNELS_IMG = 3
NUM_DISTRIBUTIONS = 9
TOTAL_STEPS = 250 # T used during training AND for forward diffusion
OMEGA = 0.01

# Inference Steps Configuration
STEPS_A_UNET = 2
STEPS_B_LBM = 8
NUM_BLOCKS_C = TOTAL_STEPS // (STEPS_A_UNET + STEPS_B_LBM) # Number of times to repeat the a+b block
TOTAL_STEPS = (STEPS_A_UNET + STEPS_B_LBM) * NUM_BLOCKS_C # Total steps for the entire process

# Ensure (a + b) * c == TOTAL_STEPS
if (STEPS_A_UNET + STEPS_B_LBM) * NUM_BLOCKS_C != TOTAL_STEPS:
    raise ValueError(f"Configuration error: (a + b) * c != {TOTAL_STEPS}")

MODEL_PATH = "unet_lbm_model_32_reversal.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_IMAGE_PATH = "assets/img35.png" # <<< Path to your input image
os.makedirs("bin3", exist_ok=True) # Ensure output directory exists
OUTPUT_FILENAME = f"bin3/generated_from_{os.path.basename(INPUT_IMAGE_PATH)}_a{STEPS_A_UNET}_b{STEPS_B_LBM}_c{NUM_BLOCKS_C}.png"

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
        flat_channel_data = delta_f_reshaped[ch].reshape(-1)
        delta_f_gpu_list.append(cuda.to_device(flat_channel_data.astype(DTYPE))) # Ensure correct dtype

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
    total_size = nx * ny * NUM_DISTRIBUTIONS # For kernel launch bounds

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
    model.eval()
    print("Model loaded.")

    # 2. Load and Preprocess Input Image
    print(f"Loading input image: {INPUT_IMAGE_PATH}")
    try:
        img = Image.open(INPUT_IMAGE_PATH).convert('RGB')
        if img.size != (nx, ny):
            print(f"Warning: Resizing input image from {img.size} to {(nx, ny)}")
            img = img.resize((nx, ny))
        img_np = np.array(img) / 255.0 # Normalize to [0, 1]
        # Ensure HWC format (PIL loads as HWC) and contiguous C-order array
        normalized_rgb_image = np.ascontiguousarray(img_np).astype(DTYPE)
        print("Input image loaded and preprocessed.")
    except FileNotFoundError:
        print(f"Error: Input image not found at {INPUT_IMAGE_PATH}")
        return
    except Exception as e:
        print(f"Error loading or processing image: {e}")
        return

    # 3. Initialize LBM Solver
    solver = LBMDiffusionReversalSolver(nx, ny, OMEGA)

    # 4. Run Forward LBM Diffusion to get f_T
    print(f"Running forward LBM diffusion for {TOTAL_STEPS} steps...")
    solver.initialize_from_image(normalized_rgb_image) # Initialize f_0 based on image

    for step in range(TOTAL_STEPS):
        solver.forward_step() # Assumes this method exists and works
        if (step + 1) % 100 == 0:
            print(f"  Forward step {step + 1}/{TOTAL_STEPS}")

    # Get f_T (the state after TOTAL_STEPS forward steps)
    # Ensure these are copies if solver.f might be modified later
    f_current_gpu = [f.copy_to_host() for f in solver.f] # Copy to host first
    f_current_gpu = [cuda.to_device(f_host) for f_host in f_current_gpu] # Copy back to device
    print(f"Forward diffusion complete. Starting inference from f_{TOTAL_STEPS}.")


    # 5. Iterative Reversal Loop (Starts from f_T obtained above)
    current_t = TOTAL_STEPS
    print(f"Starting inference loop for {TOTAL_STEPS} steps...")
    for c_block in range(NUM_BLOCKS_C):
        print(f"  Block {c_block + 1}/{NUM_BLOCKS_C} (T = {current_t} -> {current_t - STEPS_A_UNET - STEPS_B_LBM})")

        # --- 5a. U-Net Steps ---
        if STEPS_A_UNET > 0:
            print(f"    Applying {STEPS_A_UNET} U-Net steps...")
            sub_threads = 32
            sub_blocks = (total_size + (sub_threads - 1)) // sub_threads

            for a_step in range(STEPS_A_UNET):
                if current_t <= 0: break

                f_input_tensor = prepare_unet_input(f_current_gpu, nx, ny, DEVICE)
                t_tensor = torch.tensor([current_t], dtype=torch.float32).to(DEVICE)

                with torch.no_grad():
                    delta_f_pred_tensor = model(f_input_tensor, t_tensor)

                delta_f_pred_gpu = process_unet_output(delta_f_pred_tensor, nx, ny, NUM_DISTRIBUTIONS)

                f_next_gpu = []
                for i in range(NUM_CHANNELS_IMG):
                    f_next_channel_gpu = cuda.device_array_like(f_current_gpu[i])
                    subtract_arrays_kernel[sub_blocks, sub_threads](
                        f_current_gpu[i], delta_f_pred_gpu[i], f_next_channel_gpu
                    )
                    f_next_gpu.append(f_next_channel_gpu)

                f_current_gpu = f_next_gpu
                current_t -= 1

        # --- 5b. LBM Reversal Steps ---
        if STEPS_B_LBM > 0:
            print(f"    Applying {STEPS_B_LBM} LBM reversal steps...")
            solver.f = f_current_gpu
            solver.f_streamed = []
            for f in f_current_gpu:
                new_f_streamed = cuda.device_array_like(f)
                new_f_streamed.copy_to_device(f)
                solver.f_streamed.append(new_f_streamed)

            for b_step in range(STEPS_B_LBM):
                 if current_t <= 0: break
                 solver.reverse_step()
                 current_t -= 1

            f_current_gpu = solver.f # Or solver.f_streamed

        if current_t <= 0: break

    print("Inference loop finished.")

    # 6. Get Final Image (Compute Density from f_0)
    print("Computing final image from f_0...")
    density_gpu = [cuda.device_array((ny, nx), dtype=DTYPE) for _ in range(NUM_CHANNELS_IMG)]
    bpg = blocks_per_grid(nx, ny)
    tpb = threads_per_block()

    for ch in range(NUM_CHANNELS_IMG):
        compute_density_kernel[bpg, tpb](f_current_gpu[ch], density_gpu[ch], nx, ny)

    density_host = [d.copy_to_host() for d in density_gpu]
    img_array = np.stack(density_host, axis=-1)
    img_array_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_array_uint8, 'RGB')

    # 7. Display and Save
    print(f"Saving generated image to {OUTPUT_FILENAME}...")
    img.save(OUTPUT_FILENAME)
    img.show()
    print("Done.")


if __name__ == "__main__":
    main_inference()