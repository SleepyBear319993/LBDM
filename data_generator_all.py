import numpy as np
import os
#from PIL import Image
from numba import cuda # Assuming solver uses numba cuda arrays
#import torch # Or tensorflow
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose
import h5py # For efficient data storage

# Assuming LBMDiffusionReversalSolver is in diffusion_reversal.py
from diffusion_reversal import LBMDiffusionReversalSolver, idx_host # Import idx_host if needed for reshaping/indexing
from lattice_constants import DTYPE # Make sure DTYPE matches your solver (e.g., np.float32)

def generate_data_for_image(image_idx, normalized_rgb_image, solver, total_steps, output_dir):
    """
    Runs LBM diffusion reversal for one image and saves (f_t, f_{t-1}) pairs
    by first running forward to T, then running backward from T to 1.
    """
    print(f"Processing image {image_idx} for reversal...")
    nx, ny = solver.nx, solver.ny
    num_channels = 3
    num_distributions = 9

    # --- Step 1: Run forward process to get f_T ---
    print(f"  Image {image_idx}: Running forward process ({total_steps} steps) to get f_T...")
    solver.initialize_from_image(normalized_rgb_image) # Initialize to f_0
    # Use the existing run_forward method if available and efficient
    try:
        solver.run_forward(total_steps)
    except AttributeError:
        print("  Solver does not have run_forward, running step-by-step...")
        for t in range(total_steps):
            solver.forward_step()
            # Optional: Add progress print for forward run
            # if (t + 1) % 100 == 0:
            #     print(f"    Forward step {t+1}/{total_steps}")
    cuda.synchronize() # Ensure forward run is complete
    print(f"  Image {image_idx}: Forward process complete. Starting reversal.")
    # Solver state solver.f now holds f_T

    # --- Step 2: Run backward process and save data ---
    h5_path = os.path.join(output_dir, f"image_{image_idx}_lbm_reversal_data.h5")
    with h5py.File(h5_path, 'w') as hf:
        # Create datasets for f_t and the difference (delta_f = f_t - f_{t-1})
        # Shape: (total_steps, num_channels, num_distributions, ny, nx)
        # We store pairs corresponding to the transition from t to t-1
        dset_f_t = hf.create_dataset("f_t", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)
        dset_delta_f = hf.create_dataset("delta_f", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)

        # Loop backward from t = T down to t = 1
        for t in range(total_steps, 0, -1):
            # Get f_t (current state *before* backward step)
            f_t_host = [f_ch.copy_to_host() for f_ch in solver.f]

            # Perform one backward step (calculates f_{t-1} from f_t)
            # This updates solver.f in-place to f_{t-1}
            try:
                solver.reverse_step()
                cuda.synchronize() # Ensure step is complete before copying
            except AttributeError:
                print("\nERROR: solver.reverse_step() method not found!")
                print("       Cannot proceed with reversal simulation.")
                # Clean up HDF5 file
                hf.close()
                if os.path.exists(h5_path): os.remove(h5_path)
                return # Stop processing this image
            except Exception as e:
                print(f"\nERROR during solver.reverse_step() at t={t}: {e}")
                 # Clean up HDF5 file
                hf.close()
                if os.path.exists(h5_path): os.remove(h5_path)
                return # Stop processing this image

            # Get f_{t-1} (state *after* backward step)
            f_prev_host = [f_ch.copy_to_host() for f_ch in solver.f]

            # Calculate difference: delta_f = f_t - f_{t-1}
            delta_f_host = []
            for ch in range(num_channels):
                 # Assuming f_t_host/f_prev_host are flat [nx*ny*9]
                 f_t_ch_flat = f_t_host[ch]
                 f_prev_ch_flat = f_prev_host[ch]
                 delta_f_ch_flat = f_t_ch_flat - f_prev_ch_flat
                 delta_f_host.append(delta_f_ch_flat)

            # Store f_t and delta_f in HDF5 file
            # Use index t-1 (maps T->T-1, T-1->T-2, ..., 1->0)
            storage_idx = t - 1
            for ch in range(num_channels):
                # Reshape flat arrays to (num_distributions, ny, nx) before storing
                # Ensure the reshape order matches how the flat array is structured
                # If flat array is (k, y, x) flattened C-style, reshape is correct.
                # If flat array uses idx_host, reshaping might need care or direct indexing.
                try:
                    f_t_ch_reshaped = f_t_host[ch].reshape((num_distributions, ny, nx))
                    delta_f_ch_reshaped = delta_f_host[ch].reshape((num_distributions, ny, nx))
                except ValueError:
                     print(f"\nERROR: Reshape failed. Array size {f_t_host[ch].size} != {num_distributions*ny*nx}")
                     print("       Check if flat array structure matches reshape order (num_distributions, ny, nx).")
                     # Clean up HDF5 file
                     hf.close()
                     if os.path.exists(h5_path): os.remove(h5_path)
                     return # Stop processing this image

                dset_f_t[storage_idx, ch, :, :, :] = f_t_ch_reshaped
                dset_delta_f[storage_idx, ch, :, :, :] = delta_f_ch_reshaped

            # Solver state is now f_{t-1}, ready for the next iteration (t-1)

            if t % 100 == 0 or t == 1:
                print(f"  Image {image_idx}: Completed reverse step t={t}. Stored pair ({t}, {t-1}) at index {storage_idx}.")

    print(f"Finished image {image_idx}. Reversal data saved to {h5_path}")


def main_generate_data():
    # --- Parameters ---
    img_size = 32
    omega = 0.01
    total_steps = 1000 # T
    # --- Changed output directory name ---
    base_output_dir = "lbm_reversal_data_cifar_all_32" # Base name for all classes
    max_train_images = None # Limit number of training images, set to None for all
    max_test_images = None  # Limit number of test images, set to None for all
    # --- ---

    # Create output directories
    train_output_dir = f"{base_output_dir}_train"
    test_output_dir = f"{base_output_dir}_test"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Load CIFAR10 dataset
    print("Loading CIFAR10 dataset...")
    transform = Compose([
        ToTensor() # Converts to [0, 1] tensor C x H x W
    ])
    
    # Load training data
    cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Load test data
    cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize LBM solver (once, dimensions won't change)
    nx, ny = img_size, img_size
    solver = LBMDiffusionReversalSolver(nx, ny, omega)

    # --- Process Training Data ---
    print(f"\n--- Processing ALL Training Data ---")
    num_total_train_images = len(cifar10_train)
    images_to_process_train = range(num_total_train_images)
    if max_train_images is not None:
        images_to_process_train = range(min(num_total_train_images, max_train_images))
    
    print(f"Processing {len(list(images_to_process_train))} training images out of {num_total_train_images} total.")

    for i in images_to_process_train:
        image_tensor, label = cifar10_train[i] # Get image and its label
        normalized_rgb_image = np.ascontiguousarray(image_tensor.permute(1, 2, 0).numpy()).astype(DTYPE)
        # Create a unique image ID including the class label and original index
        file_image_idx = f"train_class{label}_idx{i}"
        generate_data_for_image(file_image_idx, normalized_rgb_image, solver, total_steps, train_output_dir)

    print("Training data generation complete.")

    # --- Process Test Data ---
    print(f"\n--- Processing ALL Test Data ---")
    num_total_test_images = len(cifar10_test)
    images_to_process_test = range(num_total_test_images)
    if max_test_images is not None:
        images_to_process_test = range(min(num_total_test_images, max_test_images))

    print(f"Processing {len(list(images_to_process_test))} test images out of {num_total_test_images} total.")
    
    for i in images_to_process_test:
        image_tensor, label = cifar10_test[i] # Get image and its label
        normalized_rgb_image = np.ascontiguousarray(image_tensor.permute(1, 2, 0).numpy()).astype(DTYPE)
        # Create a unique image ID including the class label and original index
        file_image_idx = f"test_class{label}_idx{i}"
        generate_data_for_image(file_image_idx, normalized_rgb_image, solver, total_steps, test_output_dir)

    print("Test data generation complete.")
    print("\nAll data generation complete.")

if __name__ == "__main__":
    main_generate_data()