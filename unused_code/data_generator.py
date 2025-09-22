import numpy as np
import os
#from PIL import Image
#from numba import cuda
#import torch # Or tensorflow
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose
import h5py # For efficient data storage

# Assuming LBMDiffusionReversalSolver is in diffusion_reversal.py
from diffusion_reversal import LBMDiffusionReversalSolver
from lattice_constants import DTYPE # Make sure DTYPE matches your solver (e.g., np.float32)

def generate_data_for_image(image_idx, normalized_rgb_image, solver, total_steps, output_dir):
    """
    Runs LBM diffusion for one image and saves (f_t, f_{t-1}) pairs.
    """
    print(f"Processing image {image_idx}...")
    nx, ny = solver.nx, solver.ny
    num_channels = 3
    num_distributions = 9

    # Initialize solver for the current image
    solver.initialize_from_image(normalized_rgb_image)

    # Store f_t-1
    f_prev = [f_ch.copy_to_host() for f_ch in solver.f] # Store initial state f_0

    # Create HDF5 file for this image's data
    h5_path = os.path.join(output_dir, f"image_{image_idx}_lbm_data.h5")
    with h5py.File(h5_path, 'w') as hf:
        # Create datasets for f_t and the difference (delta_f)
        # Shape: (total_steps, num_channels, num_distributions, ny, nx)
        # Note: We store f_t and delta_f = f_t - f_{t-1} for t = 1 to total_steps
        dset_f_t = hf.create_dataset("f_t", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)
        dset_delta_f = hf.create_dataset("delta_f", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)

        for t in range(1, total_steps + 1):
            # Perform one forward step (calculates f_t from f_{t-1})
            solver.forward_step()

            # Get f_t
            f_curr = [f_ch.copy_to_host() for f_ch in solver.f]

            # Calculate difference: delta_f = f_t - f_{t-1}
            delta_f = []
            for ch in range(num_channels):
                 # Reshape f_prev and f_curr from flat (nx*ny*9) to (9, ny, nx) for easier subtraction if needed,
                 # or perform subtraction on the flat arrays if indexing is consistent.
                 # Assuming f_prev/f_curr are flat [nx*ny*9]
                 f_curr_ch_flat = f_curr[ch]
                 f_prev_ch_flat = f_prev[ch]
                 delta_f_ch_flat = f_curr_ch_flat - f_prev_ch_flat
                 delta_f.append(delta_f_ch_flat)

            # Store f_t and delta_f in HDF5 file (reshape appropriately)
            # Reshape flat arrays to (num_distributions, ny, nx) before storing
            for ch in range(num_channels):
                # Assuming idx_host maps (x, y, k) -> flat_index correctly
                # It might be easier to work with reshaped arrays:
                f_curr_ch_reshaped = f_curr[ch].reshape((num_distributions, ny, nx)) # -> (9, ny, nx)
                delta_f_ch_reshaped = delta_f[ch].reshape((num_distributions, ny, nx)) # -> (9, ny, nx)

                dset_f_t[t-1, ch, :, :, :] = f_curr_ch_reshaped
                dset_delta_f[t-1, ch, :, :, :] = delta_f_ch_reshaped

            # Update f_prev for the next iteration
            f_prev = f_curr

            if t % 100 == 0:
                print(f"  Image {image_idx}: Completed step {t}/{total_steps}")

    print(f"Finished image {image_idx}. Data saved to {h5_path}")


def main_generate_data():
    # --- Parameters ---
    img_size = 32
    target_class = 0 # e.g., class 0 from CIFAR10
    omega = 0.01
    total_steps = 1000 # T
    output_dir = "lbm_diffusion_data_cifar0_32"
    max_images = None #100 # Limit number of images for testing, set to None for all
    # --- ---

    os.makedirs(output_dir, exist_ok=True)

    # Load CIFAR10 dataset
    print("Loading CIFAR10 dataset...")
    transform = Compose([
        ToTensor() # Converts to [0, 1] tensor C x H x W
    ])
    # Load only training data, adjust path if needed
    cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Filter for the target class
    class_indices = [i for i, (_, label) in enumerate(cifar10_train) if label == target_class]
    if max_images is not None:
        class_indices = class_indices[:max_images]
    print(f"Found {len(class_indices)} images for class {target_class}.")

    # Initialize LBM solver (once, dimensions won't change)
    nx, ny = img_size, img_size
    solver = LBMDiffusionReversalSolver(nx, ny, omega)

    # Process each image
    for i, img_idx in enumerate(class_indices):
        image_tensor, _ = cifar10_train[img_idx] # Shape: C x H x W
        # Convert to numpy H x W x C and ensure correct dtype and contiguity
        normalized_rgb_image = np.ascontiguousarray(image_tensor.permute(1, 2, 0).numpy()).astype(DTYPE)

        generate_data_for_image(img_idx, normalized_rgb_image, solver, total_steps, output_dir)

    print("Data generation complete.")

if __name__ == "__main__":
    main_generate_data()