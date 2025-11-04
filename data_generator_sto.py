import numpy as np
import os
from numba import cuda
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose
import h5py

# Import the stochastic LBM solver
from diffusion_reversal_sto import LBMDiffusionReversalSolverStochastic, idx_host
from lattice_constants import DTYPE

def generate_data_for_image(image_idx, normalized_rgb_image, solver, total_steps, output_dir):
    """
    Runs LBM stochastic diffusion for one image and saves (f_t, noise_t) pairs.
    
    The process:
    1. Initialize from image (f_0)
    2. Run forward stochastic diffusion for T steps, storing noise at each step
    3. Save pairs of (f_t, noise_t) for t=1 to T
    
    During training, the U-Net will learn to predict noise_t from f_t.
    During inference, we can use predicted noise to perform reverse stochastic collision.
    """
    print(f"Processing image {image_idx} for stochastic diffusion...")
    nx, ny = solver.nx, solver.ny
    num_channels = 3
    num_distributions = 9

    # Initialize from the input image
    print(f"  Image {image_idx}: Initializing from image (f_0)...")
    solver.initialize_from_image(normalized_rgb_image)
    
    # Prepare noise storage for the total number of steps
    print(f"  Image {image_idx}: Preparing noise storage for {total_steps} steps...")
    solver.prepare_noise_storage(total_steps)

    # Create HDF5 file to store the data
    h5_path = os.path.join(output_dir, f"image_{image_idx}_lbm_stochastic_data.h5")
    with h5py.File(h5_path, 'w') as hf:
        # Create datasets for f_t and noise_t
        # Shape: (total_steps, num_channels, num_distributions, ny, nx)
        # We store pairs (f_t, noise_t) for t=1 to T
        # f_t is the state AFTER applying noise_t at step t
        dset_f_t = hf.create_dataset("f_t", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)
        dset_noise_t = hf.create_dataset("noise_t", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)

        # Run forward stochastic diffusion step by step
        print(f"  Image {image_idx}: Running forward stochastic diffusion for {total_steps} steps...")
        for t in range(total_steps):
            # Perform one forward step (collision with noise + streaming)
            # This updates solver.f to f_{t+1} and stores noise_t in solver.noise_storage[t]
            solver.forward_step()
            cuda.synchronize()  # Ensure step is complete
            
            # Get f_t (state after applying noise at step t, which is now stored in solver.f)
            f_t_host = [f_ch.copy_to_host() for f_ch in solver.f]
            
            # Get noise_t (noise applied at step t, stored in solver.noise_storage)
            # Retrieve noise for all channels at step t
            noise_t_host = []
            for ch in range(num_channels):
                # Extract noise for channel ch at step t
                noise_ch_flat = np.zeros(nx * ny * 9, dtype=DTYPE)
                noise_storage_host = solver.noise_storage[ch].copy_to_host()
                # Noise for step t starts at index t * nx * ny * 9
                start_idx = t * nx * ny * 9
                end_idx = (t + 1) * nx * ny * 9
                noise_ch_flat = noise_storage_host[start_idx:end_idx]
                noise_t_host.append(noise_ch_flat)
            
            # Store f_t and noise_t in HDF5 file
            # Index t represents the t-th step (t=0 corresponds to step 1, t=1 to step 2, etc.)
            for ch in range(num_channels):
                # Reshape flat arrays to (num_distributions, ny, nx) before storing
                try:
                    f_t_ch_reshaped = f_t_host[ch].reshape((num_distributions, ny, nx))
                    noise_t_ch_reshaped = noise_t_host[ch].reshape((num_distributions, ny, nx))
                except ValueError:
                    print(f"\nERROR: Reshape failed at step {t}. Array size {f_t_host[ch].size} != {num_distributions*ny*nx}")
                    print("       Check if flat array structure matches reshape order (num_distributions, ny, nx).")
                    # Clean up HDF5 file
                    hf.close()
                    if os.path.exists(h5_path):
                        os.remove(h5_path)
                    return  # Stop processing this image
                
                dset_f_t[t, ch, :, :, :] = f_t_ch_reshaped
                dset_noise_t[t, ch, :, :, :] = noise_t_ch_reshaped
            
            # Print progress
            if (t + 1) % 100 == 0 or (t + 1) == total_steps:
                print(f"  Image {image_idx}: Completed step {t+1}/{total_steps}. Stored (f_{t+1}, noise_{t+1}).")
    
    print(f"Finished image {image_idx}. Stochastic diffusion data saved to {h5_path}")


def main_generate_data():
    # --- Parameters ---
    img_size = 32
    target_class = 0  # e.g., class 0 from CIFAR10
    omega = 0.01
    omega_noise = 0.01  # Stochastic term strength
    total_steps = 1000  # T
    base_output_dir = "lbm_stochastic_data_cifar0_32"  # Base name
    max_train_images = None  # Limit number of training images, set to None for all
    max_test_images = None   # Limit number of test images, set to None for all
    # --- ---

    # Create output directories
    train_output_dir = f"{base_output_dir}_train"
    test_output_dir = f"{base_output_dir}_test"
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Load CIFAR10 dataset
    print("Loading CIFAR10 dataset...")
    transform = Compose([
        ToTensor()  # Converts to [0, 1] tensor C x H x W
    ])
    
    # Load training data
    cifar10_train = CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Load test data
    cifar10_test = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Initialize LBM stochastic solver (once, dimensions won't change)
    nx, ny = img_size, img_size
    solver = LBMDiffusionReversalSolverStochastic(nx, ny, omega, omega_noise)

    # --- Process Training Data ---
    print(f"\n--- Processing Training Data for class {target_class} ---")
    train_class_indices = [i for i, (_, label) in enumerate(cifar10_train) if label == target_class]
    if max_train_images is not None:
        train_class_indices = train_class_indices[:max_train_images]
    print(f"Found {len(train_class_indices)} training images for class {target_class}.")

    for i, img_idx in enumerate(train_class_indices):
        image_tensor, _ = cifar10_train[img_idx]  # Shape: C x H x W
        normalized_rgb_image = np.ascontiguousarray(image_tensor.permute(1, 2, 0).numpy()).astype(DTYPE)
        # Pass the specific output directory for training data
        generate_data_for_image(f"train_{img_idx}", normalized_rgb_image, solver, total_steps, train_output_dir)

    print("Training data generation complete.")

    # --- Process Test Data ---
    print(f"\n--- Processing Test Data for class {target_class} ---")
    test_class_indices = [i for i, (_, label) in enumerate(cifar10_test) if label == target_class]
    if max_test_images is not None:
        test_class_indices = test_class_indices[:max_test_images]
    print(f"Found {len(test_class_indices)} test images for class {target_class}.")

    for i, img_idx in enumerate(test_class_indices):
        image_tensor, _ = cifar10_test[img_idx]  # Shape: C x H x W
        normalized_rgb_image = np.ascontiguousarray(image_tensor.permute(1, 2, 0).numpy()).astype(DTYPE)
        # Pass the specific output directory for test data
        generate_data_for_image(f"test_{img_idx}", normalized_rgb_image, solver, total_steps, test_output_dir)

    print("Test data generation complete.")
    print("\nAll stochastic data generation complete.")


if __name__ == "__main__":
    main_generate_data()
