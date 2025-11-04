import numpy as np
import os
from numba import cuda
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose
import h5py

# Import the stochastic solver
from diffusion_reversal_sto import LBMDiffusionReversalSolverStochastic, idx_host
from lattice_constants import DTYPE

def generate_data_for_image_stochastic(image_idx, normalized_rgb_image, solver, total_steps, output_dir):
    """
    Runs LBM stochastic diffusion reversal for one image and saves (f_t, noise_t) pairs.
    
    Process:
    1. Run forward process from f_0 to f_T (stores noise terms automatically)
    2. Run backward process from f_T to f_0
    3. Save pairs of (f_t, noise_t) where noise_t is the noise used in the forward step from t-1 to t
    
    The UNet will learn to predict noise_t from f_t, which can then be used with
    stochastic reverse collision to recover f_{t-1}.
    """
    print(f"Processing image {image_idx} for stochastic reversal...")
    nx, ny = solver.nx, solver.ny
    num_channels = 3
    num_distributions = 9

    # --- Step 1: Initialize and prepare noise storage ---
    print(f"  Image {image_idx}: Initializing and preparing noise storage for {total_steps} steps...")
    solver.initialize_from_image(normalized_rgb_image)
    solver.prepare_noise_storage(total_steps)
    
    # --- Step 2: Run forward process to get f_T and store noise terms ---
    print(f"  Image {image_idx}: Running forward process ({total_steps} steps) to get f_T and noise terms...")
    solver.run_forward(total_steps)
    cuda.synchronize()
    print(f"  Image {image_idx}: Forward process complete. Noise terms stored. Starting reversal.")
    
    # --- Step 3: Run backward process and save data ---
    h5_path = os.path.join(output_dir, f"image_{image_idx}_lbm_sto_reversal_data.h5")
    with h5py.File(h5_path, 'w') as hf:
        # Create datasets for f_t and noise_t
        # Shape: (total_steps, num_channels, num_distributions, ny, nx)
        # We store pairs corresponding to the transition from t to t-1
        dset_f_t = hf.create_dataset("f_t", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)
        dset_noise = hf.create_dataset("noise_t", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)

        # Loop backward from t = T down to t = 1
        for t in range(total_steps, 0, -1):
            # Get f_t (current state *before* backward step)
            f_t_host = [f_ch.copy_to_host() for f_ch in solver.f]
            
            # Get noise_t from noise_storage
            # The noise at step t-1 in the storage corresponds to the noise used to go from t-1 to t
            noise_step_idx = t - 1  # noise_storage index for the noise used at step t-1 in forward process
            noise_t_host = []
            for ch in range(num_channels):
                # Extract only the required slice from GPU (much faster!)
                start_idx = noise_step_idx * nx * ny * 9
                end_idx = start_idx + nx * ny * 9
                # Create a device array view for just this timestep's noise, then copy to host
                noise_t_ch_flat = solver.noise_storage[ch][start_idx:end_idx].copy_to_host()
                noise_t_host.append(noise_t_ch_flat)
            
            # Perform one backward step (calculates f_{t-1} from f_t)
            try:
                solver.reverse_step()
                cuda.synchronize()
            except AttributeError:
                print("\nERROR: solver.reverse_step() method not found!")
                print("       Cannot proceed with reversal simulation.")
                hf.close()
                if os.path.exists(h5_path): 
                    os.remove(h5_path)
                return
            except Exception as e:
                print(f"\nERROR during solver.reverse_step() at t={t}: {e}")
                hf.close()
                if os.path.exists(h5_path): 
                    os.remove(h5_path)
                return

            # Store f_t and noise_t in HDF5 file
            # Use index t-1 (maps T->T-1, T-1->T-2, ..., 1->0)
            storage_idx = t - 1
            for ch in range(num_channels):
                # Reshape flat arrays to (num_distributions, ny, nx) before storing
                try:
                    f_t_ch_reshaped = f_t_host[ch].reshape((num_distributions, ny, nx))
                    noise_t_ch_reshaped = noise_t_host[ch].reshape((num_distributions, ny, nx))
                except ValueError:
                    print(f"\nERROR: Reshape failed. Array size {f_t_host[ch].size} != {num_distributions*ny*nx}")
                    print("       Check if flat array structure matches reshape order (num_distributions, ny, nx).")
                    hf.close()
                    if os.path.exists(h5_path): 
                        os.remove(h5_path)
                    return

                dset_f_t[storage_idx, ch, :, :, :] = f_t_ch_reshaped
                dset_noise[storage_idx, ch, :, :, :] = noise_t_ch_reshaped

            if t % 100 == 0 or t == 1:
                print(f"  Image {image_idx}: Completed reverse step t={t}. Stored pair ({t}, noise_{t}) at index {storage_idx}.")

    print(f"Finished image {image_idx}. Stochastic reversal data saved to {h5_path}")


def main_generate_data_stochastic():
    # --- Parameters ---
    img_size = 32
    target_class = 0  # e.g., class 0 from CIFAR10
    omega = 0.01
    omega_noise = 0.01  # Stochastic noise strength
    total_steps = 1000  # T
    base_output_dir = "lbm_sto_reversal_data_cifar0_32"
    max_train_images = None  # Limit number of training images, set to None for all
    max_test_images = None   # Limit number of test images, set to None for all

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

    # Initialize stochastic LBM solver
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
        generate_data_for_image_stochastic(f"train_{img_idx}", normalized_rgb_image, solver, total_steps, train_output_dir)

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
        generate_data_for_image_stochastic(f"test_{img_idx}", normalized_rgb_image, solver, total_steps, test_output_dir)

    print("Test data generation complete.")
    print("\nAll stochastic data generation complete.")


if __name__ == "__main__":
    main_generate_data_stochastic()
