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

def _ensure_h5_datasets(hf, total_steps, num_channels, num_distributions, ny, nx):
    """
    Ensure required datasets/attributes exist. Create if missing.
    Returns the 'written' boolean array and last completed index (or -1).
    """
    # Create core datasets if absent
    if 'f_t' not in hf:
        hf.create_dataset("f_t", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)
    if 'delta_f' not in hf:
        hf.create_dataset("delta_f", (total_steps, num_channels, num_distributions, ny, nx), dtype=DTYPE)

    # Create or load a progress tracker
    if 'written' not in hf:
        written = hf.create_dataset('written', (total_steps,), dtype='uint8')  # 0 = not written, 1 = written
        written[...] = 0
    else:
        written = hf['written']

    # Store helpful metadata
    hf.attrs.setdefault('ny', ny)
    hf.attrs.setdefault('nx', nx)
    hf.attrs.setdefault('num_channels', num_channels)
    hf.attrs.setdefault('num_distributions', num_distributions)
    hf.attrs.setdefault('total_steps', total_steps)

    # Determine last completed index
    last_completed_idx = -1
    try:
        # Fast path using the written mask
        w = written[...]
        if np.any(w):
            last_completed_idx = int(np.max(np.where(w != 0)))
    except Exception:
        pass

    # Fallback for legacy files without 'written' data populated: heuristic scan from the end
    if last_completed_idx < 0:
        f_t = hf['f_t']
        delta_f = hf['delta_f']
        for idx in range(total_steps - 1, -1, -1):
            try:
                # Heuristic: if any non-zero exists in delta_f or f_t slice, consider it written
                if np.any(delta_f[idx, ...]) or np.any(f_t[idx, ...]):
                    last_completed_idx = idx
                    # Update written mask retroactively for future fast restarts
                    written[:last_completed_idx + 1] = 1
                    hf.flush()
                    break
            except Exception:
                break

    # Also mirror as attribute for human inspection
    hf.attrs['last_completed_idx'] = last_completed_idx
    hf.flush()
    return written, last_completed_idx


def _reconstruct_next_state_from_h5(hf, ny, nx):
    """
    Using the last written step k (storage index), reconstruct f_{k} and f_{k-1} to get
    the solver's next working state f_{k-1} without recomputing. Returns:
     - next_t: integer t to continue with (k)
     - flat channel arrays for f_{k-1} in a list length num_channels
    If nothing is written, returns (None, None)
    """
    total_steps = int(hf.attrs['total_steps'])
    num_channels = int(hf.attrs['num_channels'])
    # Ensure datasets exist
    written, last_completed_idx = _ensure_h5_datasets(
        hf,
        total_steps,
        num_channels,
        int(hf.attrs['num_distributions']),
        ny,
        nx,
    )

    if last_completed_idx < 0:
        return None, None

    # storage index k corresponds to time t = k + 1
    k = last_completed_idx
    t = k + 1
    f_t = hf['f_t'][k]        # shape (C, Q, ny, nx)
    delta_f = hf['delta_f'][k]  # shape (C, Q, ny, nx)

    # f_{t-1} = f_t - delta_f
    f_prev = f_t - delta_f
    flat_prev = [np.ascontiguousarray(f_prev[ch].reshape(-1)).astype(DTYPE) for ch in range(num_channels)]
    return t, flat_prev


def _load_state_into_solver(solver, flat_prev):
    """
    Copy flat host arrays (one per channel) into the solver's device arrays in-place.
    """
    for ch in range(len(flat_prev)):
        # Copy into existing device buffer
        cuda.to_device(flat_prev[ch], to=solver.f[ch])
    cuda.synchronize()


def generate_data_for_image(image_idx, normalized_rgb_image, solver, total_steps, output_dir):
    """
    Runs LBM diffusion reversal for one image and saves (f_t, f_{t-1}) pairs
    by first running forward to T, then running backward from T to 1.
    """
    print(f"Processing image {image_idx} for reversal...")
    nx, ny = solver.nx, solver.ny
    num_channels = 3
    num_distributions = 9

    # --- Step 1: Prepare output file and resume state if present ---
    h5_path = os.path.join(output_dir, f"image_{image_idx}_lbm_reversal_data.h5")

    file_mode = 'a' if os.path.exists(h5_path) else 'w'
    with h5py.File(h5_path, file_mode) as hf:
        # Ensure datasets, metadata, and progress markers exist
        written, last_completed_idx = _ensure_h5_datasets(hf, total_steps, num_channels, num_distributions, ny, nx)

        # Determine start t for reversal and initialize solver state accordingly
        start_t = None
        if last_completed_idx >= 0:
            # Resume from the next step: t_next = last_completed_t - 1
            last_t = last_completed_idx + 1
            start_t = last_t - 1
            print(f"  Image {image_idx}: Resuming from t={start_t} (last completed storage_idx={last_completed_idx}).")
            # Reconstruct f_{last_t-1} and load into solver
            t_for_prev, flat_prev = _reconstruct_next_state_from_h5(hf, ny, nx)
            if t_for_prev is not None and flat_prev is not None:
                _load_state_into_solver(solver, flat_prev)
            else:
                start_t = None  # Fallback to recomputing from scratch

        if start_t is None:
            # We need to compute f_T first, then start from T
            print(f"  Image {image_idx}: Running forward process ({total_steps} steps) to get f_T...")
            solver.initialize_from_image(normalized_rgb_image)  # Initialize to f_0
            try:
                solver.run_forward(total_steps)
            except AttributeError:
                print("  Solver does not have run_forward, running step-by-step...")
                for t_fwd in range(total_steps):
                    solver.forward_step()
            cuda.synchronize()
            print(f"  Image {image_idx}: Forward process complete. Starting/continuing reversal.")
            start_t = total_steps

        # Aliases to datasets
        dset_f_t = hf['f_t']
        dset_delta_f = hf['delta_f']

        # Loop backward from t = start_t down to t = 1
        for t in range(start_t, 0, -1):
            storage_idx = t - 1

            # Skip if already written (safe idempotency)
            if int(written[storage_idx]) == 1:
                if t % 200 == 0 or t == 1:
                    print(f"  Image {image_idx}: Skipping already written t={t} (idx={storage_idx}).")
                continue

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
                # Mark failure and exit gracefully without deleting progress
                hf.attrs['error'] = 'reverse_step_not_found'
                hf.flush()
                return # Stop processing this image
            except Exception as e:
                print(f"\nERROR during solver.reverse_step() at t={t}: {e}")
                # Persist partial data and exit gracefully
                hf.attrs['error'] = f'exception_at_t_{t}: {e}'
                hf.flush()
                return  # Stop processing this image

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
                     # Persist partial data and exit gracefully
                     hf.attrs['error'] = f'reshape_failed_at_idx_{storage_idx}'
                     hf.flush()
                     return # Stop processing this image

                dset_f_t[storage_idx, ch, :, :, :] = f_t_ch_reshaped
                dset_delta_f[storage_idx, ch, :, :, :] = delta_f_ch_reshaped

            # Mark as written only after both datasets are stored
            written[storage_idx] = 1
            hf.attrs['last_completed_idx'] = int(storage_idx)
            hf.flush()

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
        out_path = os.path.join(train_output_dir, f"image_{file_image_idx}_lbm_reversal_data.h5")
        # Skip fully completed images
        if os.path.exists(out_path):
            try:
                with h5py.File(out_path, 'r') as hf:
                    if 'written' in hf and int(np.sum(hf['written'][...])) == total_steps:
                        print(f"Skipping completed training image {file_image_idx}.")
                        continue
            except OSError:
                # Corrupted file; fall through to regenerate
                pass
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
        out_path = os.path.join(test_output_dir, f"image_{file_image_idx}_lbm_reversal_data.h5")
        # Skip fully completed images
        if os.path.exists(out_path):
            try:
                with h5py.File(out_path, 'r') as hf:
                    if 'written' in hf and int(np.sum(hf['written'][...])) == total_steps:
                        print(f"Skipping completed test image {file_image_idx}.")
                        continue
            except OSError:
                # Corrupted file; fall through to regenerate
                pass
        generate_data_for_image(file_image_idx, normalized_rgb_image, solver, total_steps, test_output_dir)

    print("Test data generation complete.")
    print("\nAll data generation complete.")

if __name__ == "__main__":
    main_generate_data()