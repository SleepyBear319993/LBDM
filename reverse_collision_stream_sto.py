from diffusion_sto import diffusion_collision_kernel_stochastic, diffusion_collision_kernel, streaming_kernel_periodic, idx, w_const
from reverse_collision_sto import diffusion_collision_kernel_stochastic_with_noise_storage, diffusion_collision_kernel_reverse_stochastic
from reverse_pull import streaming_kernel_periodic_reverse
import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from kernel_gpu import DTYPE

if __name__ == "__main__":
    # do stochastic collision streaming and reverse collision streaming
    nx, ny = 256, 256
    omega = 0.01
    omega_noise = 0.01
    num_steps = 50
    
    # Use flat array instead of 3D array
    np.random.seed(42)
    f_in = np.random.randn(nx*ny*9).astype(np.float32)
    f_in_original = f_in.copy()  # Keep original for comparison
    f_in_gpu = cuda.to_device(f_in)

    # Output should also be flat
    f_out_gpu = cuda.device_array(nx*ny*9, dtype=DTYPE)
    
    # Create noise storage array for ALL time steps: [num_steps, nx*ny*9]
    noise_storage = cuda.device_array(num_steps * nx * ny * 9, dtype=DTYPE)
    
    # Initialize random number generator states
    seed = np.random.randint(0, 2**31 - 1)
    rng_states = create_xoroshiro128p_states(nx * ny, seed=seed)

    threadsperblock = (16, 16)
    blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    print(f"Running {num_steps} steps of stochastic diffusion (omega={omega}, omega_noise={omega_noise})")
    
    # Forward: stochastic collision + streaming
    for step in range(num_steps):
        diffusion_collision_kernel_stochastic_with_noise_storage[blockspergrid, threadsperblock](
            f_in_gpu, omega, omega_noise, rng_states, noise_storage, step, nx, ny)
        streaming_kernel_periodic[blockspergrid, threadsperblock](f_in_gpu, f_out_gpu, nx, ny)
        f_in_gpu, f_out_gpu = f_out_gpu, f_in_gpu
    
    print(f"Running {num_steps} steps of reverse stochastic diffusion")
    
    # Reverse: reverse streaming + reverse stochastic collision
    for step in range(num_steps):
        reverse_step = num_steps - 1 - step  # Use noise terms in reverse order
        streaming_kernel_periodic_reverse[blockspergrid, threadsperblock](f_in_gpu, f_out_gpu, nx, ny)
        f_in_gpu, f_out_gpu = f_out_gpu, f_in_gpu
        diffusion_collision_kernel_reverse_stochastic[blockspergrid, threadsperblock](
            f_in_gpu, omega, noise_storage, reverse_step, nx, ny)

    f_out = f_in_gpu.copy_to_host()
    
    print(f"Original f_in range: [{np.min(f_in_original):.6f}, {np.max(f_in_original):.6f}]")
    print(f"Reverse diffusion f_out range: [{np.min(f_out):.6f}, {np.max(f_out):.6f}]")
    print(f"Arrays are exactly equal: {np.array_equal(f_in_original, f_out)}")
    
    # Tolerance testing
    rtol_values = [1e-3, 1e-4, 1e-5, 1e-6]
    atol_values = [1e-3, 1e-4, 1e-5, 1e-6]
    
    for rtol, atol in zip(rtol_values, atol_values):
        is_close = np.allclose(f_in_original, f_out, rtol=rtol, atol=atol)
        print(f"Arrays are approximately equal with rtol={rtol}, atol={atol}: {is_close}")

    # Calculate and print error metrics
    l2_error = np.linalg.norm(f_in_original - f_out)
    print(f"L2 error: {l2_error:.6e}")

    # Relative L2 error (normalized by the magnitude of f_in)
    relative_l2_error = l2_error / np.linalg.norm(f_in_original)
    print(f"Relative L2 error: {relative_l2_error:.6e}")
    
    # Maximum absolute error
    abs_errors = np.abs(f_in_original - f_out)
    max_error_idx = np.argmax(abs_errors)
    max_abs_error = abs_errors[max_error_idx]
    print(f"Maximum absolute error: {max_abs_error:.6e}")
    print(f"Maximum error location index: {max_error_idx}")
    print(f"Value at f_in: {f_in_original[max_error_idx]:.6f}, value at f_out: {f_out[max_error_idx]:.6f}")
    
    # Maximum relative error (avoid division by zero)
    nonzero_mask = np.abs(f_in_original) > 1e-12
    if np.any(nonzero_mask):
        relative_errors = np.abs(f_in_original - f_out)[nonzero_mask] / np.abs(f_in_original)[nonzero_mask]
        max_rel_error = np.max(relative_errors)
        max_rel_error_idx = np.argmax(relative_errors)
        print(f"Maximum relative error: {max_rel_error:.6e}")
    else:
        print("Cannot compute relative error: all original values are essentially zero")
    
    # Mean absolute error
    mean_abs_error = np.mean(abs_errors)
    print(f"Mean absolute error: {mean_abs_error:.6e}")
    
    # Root mean square error
    rmse = np.sqrt(np.mean((f_in_original - f_out)**2))
    print(f"Root mean square error (RMSE): {rmse:.6e}")
    
    # Statistical summary
    print(f"\nError Statistics Summary:")
    print(f"  L2 Error: {l2_error:.6e}")
    print(f"  Relative L2 Error: {relative_l2_error:.6e}")
    print(f"  Max Absolute Error: {max_abs_error:.6e}")
    print(f"  Mean Absolute Error: {mean_abs_error:.6e}")
    print(f"  RMSE: {rmse:.6e}")
    
    # Success/failure assessment
    if relative_l2_error < 1e-10:
        print(f"\n✓ SUCCESS: Reverse stochastic diffusion is highly accurate (relative L2 error < 1e-10)")
    elif relative_l2_error < 1e-4:  # Relaxed for stochastic + streaming case
        print(f"\n✓ SUCCESS: Reverse stochastic diffusion is reasonably accurate (relative L2 error < 1e-4)")
    elif relative_l2_error < 1e-2:
        print(f"\n⚠ WARNING: Reverse stochastic diffusion has moderate accuracy (relative L2 error < 1e-2)")
    else:
        print(f"\n✗ FAILURE: Reverse stochastic diffusion has poor accuracy (relative L2 error >= 1e-2)")