from diffusion import diffusion_collision_kernel, idx, w_const
import numpy as np
from numba import cuda

@cuda.jit(fastmath=True)
def diffusion_collision_kernel_reverse(fp, omega, nx, ny):
    """Solution of solving collision equations for reverse diffusion"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Compute density
        rho = 0.0
        for k in range(9):
            rho += fp[idx(i, j, k, nx, ny)]
        
        # Collision step for reverse diffusion
        for k in range(9):
            feq = w_const[k] * rho
            fp[idx(i, j, k, nx, ny)] = (fp[idx(i, j, k, nx, ny)] - omega * feq) / (1.0 - omega)

if __name__ == "__main__":
    nx, ny = 256, 256
    omega = 0.1
    num_steps = 10
    # Use flat array instead of 3D array
    f_in = np.random.randn(nx*ny*9).astype(np.float32)  # Changed from (nx, ny, 9) to flat array
    f_in_gpu = cuda.to_device(f_in)

    threadsperblock = (16, 16)
    blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)


    for step in range(num_steps):
        diffusion_collision_kernel[blockspergrid, threadsperblock](f_in_gpu, omega, nx, ny)
    for step in range(num_steps):
        diffusion_collision_kernel_reverse[blockspergrid, threadsperblock](f_in_gpu, omega, nx, ny)

    f_out = f_in_gpu.copy_to_host()
    print(f"Original f_in: {f_in}")
    print(f"Diffusion reversed f_out: {f_out}")
    print(f"Arrays are exactly equal: {np.array_equal(f_in, f_out)}")
    
    # abs(a - b) <= (atol + rtol * abs(b))
    rtol_value = 1e-5
    atol_value = 1e-5
    print(f"Arrays are approximately equal with relative tolerance {rtol_value} and absolute tolerance {atol_value}: {np.allclose(f_in, f_out, rtol=rtol_value, atol=atol_value)}")

    # Calculate and print L2 error
    l2_error = np.linalg.norm(f_in - f_out)
    print(f"L2 error: {l2_error}")

    # Also print the relative L2 error (normalized by the magnitude of f_in)
    relative_l2_error = l2_error / np.linalg.norm(f_in)
    print(f"Relative L2 error: {relative_l2_error}")
    
    # Maximum absolute error
    max_error_idx = np.argmax(np.abs(f_in - f_out))
    print(f"Maximum error location index: {max_error_idx}")
    print(f"Value at f_in: {f_in[max_error_idx]}, value at f_out: {f_out[max_error_idx]}")
    print(f"Maximum absolute error: {np.abs(f_in[max_error_idx]-f_out[max_error_idx])}")
    print(f"Maximum relative error: {np.abs(f_in[max_error_idx]-f_out[max_error_idx])/np.abs(f_in[max_error_idx])}")