from diffusion import idx, cx_const, cy_const
from numba import cuda

@cuda.jit
def streaming_kernel_periodic_reverse(f_in, f_out, nx, ny):
    """Reverse streaming with periodic boundary conditions"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        for k in range(9):
            # Periodic boundary conditions
            ip = (i - cx_const[k] + nx) % nx
            jp = (j - cy_const[k] + ny) % ny
            f_out[idx(ip, jp, k, nx, ny)] = f_in[idx(i, j, k, nx, ny)]