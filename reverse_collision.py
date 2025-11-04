from diffusion import diffusion_collision_kernel, idx, w_const
import numpy as np
from numba import cuda
from lattice_constants import DTYPE

@cuda.jit(fastmath=True)
def diffusion_collision_kernel_reverse(fp, omega, nx, ny):
    """Solution of solving collision equations for reverse diffusion"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Compute density
        rho = DTYPE(0.0)
        for k in range(9):
            rho += fp[idx(i, j, k, nx, ny)]
        
        # Collision step for reverse diffusion
        for k in range(9):
            feq = w_const[k] * rho
            fp[idx(i, j, k, nx, ny)] = (fp[idx(i, j, k, nx, ny)] - omega * feq) / (DTYPE(1.0) - omega)