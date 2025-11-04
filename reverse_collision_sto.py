from diffusion_sto import idx, w_const
import numpy as np
from numba import cuda
from numba.cuda.random import (
    xoroshiro128p_uniform_float32,
    xoroshiro128p_uniform_float64,
)
from lattice_constants import DTYPE

if np.dtype(DTYPE) == np.float64:
    xoroshiro_uniform = xoroshiro128p_uniform_float64
else:
    xoroshiro_uniform = xoroshiro128p_uniform_float32

@cuda.jit(fastmath=True)
def diffusion_collision_kernel_stochastic_with_noise_storage(f, omega, omega_noise, rng_states, noise_storage, step, nx, ny):
    """Collision for diffusion with stochastic term and noise storage for specific time step"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Get thread ID for random number generation
        thread_id = i + j * nx
        
        # Compute density
        rho = DTYPE(0.0)
        for k in range(9):
            rho += f[idx(i, j, k, nx, ny)]
        
        # Collision step with stochastic term
        for k in range(9):
            feq = w_const[k] * rho  # Equilibrium for diffusion
            
            # Generate random noise term: omega_noise * w[k] * random_noise
            random_val = xoroshiro_uniform(rng_states, thread_id)
            noise_term = w_const[k] * omega_noise * (random_val - DTYPE(0.5)) * DTYPE(2.0)  # Scale to [-1, 1]
            
            # Store the noise term for this time step: noise_storage[step][i,j,k]
            noise_storage[step * nx * ny * 9 + idx(i, j, k, nx, ny)] = noise_term
            
            # Apply collision with stochastic term
            f[idx(i, j, k, nx, ny)] = (DTYPE(1.0) - omega) * f[idx(i, j, k, nx, ny)] + omega * feq + noise_term


@cuda.jit(fastmath=True)
def diffusion_collision_kernel_reverse_stochastic(fp, omega, noise_storage, step, nx, ny):
    """Reverse collision for stochastic diffusion using stored noise terms for specific time step"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Compute density
        rho = DTYPE(0.0)
        noise_sum = DTYPE(0.0)
        for k in range(9):
            rho += fp[idx(i, j, k, nx, ny)]
            noise_sum += noise_storage[step * nx * ny * 9 + idx(i, j, k, nx, ny)]

        # Reverse collision step for stochastic diffusion
        for k in range(9):
            # Get the stored noise term for this time step (reverse order)
            noise_term = noise_storage[step * nx * ny * 9 + idx(i, j, k, nx, ny)]
            # Stochastic reverse collision: (fp - omega * w_const[k] * (rho - noise_sum) - noise_term) / (1 - omega)
            fp[idx(i, j, k, nx, ny)] = (fp[idx(i, j, k, nx, ny)] - omega * w_const[k] * (rho - noise_sum) - noise_term) / (DTYPE(1.0) - omega)
