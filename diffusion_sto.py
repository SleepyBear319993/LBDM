from PIL import Image
import numpy as np
import time
from numba import cuda
import matplotlib.pyplot as plt

# Use constants and helper functions from lattice_constants.py
from lattice_constants import DTYPE, cx_const, cy_const, w_const

# Define helper indexing functions
@cuda.jit(device=True, inline=True)
def idx(i, j, k, nx, ny):
    """Linear memory indexing for GPU"""
    return i + j * nx + k * nx * ny

def idx_host(i, j, k, nx, ny):
    """Linear memory indexing for CPU"""
    return i + j * nx + k * nx * ny

# Add random number generation support
from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_uniform_float32,
    xoroshiro128p_uniform_float64,
)

if np.dtype(DTYPE) == np.float64:
    xoroshiro_uniform = xoroshiro128p_uniform_float64
else:
    xoroshiro_uniform = xoroshiro128p_uniform_float32

# LBM kernels for diffusion with stochastic term
@cuda.jit(fastmath=True)
def diffusion_collision_kernel_stochastic(f, omega, omega_noise, rng_states, nx, ny):
    """Collision for diffusion with stochastic term: omega*w*f_noise"""
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
            
            # Apply collision with stochastic term
            f[idx(i, j, k, nx, ny)] = (DTYPE(1.0) - omega) * f[idx(i, j, k, nx, ny)] + omega * feq + noise_term

@cuda.jit(fastmath=True)
def diffusion_collision_kernel(f, omega, nx, ny):
    """Original collision for diffusion (no stochastic term)"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Compute density
        rho = DTYPE(0.0)
        for k in range(9):
            rho += f[idx(i, j, k, nx, ny)]
        
        # Collision step for diffusion (no velocity component)
        for k in range(9):
            feq = w_const[k] * rho  # Equilibrium for diffusion
            f[idx(i, j, k, nx, ny)] = (1.0 - omega) * f[idx(i, j, k, nx, ny)] + omega * feq

@cuda.jit
def streaming_kernel_periodic(f_in, f_out, nx, ny):
    """Streaming with periodic boundary conditions"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        for k in range(9):
            # Periodic boundary conditions
            ip = (i - cx_const[k] + nx) % nx
            jp = (j - cy_const[k] + ny) % ny
            f_out[idx(i, j, k, nx, ny)] = f_in[idx(ip, jp, k, nx, ny)]

@cuda.jit
def compute_density_kernel(f, density, nx, ny):
    """Compute density field from distributions"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        rho = 0.0
        for k in range(9):
            rho += f[idx(i, j, k, nx, ny)]
        density[j, i] = rho  # Note: density is in [j,i] order to match image convention

@cuda.jit
def init_from_image_kernel(image_data, f, channel, nx, ny):
    """Initialize distribution functions from image data for a specific channel directly on GPU"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Get normalized channel value (image data is in [j,i] suffix)
        density = float(image_data[j, i, channel])
        
        # Initialize distributions with equilibrium for zero velocity
        for k in range(9):
            f[idx(i, j, k, nx, ny)] = w_const[k] * density

class LBMDiffusionSolver:
    def __init__(self, nx, ny, omega, omega_noise=0.0, use_stochastic=False):
        self.nx = nx
        self.ny = ny
        self.omega = DTYPE(omega)
        self.omega_noise = DTYPE(omega_noise)
        self.use_stochastic = use_stochastic
        
        # GPU arrays for 3 channels (R,G,B)
        self.f = [cuda.device_array(nx*ny*9, dtype=DTYPE) for _ in range(3)]
        self.f_new = [cuda.device_array(nx*ny*9, dtype=DTYPE) for _ in range(3)]
        
        # For visualization
        self.density = [cuda.device_array((ny, nx), dtype=DTYPE) for _ in range(3)]
        
        # Grid and block dimensions
        self.blockdim = (16, 16)
        self.griddim = ((nx + self.blockdim[0] - 1)//self.blockdim[0],
                        (ny + self.blockdim[1] - 1)//self.blockdim[1])
        
        # Initialize random number generator states if using stochastic collision
        if self.use_stochastic:
            # Use a smaller seed value to avoid int32 overflow
            seed = np.random.randint(0, 2**31 - 1)
            self.rng_states = create_xoroshiro128p_states(nx * ny, seed=seed)
           
    def initialize_from_image(self, rgb_image):
        """Initialize LBM distributions from RGB image data using GPU"""
        height, width, _ = rgb_image.shape
        assert height == self.ny and width == self.nx, "Image dimensions must match LBM grid"
        
        # Upload image data to GPU once
        d_image = cuda.to_device(rgb_image)
        
        # Initialize distributions for each channel
        for channel in range(3):
            init_from_image_kernel[self.griddim, self.blockdim](
                d_image, self.f[channel], channel, self.nx, self.ny)

    def step(self):
        """Perform one LBM diffusion timestep for all channels"""
        for channel in range(3):
            # 1) Collision (with or without stochastic term)
            if self.use_stochastic:
                diffusion_collision_kernel_stochastic[self.griddim, self.blockdim](
                    self.f[channel], self.omega, self.omega_noise, self.rng_states, self.nx, self.ny)
            else:
                diffusion_collision_kernel[self.griddim, self.blockdim](
                    self.f[channel], self.omega, self.nx, self.ny)
            
            # 2) Streaming with periodic boundary
            streaming_kernel_periodic[self.griddim, self.blockdim](
                self.f[channel], self.f_new[channel], self.nx, self.ny)
            
            # 3) Swap buffers
            self.f[channel], self.f_new[channel] = self.f_new[channel], self.f[channel]

    def run(self, steps):
        """Run diffusion for specified number of steps"""
        for _ in range(steps):
            self.step()
    
    def get_result_image(self):
        """Get current state as RGB image"""
        # Compute density for each channel
        for channel in range(3):
            compute_density_kernel[self.griddim, self.blockdim](
                self.f[channel], self.density[channel], self.nx, self.ny)
        
        # Copy back to host
        r = self.density[0].copy_to_host()
        g = self.density[1].copy_to_host()
        b = self.density[2].copy_to_host()
        rgb = np.stack((r, g, b), axis=-1)
        return rgb

def main():
    # Load the image
    name = 'img35'
    suffix = 'png'
    img = Image.open(f'assets/{name}.{suffix}')
    rgb_img = img.convert('RGB')
    rgb_values = np.array(rgb_img) / 255.0  # Normalize to [0, 1]
    
    print(f"Image shape: {rgb_values.shape}")
    
    # Create LBM solver
    nx, ny = rgb_values.shape[1], rgb_values.shape[0]
    
    # Diffusion parameters
    omega = DTYPE(0.01)  # Value between 0 and 2, smaller is more diffusive
    omega_noise = DTYPE(0.01)  # Stochastic term strength (adjust as needed)
    use_stochastic = True  # Enable stochastic collision
    
    # Load and initialize the solver
    solver = LBMDiffusionSolver(nx, ny, omega, omega_noise, use_stochastic)
    solver.initialize_from_image(rgb_values)

    # Define checkpoints for visualization
    checkpoints = [10, 50, 100, 200, 500, 600, 700, 800, 900, 1000]
    
    # Create figure for visualization
    plt.figure(figsize=(12, 8))
    
    # Plot original image
    plt.subplot(1, len(checkpoints)+1, 1)
    plt.imshow(rgb_values)
    plt.title("Original")
    plt.axis('off')
    
    # Run forward diffusion with checkpoints
    results = []
    current_step = 0
    
    for i, step in enumerate(checkpoints):
        # Run diffusion until this checkpoint
        steps_to_run = step - current_step
        solver.run(steps_to_run)
        current_step = step
        
        # Get and save result
        result = solver.get_result_image()
        results.append(result)
        
        # Plot the result
        plt.subplot(1, len(checkpoints)+1, i+2)
        plt.imshow(np.clip(result, 0, 1))
        plt.title(f"Forward: {step} steps")
        plt.axis('off')
    
    # Save the figure
    import os
    os.makedirs("bin", exist_ok=True)
    plt.tight_layout()
    stochastic_suffix = "_stochastic" if use_stochastic else ""
    plt.savefig(f"bin/diffusion_result_{checkpoints[-1]}_{name}_{omega}{stochastic_suffix}.{suffix}")
    plt.show()

    # Plot histograms of original and final RGB values
    plt.figure(figsize=(15, 10))
    
    # Channel names and colors for plotting
    channels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    
    # Plot histograms for the original image
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.hist(rgb_values[:,:,i].flatten(), bins=256, range=(0,1), 
                 color=colors[i], alpha=0.7)
        plt.title(f"Original - {channels[i]} Channel")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    
    final = results[-1]
    # Plot histograms for the final diffused image
    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.hist(final[:,:,i].flatten(), bins=256, range=(0,1), 
                 color=colors[i], alpha=0.7)
        plt.title(f"After {checkpoints[-1]} steps - {channels[i]} Channel")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    
    # Calculate and display mean and variance statistics
    print("\nRGB Channel Statistics:")
    print("------------------------")
    
    # Calculate statistics for each channel
    for i, channel in enumerate(channels):
        # Original image statistics
        original_values = rgb_values[:,:,i].flatten()
        original_mean = np.mean(original_values)
        original_var = np.var(original_values)
        
        # Final image statistics
        final_values = final[:,:,i].flatten()
        final_mean = np.mean(final_values)
        final_var = np.var(final_values)
        
        # Print statistics
        print(f"{channel} Channel:")
        print(f"  Original: Mean = {original_mean:.8f}, Variance = {original_var:.8f}")
        print(f"  After {checkpoints[-1]} steps: Mean = {final_mean:.8f}, Variance = {final_var:.8f}")
        print(f"  Change: Mean Δ = {final_mean - original_mean:.8f}, Variance Δ = {final_var - original_var:.8f}")
        print()
        
        # Add text annotations to the histogram plots
        plt.subplot(2, 3, i+1)
        plt.text(0.05, 0.95, f"Mean: {original_mean:.8f}\nVar: {original_var:.8f}", 
                 transform=plt.gca().transAxes, va='top', fontsize=9)
        
        plt.subplot(2, 3, i+4)
        plt.text(0.05, 0.95, f"Mean: {final_mean:.8f}\nVar: {final_var:.8f}", 
                 transform=plt.gca().transAxes, va='top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"bin/rgb_histograms_{checkpoints[-1]}_{name}_{omega}_{omega_noise}{stochastic_suffix}.{suffix}")
    
    print(f"Diffusion simulation completed and saved (Stochastic: {use_stochastic})")
    print(f"Omega: {omega}, Omega_noise: {omega_noise}")

if __name__ == "__main__":
    main()