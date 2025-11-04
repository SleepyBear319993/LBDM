import numpy as np
from numba import cuda

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

# LBM kernels for diffusion
@cuda.jit(fastmath=True)
def diffusion_collision_kernel(f, omega, nx, ny):
    """Simplified collision for diffusion (no velocity components)"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Compute density
        rho = DTYPE(0.0)
        for k in range(9):
            rho += f[idx(i, j, k, nx, ny)]
        
        # Collision step for diffusion (no velocity component)
        for k in range(9):
            feq = w_const[k] * rho  # Equilibrium for diffusion
            f[idx(i, j, k, nx, ny)] = (DTYPE(1.0) - omega) * f[idx(i, j, k, nx, ny)] + omega * feq

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
        rho = DTYPE(0.0)
        for k in range(9):
            rho += f[idx(i, j, k, nx, ny)]
        density[j, i] = rho  # Note: density is in [j,i] order to match image convention

# Add this kernel to your existing code
@cuda.jit
def init_from_image_kernel(image_data, f, channel, nx, ny):
    """Initialize distribution functions from image data for a specific channel directly on GPU"""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Get normalized channel value (image data is in [j,i] suffix)
        density = DTYPE(image_data[j, i, channel])# / 255.0
        
        # Initialize distributions with equilibrium for zero velocity
        for k in range(9):
            f[idx(i, j, k, nx, ny)] = w_const[k] * density

class LBMDiffusionSolver:
    def __init__(self, nx, ny, omega):
        self.nx = nx
        self.ny = ny
        self.omega = DTYPE(omega)
        
        # GPU arrays for 3 channels (R,G,B)
        self.f = [cuda.device_array(nx*ny*9, dtype=DTYPE) for _ in range(3)]
        self.f_new = [cuda.device_array(nx*ny*9, dtype=DTYPE) for _ in range(3)]
        
        # For visualization
        self.density = [cuda.device_array((ny, nx), dtype=DTYPE) for _ in range(3)]
        
        # Grid and block dimensions
        self.blockdim = (16, 16)
        self.griddim = ((nx + self.blockdim[0] - 1)//self.blockdim[0],
                        (ny + self.blockdim[1] - 1)//self.blockdim[1])
           
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
            # 1) Collision
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