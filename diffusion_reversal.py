from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda
import os

# Use constants and helper functions from kernel_gpu.py
from kernel_gpu import DTYPE
from diffusion import diffusion_collision_kernel, streaming_kernel_periodic, compute_density_kernel, init_from_image_kernel, idx_host
from reverse_collision import diffusion_collision_kernel_reverse
from reverse_pull import streaming_kernel_periodic_reverse

class LBMDiffusionReversalSolver:
    def __init__(self, nx, ny, omega):
        self.nx = nx
        self.ny = ny
        self.omega = DTYPE(omega)
        
        # GPU arrays for 3 channels (R,G,B) - both for the initial state and current state
        self.f_initial = [cuda.device_array(nx*ny*9, dtype=DTYPE) for _ in range(3)]
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
            
            # Make a copy of the initial state
            cuda.to_device(self.f[channel].copy_to_host(), to=self.f_initial[channel])

    def forward_step(self):
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

    def reverse_step(self):
        """Perform one LBM reverse diffusion timestep for all channels"""
        for channel in range(3):
            # 1) Reverse streaming with periodic boundary
            streaming_kernel_periodic_reverse[self.griddim, self.blockdim](
                self.f[channel], self.f_new[channel], self.nx, self.ny)
            
            # 2) Swap buffers
            self.f[channel], self.f_new[channel] = self.f_new[channel], self.f[channel]
            
            # 3) Reverse collision
            diffusion_collision_kernel_reverse[self.griddim, self.blockdim](
                self.f[channel], self.omega, self.nx, self.ny)

    def run_forward(self, steps):
        """Run forward diffusion for specified number of steps"""
        for _ in range(steps):
            self.forward_step()
    
    def run_reverse(self, steps):
        """Run reverse diffusion for specified number of steps"""
        for _ in range(steps):
            self.reverse_step()
    
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

    def calculate_error_metrics(self):
        """Calculate error metrics between initial and current state"""
        error_metrics = []
        
        for channel in range(3):
            # Get the initial and current distributions
            f_initial = self.f_initial[channel].copy_to_host()
            f_current = self.f[channel].copy_to_host()
            
            # Calculate error metrics
            l2_error = np.linalg.norm(f_initial - f_current)
            relative_l2_error = l2_error / np.linalg.norm(f_initial)
            max_error_idx = np.argmax(np.abs(f_initial - f_current))
            max_abs_error = np.abs(f_initial[max_error_idx] - f_current[max_error_idx])
            max_rel_error = max_abs_error / np.abs(f_initial[max_error_idx]) if f_initial[max_error_idx] != 0 else 0
            
            error_metrics.append({
                'l2_error': l2_error,
                'relative_l2_error': relative_l2_error,
                'max_abs_error': max_abs_error,
                'max_rel_error': max_rel_error
            })
        
        return error_metrics

def plot_histograms(original, diffused, reversed_img, image_name, checkpoint, omega):
    """
    Plot histograms of pixel intensity values for original, diffused and reversed images
    with mean and variance statistics
    """
    plt.figure(figsize=(15, 10))
    channels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    
    for i, (channel, color) in enumerate(zip(channels, colors)):
        plt.subplot(3, 1, i+1)
        
        # Get flattened channel data
        orig_data = original[:,:,i].flatten()
        diff_data = diffused[:,:,i].flatten()
        rev_data = reversed_img[:,:,i].flatten()
        
        # Calculate statistics
        orig_mean, orig_var = np.mean(orig_data), np.var(orig_data)
        diff_mean, diff_var = np.mean(diff_data), np.var(diff_data)
        rev_mean, rev_var = np.mean(rev_data), np.var(rev_data)
        
        # Calculate histograms for this channel
        hist_orig, bins = np.histogram(orig_data, bins=50, range=(0, 1))
        hist_diff, _ = np.histogram(diff_data, bins=bins)
        hist_rev, _ = np.histogram(rev_data, bins=bins)
        
        # Plot histograms
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(bin_centers, hist_orig, alpha=0.7, color=color, linestyle='-', 
                 linewidth=2, label=f'Original (μ={orig_mean:.8f}, σ²={orig_var:.8f})')
        plt.plot(bin_centers, hist_diff, alpha=0.7, color='black', linestyle='--', 
                 linewidth=2, label=f'After Diffusion (μ={diff_mean:.8f}, σ²={diff_var:.8f})')
        plt.plot(bin_centers, hist_rev, alpha=0.7, color='purple', linestyle=':', 
                 linewidth=2, label=f'After Reversal (μ={rev_mean:.8f}, σ²={rev_var:.8f})')
     
        plt.title(f'{channel} Channel')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.suptitle(f'RGB Histograms Comparison - {checkpoint} Steps', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"bin/histograms_{image_name}_{checkpoint}_{omega}.png", dpi=300, bbox_inches='tight')

def analyze_distribution_functions(f_data, nx, ny, channel_idx=0, stage_name="", save_prefix=""):
    """
    Analyze the distribution functions (f0-f8) from saved data
    
    Parameters:
    - f_data: The flattened distribution functions array (already copied to host)
    - nx, ny: Grid dimensions
    - channel_idx: Which RGB channel this is (0=Red, 1=Green, 2=Blue)
    - stage_name: Name of the current stage for labeling
    - save_prefix: Prefix for saving plots
    """
    # Statistics storage
    stats = []
    
    # Create figure for histograms
    plt.figure(figsize=(15, 12))
    
    # For each distribution function
    for k in range(9):
        # Extract this distribution function across the grid
        f_k = np.zeros((ny, nx))
        for j in range(ny):
            for i in range(nx):
                idx_value = idx_host(i, j, k, nx, ny)
                f_k[j, i] = f_data[idx_value]
        
        # Flatten for histogram
        f_k_flat = f_k.flatten()
        
        # Calculate statistics
        mean_val = np.mean(f_k_flat)
        var_val = np.var(f_k_flat)
        stats.append((mean_val, var_val))
        
        # Plot histogram
        plt.subplot(3, 3, k+1)
        hist, bins = np.histogram(f_k_flat, bins=50)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(bin_centers, hist)
        plt.title(f'f{k} (μ={mean_val:.8f}, σ²={var_val:.8f})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
    
    # Finish plot and save
    plt.suptitle(f'Distribution Functions Analysis - {stage_name}\n'
                 f'Channel: {["Red", "Green", "Blue"][channel_idx]}', fontsize=16)
    plt.tight_layout()
    
    if save_prefix:
        os.makedirs("bin", exist_ok=True)
        plt.savefig(f"bin/{save_prefix}_f_analysis_{channel_idx}.png", dpi=300, bbox_inches='tight')
    
    return stats

def main():
    # Load the image
    image_name = 'img35'
    suffix = 'png'
    try:
        img = Image.open(f'assets/{image_name}.{suffix}')
    except:
        print(f"Image {image_name}.{suffix} not found. Using a simple test image.")
        # Create a simple test image if the specified image is not found
        img = Image.new('RGB', (256, 256), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([(64, 64), (192, 192)], fill='red')
        draw.ellipse([(96, 96), (160, 160)], fill='blue')
    
    rgb_img = img.convert('RGB')
    rgb_values = np.array(rgb_img) / 255.0  # Normalize to [0, 1]
    
    print(f"Image shape: {rgb_values.shape}")
    
    # Create LBM solver
    nx, ny = rgb_values.shape[1], rgb_values.shape[0]
    
    # Diffusion coefficient ~ (1/omega - 0.5)/3
    omega = 0.01  # Value between 0 and 2, smaller is more diffusive
    
    # Load and initialize the solver
    solver = LBMDiffusionReversalSolver(nx, ny, omega)
    solver.initialize_from_image(rgb_values)

    # Save distribution functions for analysis
    f_initial_red = solver.f[0].copy_to_host()
    
    # Define checkpoints for visualization
    checkpoints = [200, 600, 1000]  # Points at which to visualize
    
    # Create figure for visualization
    plt.figure(figsize=(15, 10))
    
    # Plot original image
    plt.subplot(2, len(checkpoints)+1, 1)
    plt.imshow(rgb_values)
    plt.title("Original")
    plt.axis('off')
    
    # Run forward diffusion with checkpoints
    results = []
    current_step = 0
    
    for i, step in enumerate(checkpoints):
        # Run diffusion until this checkpoint
        steps_to_run = step - current_step
        solver.run_forward(steps_to_run)
        current_step = step
        
        # Get and save result
        result = solver.get_result_image()
        results.append(result)
        
        # Plot the result
        plt.subplot(2, len(checkpoints)+1, i+2) # (Number of rows = 2, Number of columns, Position)
        plt.imshow(np.clip(result, 0, 1))
        plt.title(f"Forward: {step} steps")
        plt.axis('off')
    
    # Get the final forward diffusion result (should be the same as results[-1])
    final_forward_result = solver.get_result_image()

    # Get distribution functions after forward diffusion
    forward_complete_red = solver.f[0].copy_to_host()

    # Plot this as the starting point for reverse diffusion (second row, first column)
    plt.subplot(2, len(checkpoints)+1, len(checkpoints)+2)  # (Number of rows = 2, Number of columns, Position = Second row and first column)
    plt.imshow(np.clip(final_forward_result, 0, 1))
    plt.title(f"Start Reversal: {checkpoints[-1]} steps")
    plt.axis('off')

    # Now run reverse diffusion
    reverse_results = []
    reversed_checkpoints = list(reversed(checkpoints))

    for i in range(len(reversed_checkpoints)):
        # Calculate the correct number of steps to run in reverse
        if i == 0:
            # First reverse step: from last checkpoint to the one before
            steps_to_run = reversed_checkpoints[0] - reversed_checkpoints[1] if len(reversed_checkpoints) > 1 else reversed_checkpoints[0]
        elif i == len(reversed_checkpoints) - 1:
            # Last reverse step: from first checkpoint back to 0
            steps_to_run = reversed_checkpoints[-1]
        else:
            # Intermediate steps: difference between consecutive checkpoints
            steps_to_run = reversed_checkpoints[i] - reversed_checkpoints[i+1]

        # Run reverse diffusion
        solver.run_reverse(steps_to_run)
        
        # Get and save result
        result = solver.get_result_image()
        reverse_results.append(result)
        
        # Plot the result
        plt.subplot(2, len(checkpoints)+1, len(checkpoints)+3+i)  # (Number of rows = 2, Number of columns, Position starting from second row and second column)
        plt.imshow(np.clip(result, 0, 1))
        
        if i == len(reversed_checkpoints) - 1:
            plt.title(f"Reverse: 0 step (Final)")
        else:
            plt.title(f"Reverse: {reversed_checkpoints[i+1]} steps")
        plt.axis('off')
    
    final_result = reverse_results[-1]

    # Save distribution functions for analysis
    reversed_complete_red = solver.f[0].copy_to_host()
    
    # Calculate error metrics
    error_metrics = solver.calculate_error_metrics()
    
    # Print error metrics
    print("\nError Metrics After Full Reversal:")
    print("----------------------------------")
    channels = ['Red', 'Green', 'Blue']
    for i, channel in enumerate(channels):
        print(f"{channel} Channel:")
        print(f"  L2 Error: {error_metrics[i]['l2_error']:.8f}")
        print(f"  Relative L2 Error: {error_metrics[i]['relative_l2_error']:.8f}")
        print(f"  Maximum Absolute Error: {error_metrics[i]['max_abs_error']:.8f}")
        print(f"  Maximum Relative Error: {error_metrics[i]['max_rel_error']:.8f}")
        print()
    
    # Calculate and plot MSE between original and final
    mse = np.mean((rgb_values - final_result)**2)
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    plt.suptitle(f"Diffusion and Reversal Process\nMSE: {mse:.8f}, PSNR: {psnr:.2f} dB")
    plt.tight_layout()
    os.makedirs("bin", exist_ok=True)
    plt.savefig(f"bin/diffusion_reversal_{checkpoints[-1]}_{image_name}_{omega}.{suffix}")
    plt.show()
    
    # Get the final forward diffusion and reversal results
    #final_forward_result = results[-1]  # Last result from forward diffusion
    #final_result = reverse_results[-1]  # Last result from reversal process

    # Save the final images
    Image.fromarray((final_forward_result * 255).astype(np.uint8)).save(f"bin/final_forward_{checkpoints[-1]}_{image_name}_{omega}.{suffix}")
    Image.fromarray((final_result * 255).astype(np.uint8)).save(f"bin/final_reversed_{checkpoints[-1]}_{image_name}_{omega}.{suffix}")

    # Plot histograms comparing all three stages
    plot_histograms(rgb_values, final_forward_result, final_result, image_name, checkpoints[-1], omega)

    # Analyze distribution functions after diffusion
    print("\nAnalyzing distribution functions...")
    analyze_distribution_functions(
        f_initial_red, nx, ny,
        channel_idx=0, 
        stage_name="Initial State", 
        save_prefix=f"{image_name}_initial"
    )

    analyze_distribution_functions(
        forward_complete_red, nx, ny,
        channel_idx=0, 
        stage_name=f"After {checkpoints[-1]} Diffusion Steps", 
        save_prefix=f"{image_name}_diffused_{checkpoints[-1]}_{omega}"
    )

    analyze_distribution_functions(
        reversed_complete_red, nx, ny,
        channel_idx=0, 
        stage_name="After Complete Reversal", 
        save_prefix=f"{image_name}_reversed_{checkpoints[-1]}_{omega}"
    )

if __name__ == "__main__":
    # Ensure necessary imports
    try:
        from PIL import ImageDraw
    except ImportError:
        pass
    
    main()