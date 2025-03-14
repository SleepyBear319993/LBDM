import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def main():
    # Create output directory
    os.makedirs("bin", exist_ok=True)
    
    # Load the image (same one used in diffusion.py)
    img = Image.open('girl.png')
    rgb_img = img.convert('RGB')
    rgb_values = np.array(rgb_img).astype(np.float32)
    
    print(f"Image shape: {rgb_values.shape}")
    
    # Parameters for diffusion
    num_timesteps = 1000
    max_noise_level = 0.9  # Maximum noise level at the end
    
    # Function to apply noise according to a schedule
    def apply_diffusion(image, t, max_noise):
        """Apply forward diffusion process at timestep t"""
        # Calculate noise level using a linear schedule
        # Varies from 0 (t=0) to max_noise (t=timesteps)
        noise_level = max_noise * (t / num_timesteps)
        
        # Generate Gaussian noise with same shape as image
        noise = np.random.normal(0, 1, image.shape)
        
        # Linear interpolation: (1-noise_level)*image + noise_level*noise
        noisy_image = (1 - noise_level) * image + noise_level * noise * 255
        
        # Normalize to [0, 255] range instead of clipping
        if t > 0:  # Only normalize if noise was added
            # Normalize each color channel separately
            normalized_image = np.zeros_like(noisy_image)
            for c in range(3):
                channel = noisy_image[:,:,c]
                min_val = np.min(channel)
                max_val = np.max(channel)
                if max_val > min_val:
                    normalized_image[:,:,c] = (channel - min_val) * 255 / (max_val - min_val)
                else:
                    normalized_image[:,:,c] = channel
        else:
            normalized_image = noisy_image
            
        return normalized_image.astype(np.uint8)
    
    # Select timesteps to show
    steps_to_show = [0, 50, 200, 500, 1000]
    results = {}
    
    # Apply diffusion for selected timesteps
    for t in steps_to_show:
        results[t] = apply_diffusion(rgb_values, t, max_noise_level)
    
    # Plot results
    plt.figure(figsize=(15, 8))
    for i, t in enumerate(steps_to_show):
        plt.subplot(1, len(steps_to_show), i+1)
        plt.imshow(results[t].astype(np.uint8))
        plt.title(f"t = {t}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("bin/gaussian_diffusion_results.png")
    
    # Plot histograms of original and final RGB values
    plt.figure(figsize=(15, 10))
    
    # Channel names and colors for plotting
    channels = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    
    # Plot histograms for the original image
    for i in range(3):
        plt.subplot(2, 3, i+1)
        plt.hist(rgb_values[:,:,i].flatten(), bins=256, range=(0,255), 
                 color=colors[i], alpha=0.7)
        plt.title(f"Original - {channels[i]} Channel")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    
    # Plot histograms for the final diffused image (maximum noise)
    final_t = steps_to_show[-1]
    for i in range(3):
        plt.subplot(2, 3, i+4)
        plt.hist(results[final_t][:,:,i].flatten(), bins=256, range=(0,255), 
                 color=colors[i], alpha=0.7)
        plt.title(f"After t={final_t} - {channels[i]} Channel")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("bin/gaussian_diffusion_histograms.png")
    
    # Save individual images
    for t in steps_to_show:
        Image.fromarray(results[t].astype(np.uint8)).save(f"bin/gaussian_diffused_t{t}.png")
    
    print("Gaussian diffusion completed and saved")

if __name__ == "__main__":
    main()