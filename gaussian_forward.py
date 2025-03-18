import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the image
image_path = 'girl.png'
img = np.array(Image.open(image_path).convert('RGB')) / 255.0  # Normalize to [0, 1]

# Calculate initial statistics
initial_mean = img.mean()
initial_variance = img.var()
print(f"Initial mean: {initial_mean:.4f}, Initial variance: {initial_variance:.4f}")

# Forward diffusion process parameters
num_steps = 1000
beta_min = 1e-4
beta_max = 0.02
betas = np.linspace(beta_min, beta_max, num_steps)

# Calculate alphas and cumulative product for direct step calculation
alphas = 1 - betas
alpha_cumprod = np.cumprod(alphas)
alpha_cumprod_t = alpha_cumprod[num_steps-1]  # Use the final timestep
print(f"sqrt Alpha cumulative product: {np.sqrt(alpha_cumprod_t):.4f}")
print(f"sqrt 1-alpha_cumprod: {np.sqrt(1 - alpha_cumprod_t):.4f}")
print(f"1-alpha_cumprod: {1 - alpha_cumprod_t:.4f}")

# Apply forward process in one step (equivalent to running all iterations)
noise = np.random.randn(*img.shape)
x_t = np.sqrt(alpha_cumprod_t) * img + np.sqrt(1 - alpha_cumprod_t) * noise

# Calculate final statistics
final_mean = x_t.mean()
final_variance = x_t.var()
print(f"Final mean: {final_mean:.4f}, Final variance: {final_variance:.4f}")

# Visualize results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title(f'Original Image\nMean: {initial_mean:.4f}, Var: {initial_variance:.4f}')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.clip(x_t, 0, 1))  # Clip for display
plt.title(f'After Forward Process\nMean: {final_mean:.4f}, Var: {final_variance:.4f}')
plt.axis('off')

plt.tight_layout()
plt.savefig('bin/forward_process_result.png')
plt.show()