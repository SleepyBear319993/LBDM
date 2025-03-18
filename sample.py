import numpy as np
import matplotlib.pyplot as plt
red = 0.4291 + 0.01 * np.random.randn(64, 64)
green = 0.4415 + 0.01 * np.random.randn(64, 64)
blue = 0.3558 + 0.01 * np.random.randn(64, 64)
print("red mean: ", red.mean())
print("red var: ", red.var())
pic = np.stack([red, green, blue], axis=-1)
#pic = 0.5 + 0.004 * np.random.randn(500, 500, 3)
plt.imshow(pic)
plt.show()
# Plot rgb histograms
plt.figure(figsize=(15, 10))
channels = ['Red', 'Green', 'Blue']
colors = ['red', 'green', 'blue']
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.hist(pic[:,:,i].flatten(), bins=256, range=(0,1), color=colors[i], alpha=0.7)
    plt.title(f"Original - {channels[i]} Channel")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
plt.show()
# Calculate and display mean and variance statistics
print("\nRGB Channel Statistics:")
print("------------------------")
for i, channel in enumerate(channels):
    original_values = pic[:,:,i].flatten()
    original_mean = np.mean(original_values)
    original_var = np.var(original_values)
    print(f"{channel} Channel:")
    print(f"Mean: {original_mean:.4f}, Variance: {original_var:.4f}")
    
