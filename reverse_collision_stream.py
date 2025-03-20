from diffusion import diffusion_collision_kernel, streaming_kernel_periodic, idx, cx_const, cy_const, w_const
from reverse_collision import reverse_collision_kernel
from reverse_pull import streaming_kernel_periodic_reverse
import numpy as np
from numba import cuda

# do collision streaming and reverse collision streaming
nx, ny = 32, 32
omega = 0.1
num_steps = 10
# Use flat array instead of 3D array
f_in = np.random.randn(nx*ny*9).astype(np.float32)  # Changed from (nx, ny, 9) to flat array
f_in_gpu = cuda.to_device(f_in)

# Output should also be flat
f_out_gpu = cuda.device_array(nx*ny*9)  
#f_out_gpu_reverse = cuda.device_array(nx*ny*9)  

threadsperblock = (16, 16)
blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

for step in range(num_steps):
    diffusion_collision_kernel[blockspergrid, threadsperblock](f_in_gpu, omega, nx, ny)
    streaming_kernel_periodic[blockspergrid, threadsperblock](f_in_gpu, f_out_gpu, nx, ny)
    f_in_gpu, f_out_gpu = f_out_gpu, f_in_gpu
for step in range(num_steps):
    streaming_kernel_periodic_reverse[blockspergrid, threadsperblock](f_in_gpu, f_out_gpu, nx, ny)
    f_in_gpu, f_out_gpu = f_out_gpu, f_in_gpu
    reverse_collision_kernel[blockspergrid, threadsperblock](f_in_gpu, omega, nx, ny)

f_out = f_in_gpu.copy_to_host()
print(f"Original f_in: {f_in}")
print(f"Reversion f_out: {f_out}")
print(f"Arrays are exactly equal: {np.array_equal(f_in, f_out)}")
rtol_value = 1e-5
print(f"Arrays are approximately equal with relative tolerance {rtol_value}: {np.allclose(f_in, f_out, rtol=rtol_value)}")