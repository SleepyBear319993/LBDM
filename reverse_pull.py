from diffusion import streaming_kernel_periodic, idx, cx_const, cy_const
import numpy as np
from numba import cuda

# @cuda.jit
# def streaming_kernel_periodic(f_in, f_out, nx, ny):
#     """Streaming with periodic boundary conditions"""
#     i, j = cuda.grid(2)
#     if i < nx and j < ny:
#         for k in range(9):
#             # Periodic boundary conditions
#             ip = (i - cx_const[k] + nx) % nx
#             jp = (j - cy_const[k] + ny) % ny
#             f_out[idx(i, j, k, nx, ny)] = f_in[idx(ip, jp, k, nx, ny)]

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

# do streaming
nx, ny = 5, 5
# Use flat array instead of 3D array
f_in = np.random.randn(nx*ny*9).astype(np.float32)  # Changed from (nx, ny, 9) to flat array
f_in_gpu = cuda.to_device(f_in)

# Output should also be flat
f_out_gpu = cuda.device_array(nx*ny*9)  
f_out_gpu_reverse = cuda.device_array(nx*ny*9)  

threadsperblock = (16, 16)
blockspergrid_x = (nx + threadsperblock[0] - 1) // threadsperblock[0]
blockspergrid_y = (ny + threadsperblock[1] - 1) // threadsperblock[1]
blockspergrid = (blockspergrid_x, blockspergrid_y)

streaming_kernel_periodic[blockspergrid, threadsperblock](f_in_gpu, f_out_gpu, nx, ny)
f_in_gpu, f_out_gpu = f_out_gpu, f_in_gpu
streaming_kernel_periodic[blockspergrid, threadsperblock](f_in_gpu, f_out_gpu, nx, ny)
streaming_kernel_periodic_reverse[blockspergrid, threadsperblock](f_out_gpu, f_out_gpu_reverse, nx, ny)
f_out_gpu, f_out_gpu_reverse = f_out_gpu_reverse, f_out_gpu
streaming_kernel_periodic_reverse[blockspergrid, threadsperblock](f_out_gpu, f_out_gpu_reverse, nx, ny)

# copy the data back to the CPU
f_out_reverse = f_out_gpu_reverse.copy_to_host()

# check if f_out_reverse after reversed streaming is equal to f_in
print(np.array_equal(f_in, f_out_reverse))

# print f_in and f_out_reverse
#print(f_in)
#print(f_out_reverse)