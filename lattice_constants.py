import numpy as np

# Data type macro
DTYPE = np.float32  # or np.float64 for double precision

#--------------------------------------------------------------------
# Global constants for D2Q9 (Python tuples are accessible inside JITed kernels)
#--------------------------------------------------------------------
cx_const = (0,  1,  0, -1,  0,  1, -1, -1,  1)
cy_const = (0,  0,  1,  0, -1,  1,  1, -1, -1)
w_const  = tuple(DTYPE(w) for w in (4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
                                   1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0))


def threads_per_block(dim_x=16, dim_y=16):
    """Returns standard 2D thread block dimensions."""
    return (dim_x, dim_y)

def blocks_per_grid(nx, ny, block_dim_x=16, block_dim_y=16):
    """Calculates the grid dimensions needed for a given nx, ny and block size."""
    grid_dim_x = (nx + block_dim_x - 1) // block_dim_x
    grid_dim_y = (ny + block_dim_y - 1) // block_dim_y
    return (grid_dim_x, grid_dim_y)