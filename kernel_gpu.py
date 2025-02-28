import numpy as np
from numba import cuda

#--------------------------------------------------------------------
# Global constants for D2Q9 (Python tuples are accessible inside JITed kernels)
#--------------------------------------------------------------------
cx_const = (0,  1,  0, -1,  0,  1, -1, -1,  1)
cy_const = (0,  0,  1,  0, -1,  1,  1, -1, -1)
w_const  = (4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0)

#--------------------------------------------------------------------
# CUDA kernel: Collision Step
# Each lattice cell computes its density, velocity, and then relaxes toward equilibrium.
#--------------------------------------------------------------------
@cuda.jit
def collision_kernel(f, omega, nx, ny):
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        rho = 0.0
        u_x = 0.0
        u_y = 0.0
        # Compute macroscopic density and momentum from distribution functions
        for k in range(9):
            val = f[i, j, k]
            rho += val
            u_x += val * cx_const[k]
            u_y += val * cy_const[k]
        if rho > 0.0:
            u_x /= rho
            u_y /= rho
        usqr = u_x*u_x + u_y*u_y
        # Relaxation (BGK collision) toward equilibrium
        for k in range(9):
            cu = 3.0 * (cx_const[k] * u_x + cy_const[k] * u_y)
            feq = w_const[k] * rho * (1.0 + cu + 0.5 * cu * cu - 1.5 * usqr)
            f[i, j, k] = (1.0 - omega) * f[i, j, k] + omega * feq

#--------------------------------------------------------------------
# CUDA kernel: Streaming Step
# For interior nodes, each lattice cell “pulls” data from its neighbors.
# Boundary nodes are simply copied (their distributions will be corrected by BC kernels).
#--------------------------------------------------------------------
@cuda.jit
def streaming_kernel(f_in, f_out, nx, ny):
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # For interior nodes, perform streaming using the opposite direction.
        if (i > 0) and (i < nx - 1) and (j > 0) and (j < ny - 1):
            for k in range(9):
                ip = i - cx_const[k]
                jp = j - cy_const[k]
                f_out[i, j, k] = f_in[ip, jp, k]
        else:
            # For boundary nodes, simply copy the current value.
            for k in range(9):
                f_out[i, j, k] = f_in[i, j, k]

@cuda.jit
def streaming_kernel_periodic(f_in, f_out, nx, ny):
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        for k in range(9):
            ip = (i - cx_const[k] + nx) % nx
            jp = (j - cy_const[k] + ny) % ny
            f_out[i, j, k] = f_in[ip, jp, k]

#--------------------------------------------------------------------
# CUDA kernel: Bounce-Back on Left, Right, and Bottom Walls
# A simple (local) bounce-back that swaps populations for wall nodes.
#--------------------------------------------------------------------
@cuda.jit
def bounce_back_kernel(f, mask, nx, ny):
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        # Apply bounce-back on left (i==0), right (i==nx-1) and bottom (j==0)
        #if (i == 0) or (i == nx - 1) or (j == 0) or (j == ny - 1):
        #if (i == 1) or (i == nx - 2) or (j == 1):
        
        # Apply bounce-back at solid node:
        if mask[i, j] == 1:
            # Swap east (1) and west (3)
            tmp = f[i, j, 1]
            f[i, j, 1] = f[i, j, 3]
            f[i, j, 3] = tmp
            # Swap north (2) and south (4)
            tmp = f[i, j, 2]
            f[i, j, 2] = f[i, j, 4]
            f[i, j, 4] = tmp
            # Swap north-east (5) and south-west (7)
            tmp = f[i, j, 5]
            f[i, j, 5] = f[i, j, 7]
            f[i, j, 7] = tmp
            # Swap north-west (6) and south-east (8)
            tmp = f[i, j, 6]
            f[i, j, 6] = f[i, j, 8]
            f[i, j, 8] = tmp

#--------------------------------------------------------------------
# CUDA kernel: Moving-Lid Boundary Condition on Top Wall
# At the top boundary (j = ny - 1), we enforce a velocity U in the x-direction.
# We first “reflect” the unknown populations and then add a momentum correction.
#--------------------------------------------------------------------
@cuda.jit
def moving_lid_kernel(f, nx, ny, U):
    i = cuda.grid(1)  # iterate along x only
    if i < nx:
        #if i != 1 or i != nx - 2:
            j = ny - 2        # top row
            rho = f[i, j, 0] + f[i, j, 1] + f[i, j, 3] + 2.0 * (f[i, j, 2] + f[i, j, 5] + f[i, j, 6])
            f[i, j, 4] = f[i, j, 2]
            f[i, j, 7] = f[i, j, 5] + 0.5 * (f[i, j, 1] - f[i, j, 3]) - 0.5 * rho * U
            f[i, j, 8] = f[i, j, 6] - 0.5 * (f[i, j, 1] - f[i, j, 3]) + 0.5 * rho * U
            
            # f[i, j, 4] = f[i, j, 2] 
            # f[i, j, 7] = f[i, j, 5] - 1.0/6.0 * U
            # f[i, j, 8] = f[i, j, 6] + 1.0/6.0 * U

class LBMSolverD2Q9GPU:
    def __init__(self, nx, ny, omega, U):
        self.nx = nx
        self.ny = ny
        self.omega = np.float32(omega)
        self.U = U

        # Allocate device memory for distribution function
        self.f = cuda.device_array((nx, ny, 9), dtype=np.float32)
        self.f_new = cuda.device_array((nx, ny, 9), dtype=np.float32)
        
        # Allocate device memory for mask
        self.mask = cuda.device_array((nx, ny), dtype=np.int8)

        # Choose thread-block dimensions
        self.blockdim = (16, 16)
        self.griddim = ((nx + self.blockdim[0] - 1)//self.blockdim[0],
                        (ny + self.blockdim[1] - 1)//self.blockdim[1])
        
    def set_mask(self):
        """
        define a mask for the solid boundary
        """
        mask_host = np.zeros((self.nx, self.ny), dtype=np.int8)

        # define a circle in the center of the domain
        # cx, cy = self.nx // 2, self.ny // 2
        # r = min(cx, cy) // 2  
        # for i in range(self.nx):
        #     for j in range(self.ny):
        #         dist2 = (i - cx)**2 + (j - cy)**2
        #         if dist2 < r*r:
        #             mask_host[i, j] = 1
        
        # define a open-lid square
        for i in range(self.nx):
            for j in range(self.ny):
                if i == 1 or i == self.nx - 2 or j == 1:
                    mask_host[i, j] = 1

        # copy to device
        self.mask.copy_to_device(mask_host)

    def initialize(self, rho0=1.0, u0x=0.1, u0y=0.0):
        """
        Initialize the distribution on the CPU, then copy to GPU.
        """
        f_host = np.zeros((self.nx, self.ny, 9), dtype=np.float32)

        for i in range(self.nx):
            for j in range(self.ny):
                usq = u0x*u0x + u0y*u0y
                for k in range(9):
                    cu = 3.0*(cx_const[k]*u0x + cy_const[k]*u0y)
                    f_host[i, j, k] = w_const[k]*rho0*(1.0 + cu + 0.5*cu*cu - 1.5*usq)

        # for j in range(self.ny):
        #     if j != 1 or j != self.ny - 2:
        #         i = self.nx - 2
        #         usq = self.U*self.U
        #         for k in range(9):
        #             cu = 3.0*(cx_const[k]*self.U)
        #             f_host[i, j, k] = w_const[k]*rho0*(1.0 + cu + 0.5*cu*cu - 1.5*usq)
            
        self.f.copy_to_device(f_host)

    def step(self):
        """
        Perform one LBM timestep:
          1) collision (in-place)
          2) streaming (f -> f_new)
          3) bounce-back on f_new
          4) swap f and f_new
        """
        # 1) Collision in-place on self.f
        collision_kernel[self.griddim, self.blockdim](self.f, self.omega,
                                                      self.nx, self.ny)
        cuda.synchronize()

        # 2) Streaming: f -> f_new
        streaming_kernel[self.griddim, self.blockdim](self.f, self.f_new,
                                                      self.nx, self.ny)
        cuda.synchronize()
        
        # 2) Streaming with periodic boundary conditions
        # streaming_kernel_periodic[self.griddim, self.blockdim](self.f, self.f_new,
        #                                               self.nx, self.ny)
        # cuda.synchronize()

        # 3) Bounce-back on f_new
        bounce_back_kernel[self.griddim, self.blockdim](self.f_new, self.mask,
                                                        self.nx, self.ny)
        cuda.synchronize()
        
        # 4) Moving-lid boundary condition on top wall
        moving_lid_kernel[self.griddim, self.blockdim](self.f_new, self.nx, self.ny, self.U)
        cuda.synchronize()

        # 5) Swap
        self.f, self.f_new = self.f_new, self.f

    def stream_periodic(self):
        # Streaming with periodic boundary conditions
        streaming_kernel_periodic[self.griddim, self.blockdim](self.f, self.f_new,
                                                      self.nx, self.ny)
        cuda.synchronize()
        # Swap
        self.f, self.f_new = self.f_new, self.f


    def run(self, num_steps=1000):
        """
        Run LBM for num_steps timesteps.
        """
        for _ in range(num_steps):
            self.step()

    def get_distribution(self):
        """
        Copy the current distribution f from device to host.
        """
        return self.f.copy_to_host()

    def compute_macroscopic(self):
        """
        Optionally compute density and velocity on the host after copying f.
        """
        f_host = self.get_distribution()
        rho = np.zeros((self.nx, self.ny), dtype=np.float32)
        ux  = np.zeros((self.nx, self.ny), dtype=np.float32)
        uy  = np.zeros((self.nx, self.ny), dtype=np.float32)

        for i in range(self.nx):
            for j in range(self.ny):
                s = 0.0
                sx = 0.0
                sy = 0.0
                for k in range(9):
                    val = f_host[i, j, k]
                    s  += val
                    sx += val * cx_const[k]
                    sy += val * cy_const[k]
                rho[i,j] = s
                if s > 1e-12:
                    ux[i,j] = sx / s
                    uy[i,j] = sy / s
        return rho, ux, uy

def main():
    import time

    nx, ny = 256, 256
    omega = 0.55
    U = 0.01
    solver = LBMSolverD2Q9GPU(nx, ny, omega, U)

    # Initialize with uniform density 1.0 and velocity (0.0, 0.0)
    solver.initialize(rho0=1.0, u0x=0.0, u0y=0.0)
    
    # Set mask
    solver.set_mask()

    nsteps = 10000
    t0 = time.time()
    solver.run(nsteps)
    t1 = time.time()

    # Print performance
    elapsed = t1 - t0
    print(f"Ran {nsteps} steps in {elapsed:.3f} s, ~{nsteps/elapsed:.1f} steps/s")
    
    # MLUPS (Million Lattice Updates Per Second)
    mlups = nx*ny*nsteps / (elapsed * 1e6)
    print(f"Performance: {mlups:.2f} MLUPS")

    # Check center cell's fields
    rho, ux, uy = solver.compute_macroscopic()
    cx, cy = nx//2, ny//2
    print("Density at center:", rho[cx, cy])
    print("Velocity at center:", ux[cx, cy], uy[cx, cy])
    print("Average density", np.mean(rho))
    print("Top wall velocity", ux[cx, ny-2], uy[cx, ny-2])
    print("Top second wall velocity", ux[cx, ny-3], uy[cx, ny-3])
    
    # Plot the velocity field
    from plotter import plot_velocity_field_uxy
    plot_velocity_field_uxy(ux, uy)

if __name__ == "__main__":
    main()
