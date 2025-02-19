import numpy as np
import time
from kernel_cpu import LatticeInitializerD2Q9
from plotter import plot_velocity_field

def run_test_case():
    nx, ny = 50, 50
    rho = 1.0
    u = np.array([0.1, 0.2])
    omega = 0.55
    num_steps = 100

    initializer = LatticeInitializerD2Q9(nx, ny)
    initializer.initialize(rho, u)
    
    start_time = time.time()
    
    for step in range(num_steps):
        initializer.collision(omega)
        initializer.streaming()
        initializer.bounce_back()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total computation time: {total_time:.4f} seconds")
    # Plot the velocity field
    plot_velocity_field(initializer)

if __name__ == '__main__':
    run_test_case()
