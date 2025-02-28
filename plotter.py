import numpy as np
import matplotlib.pyplot as plt
from kernel_cpu import D2Q9

def plot_velocity_field(initializer):
    nx, ny = initializer.nx, initializer.ny
    u_x = np.zeros((nx, ny))
    u_y = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            u_x[i, j] = np.sum(initializer.f[i, j, :] * D2Q9.cx) / np.sum(initializer.f[i, j, :])
            u_y[i, j] = np.sum(initializer.f[i, j, :] * D2Q9.cy) / np.sum(initializer.f[i, j, :])

    X, Y = np.meshgrid(range(nx), range(ny), indexing='ij')
    plt.quiver(X, Y, u_x, u_y)
    plt.title("Velocity Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
def plot_velocity_field_uxy(ux, uy):
    nx, ny = ux.shape[0], ux.shape[1]
    X, Y = np.meshgrid(range(nx), range(ny), indexing='ij')
    plt.quiver(X, Y, ux, uy)
    plt.title("Velocity Field")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()