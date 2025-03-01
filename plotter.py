import numpy as np
import matplotlib.pyplot as plt
from kernel_cpu import D2Q9
import vtk

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

 
def save_velocity_field_vti(ux, uy, filename):
    nx, ny = ux.shape[0], ux.shape[1]
    
    # Create a vtkImageData object
    image_data = vtk.vtkImageData()
    # Swap nx and ny in dimensions
    image_data.SetDimensions(ny, nx, 1)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    
    # Create a vtkFloatArray for the velocity vectors
    velocity = vtk.vtkFloatArray()
    velocity.SetNumberOfComponents(3)
    velocity.SetName("velocity")
    
    # Transpose and combine ux and uy into a single array
    ux_t = ux.T.flatten()
    uy_t = uy.T.flatten()
    
    # Combine ux and uy into a single array with 3 components (ux, uy, 0)
    for i in range(nx * ny):
        velocity.InsertNextTuple((ux_t[i], uy_t[i], 0.0))
    
    # Add the velocity array to the vtkImageData object
    image_data.GetPointData().SetVectors(velocity)
    
    # Write the vtkImageData to a .vti file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()