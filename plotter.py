import numpy as np
import matplotlib.pyplot as plt
import vtk
    
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