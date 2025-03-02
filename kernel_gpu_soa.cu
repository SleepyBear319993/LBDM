#include <stdio.h>
#include <math.h>
#include <chrono>
// for vti output
#include <fstream>
#include <iomanip>

//-----------------------------------------------------
// Lattice parameters and simulation constants
//-----------------------------------------------------
const int nx = 256;
const int ny = 256;
const int numDirs = 9;
typedef float DTYPE;

__device__ int cx_const[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
__device__ int cy_const[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
__device__ DTYPE w_const[9] = {
    4.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1 };
int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1 };
DTYPE w[9] = {
    4.0f/9.0f,
    1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};

DTYPE U = 0.1f;
DTYPE Re = 1000.0f;
DTYPE nu, omega;  // nu = 3*(U*nx/Re)+0.5; omega = 1/nu

// Device pointers for SoA layout
DTYPE **d_f = nullptr, **d_f_new = nullptr;
char *d_mask = nullptr;


//-----------------------------------------------------
// CUDA kernels for the LBM solver
//-----------------------------------------------------

// Collision kernel: computes macroscopic variables and relaxes toward equilibrium.
__global__ void collision_kernel(DTYPE **f, DTYPE omega, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        DTYPE rho = 0.0f;
        DTYPE u_x = 0.0f;
        DTYPE u_y = 0.0f;
        
        // Compute macroscopic variables
        for (int k = 0; k < numDirs; k++) {
            DTYPE val = f[k][i + j*nx];
            rho += val;
            u_x += val * cx_const[k];
            u_y += val * cy_const[k];
        }
        
        if (rho > 0.0f) {
            u_x /= rho;
            u_y /= rho;
        }
        
        DTYPE usqr = u_x*u_x + u_y*u_y;
        for (int k = 0; k < numDirs; k++) {
            DTYPE cu = 3.0f * (cx_const[k]*u_x + cy_const[k]*u_y);
            DTYPE feq = w_const[k] * rho * (1.0f + cu + 0.5f*cu*cu - 1.5f*usqr);
            f[k][i + j*nx] = (1.0f - omega) * f[k][i + j*nx] + omega * feq;
        }
    }
}

// Streaming kernel: "pull" from neighbor nodes.
__global__ void streaming_kernel(DTYPE **f_in, DTYPE **f_out, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
            for (int k = 0; k < numDirs; k++) {
                int ip = i - cx_const[k];
                int jp = j - cy_const[k];
                f_out[k][i + j*nx] = f_in[k][ip + jp*nx];
            }
        } else {
            for (int k = 0; k < numDirs; k++) {
                f_out[k][i + j*nx] = f_in[k][i + j*nx];
            }
        }
    }
}

// Bounce-back kernel modified for SoA
__global__ void bounce_back_kernel(DTYPE **f, char* mask, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        if (mask[i + j*nx] == 1) {
            int idx = i + j*nx;
            // Swap east (1) and west (3)
            DTYPE tmp = f[1][idx];
            f[1][idx] = f[3][idx];
            f[3][idx] = tmp;
            
            // Swap north (2) and south (4)
            tmp = f[2][idx];
            f[2][idx] = f[4][idx];
            f[4][idx] = tmp;
            
            // Swap north-east (5) and south-west (7)
            tmp = f[5][idx];
            f[5][idx] = f[7][idx];
            f[7][idx] = tmp;
            
            // Swap north-west (6) and south-east (8)
            tmp = f[6][idx];
            f[6][idx] = f[8][idx];
            f[8][idx] = tmp;
        }
    }
}

// Moving-lid kernel modified for SoA
__global__ void moving_lid_kernel(DTYPE **f, int nx, int ny, DTYPE U) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        int j = ny - 2;
        int idx = i + j*nx;
        
        DTYPE rho = f[0][idx] + f[1][idx] + f[3][idx] +
                   2.0f * (f[2][idx] + f[5][idx] + f[6][idx]);
                   
        f[4][idx] = f[2][idx];
        f[7][idx] = f[5][idx] + 0.5f*(f[1][idx] - f[3][idx]) - 0.5f*rho*U;
        f[8][idx] = f[6][idx] - 0.5f*(f[1][idx] - f[3][idx]) + 0.5f*rho*U;
    }
}

// Velocity field computation kernel modified for SoA
__global__ void compute_velocity_field_kernel(DTYPE **f, DTYPE* velocity_mag, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        int idx = i + j*nx;
        DTYPE rho = 0.0f;
        DTYPE u_x = 0.0f;
        DTYPE u_y = 0.0f;
        
        for (int k = 0; k < numDirs; k++) {
            DTYPE val = f[k][idx];
            rho += val;
            u_x += val * cx_const[k];
            u_y += val * cy_const[k];
        }
        
        if (rho > 1e-12f) {
            u_x /= rho;
            u_y /= rho;
        }
        
        DTYPE vel = sqrtf(u_x*u_x + u_y*u_y);
        // Multiply velocity by a scaling factor to amplify it for display
        // const DTYPE scale_factor = 1000.0f;
        // velocity_mag[idx] = vel * scale_factor;
    }
}

//-----------------------------------------------------
// Host routines for initialization and simulation
//-----------------------------------------------------
void initialize_simulation() {
    // Allocate host pointer arrays
    DTYPE **h_f = new DTYPE*[numDirs];
    DTYPE **h_f_new = new DTYPE*[numDirs];
    
    // Allocate device pointer arrays
    cudaMalloc(&d_f, numDirs * sizeof(DTYPE*));
    cudaMalloc(&d_f_new, numDirs * sizeof(DTYPE*));
    
    // Allocate memory for each direction
    for(int k = 0; k < numDirs; k++) {
        cudaMalloc(&h_f[k], nx * ny * sizeof(DTYPE));
        cudaMalloc(&h_f_new[k], nx * ny * sizeof(DTYPE));
        
        // Initialize values
        DTYPE* temp = new DTYPE[nx * ny];
        for(int j = 0; j < ny; j++) {
            for(int i = 0; i < nx; i++) {
                DTYPE usq = 0.0f;
                DTYPE cu = 3.0f*(cx[k]*0.0f + cy[k]*0.0f);
                temp[i + j*nx] = w[k] * 1.0f * (1.0f + cu + 0.5f*cu*cu - 1.5f*usq);
            }
        }
        cudaMemcpy(h_f[k], temp, nx * ny * sizeof(DTYPE), cudaMemcpyHostToDevice);
        delete[] temp;
    }
    
    // Copy pointer arrays to device
    cudaMemcpy(d_f, h_f, numDirs * sizeof(DTYPE*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f_new, h_f_new, numDirs * sizeof(DTYPE*), cudaMemcpyHostToDevice);
    
    // Initialize mask
    cudaMalloc(&d_mask, nx * ny * sizeof(char));
    char* h_mask = new char[nx * ny];
    for(int j = 0; j < ny; j++) {
        for(int i = 0; i < nx; i++) {
            h_mask[i + j*nx] = (i == 1 || i == nx-2 || j == 1) ? 1 : 0;
        }
    }
    cudaMemcpy(d_mask, h_mask, nx * ny * sizeof(char), cudaMemcpyHostToDevice);
    
    // Cleanup
    delete[] h_mask;
    delete[] h_f;
    delete[] h_f_new;
}

// Runs one simulation step (collision, streaming, bounce-back, moving lid)
void simulation_step() {
    dim3 blockDim(8,8);
    dim3 gridDim((nx+blockDim.x-1)/blockDim.x, (ny+blockDim.y-1)/blockDim.y);
    
    collision_kernel<<<gridDim, blockDim>>>(d_f, omega, nx, ny);
    cudaDeviceSynchronize();
    
    streaming_kernel<<<gridDim, blockDim>>>(d_f, d_f_new, nx, ny);
    cudaDeviceSynchronize();
    
    bounce_back_kernel<<<gridDim, blockDim>>>(d_f_new, d_mask, nx, ny);
    cudaDeviceSynchronize();
    
    dim3 blockDim1(256);
    dim3 gridDim1((nx+blockDim1.x-1)/blockDim1.x);
    moving_lid_kernel<<<gridDim1, blockDim1>>>(d_f_new, nx, ny, U);
    cudaDeviceSynchronize();
    
    // Swap pointers for next step
    DTYPE **temp = d_f;
    d_f = d_f_new;
    d_f_new = temp;
}



DTYPE* compute_macroscopic(DTYPE** f, int nx, int ny) {
    DTYPE* macro = new DTYPE[nx * ny * 3];
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            DTYPE rho = 0.0f;
            DTYPE u_x = 0.0f;
            DTYPE u_y = 0.0f;
            for (int k = 0; k < numDirs; k++) {
                DTYPE val = f[k][i + j*nx];
                rho += val;
                u_x += val * cx[k];
                u_y += val * cy[k];
            }
            if (rho > 0.0f) {
                u_x /= rho;
                u_y /= rho;
            }
            macro[i + j*nx + 0*nx*ny] = rho;
            macro[i + j*nx + 1*nx*ny] = u_x;
            macro[i + j*nx + 2*nx*ny] = u_y;
        }
    }
    return macro;
}

void write_vti(const char* filename, DTYPE* macro, int nx, int ny) {
    std::ofstream outfile(filename);
    outfile << "<?xml version=\"1.0\"?>\n";
    outfile << "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
    outfile << "  <ImageData WholeExtent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\"\n";
    outfile << "             Origin=\"0 0 0\"\n";
    outfile << "             Spacing=\"1 1 1\">\n";
    outfile << "    <Piece Extent=\"0 " << nx-1 << " 0 " << ny-1 << " 0 0\">\n";
    outfile << "      <PointData Scalars=\"density\" Vectors=\"velocity\">\n";
    
    // Write density
    outfile << "        <DataArray type=\"Float32\" Name=\"density\" format=\"ascii\">\n";
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            outfile << std::scientific << std::setprecision(6) 
                   << macro[i + j*nx + 0*nx*ny] << " ";
        }
        outfile << "\n";
    }
    outfile << "        </DataArray>\n";

    // Write velocity as a vector field
    outfile << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            outfile << std::scientific << std::setprecision(6)
                   << macro[i + j*nx + 1*nx*ny] << " "  // ux
                   << macro[i + j*nx + 2*nx*ny] << " "  // uy
                   << "0.0 ";                           // uz (2D simulation)
        }
        outfile << "\n";
    }
    outfile << "        </DataArray>\n";
    
    outfile << "      </PointData>\n";
    outfile << "    </Piece>\n";
    outfile << "  </ImageData>\n";
    outfile << "</VTKFile>\n";
    outfile.close();
}

// Function to copy a device double pointer (array of device pointers)
// into a host array of pointers containing the copied data.
DTYPE** copyDeviceDoublePointerToHost(DTYPE** d_f, int numDirs, int nx, int ny) {
    // Allocate a host array to hold the device pointers.
    DTYPE** h_d_f = new DTYPE*[numDirs];
    cudaMemcpy(h_d_f, d_f, numDirs * sizeof(DTYPE*), cudaMemcpyDeviceToHost);
    
    // Allocate a host array to hold the copied data from each device array.
    DTYPE** h_f = new DTYPE*[numDirs];
    for (int k = 0; k < numDirs; k++) {
        h_f[k] = new DTYPE[nx * ny];
        // Copy each distribution array from the device to the host.
        cudaMemcpy(h_f[k], h_d_f[k], nx * ny * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    }
    
    // Clean up the temporary host array for device pointers.
    delete[] h_d_f;
    return h_f;
}


//-----------------------------------------------------
// Main: initialize simulation, OpenGL, and run GLUT loop
//-----------------------------------------------------
int main(int argc, char** argv) {
    // Compute relaxation parameter omega
    nu = 3.0f * (U * float(nx) / Re) + 0.5f;
    omega = 1.0f / nu;
    int timestep = 10000;

    // Initialize simulation arrays on the GPU
    initialize_simulation();

    // Use chrono for high-resolution timing
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; i++) {
        simulation_step();
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate elapsed time in seconds with microsecond precision
    std::chrono::duration<double> elapsed = end - start;
    double elapsed_seconds = elapsed.count();
    printf("Time elapsed: %.6f seconds\n", elapsed_seconds);
    // Calculate MLUPS with higher precision
    double mlups = (double(nx) * double(ny) * double(timestep)) / (1e6 * elapsed_seconds);
    printf("MLUPS: %.6f\n", mlups);


    // copy the distribution function from device to host for debugging
    int i = nx / 2;
    //int j = ny / 2; 
    int j2 = ny - 2;

    DTYPE** h_f = copyDeviceDoublePointerToHost(d_f, numDirs, nx, ny);

    DTYPE* macro = compute_macroscopic(h_f, nx, ny);
    DTYPE uxw = macro[i + j2*nx + 1*nx*ny];
    DTYPE uyw = macro[i + j2*nx + 2*nx*ny];
    printf("u at top: %.6f, %.6f\n", uxw, uyw);

    write_vti("cavity_flow.vti", macro, nx, ny);
    delete[] macro;

    return 0;
}
