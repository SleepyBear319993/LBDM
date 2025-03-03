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
DTYPE Re = 100.0f;
DTYPE nu, omega;  // nu = 3*(U*nx/Re)+0.5; omega = 1/nu

// Simulation arrays (device pointers)
DTYPE *d_f = nullptr, *d_f_new = nullptr;
char *d_mask = nullptr;

size_t simSize = nx * ny * numDirs * sizeof(DTYPE);
size_t maskSize = nx * ny * sizeof(char);


//-----------------------------------------------------
// Helper device function for indexing 3D arrays
//-----------------------------------------------------
__device__ inline int idx(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

inline int idx_h(int i, int j, int k, int nx, int ny) {
    return i + j * nx + k * nx * ny;
}

//-----------------------------------------------------
// CUDA kernels for the LBM solver
//-----------------------------------------------------

// Collision kernel: computes macroscopic variables and relaxes toward equilibrium.
__global__ void collision_kernel(DTYPE* f, DTYPE omega, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        DTYPE rho = 0.0f;
        DTYPE u_x = 0.0f;
        DTYPE u_y = 0.0f;
        for (int k = 0; k < numDirs; k++) {
            DTYPE val = f[idx(i,j,k, nx, ny)];
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
            DTYPE cu = 3.0f * (cx_const[k] * u_x + cy_const[k] * u_y);
            DTYPE feq = w_const[k] * rho * (1.0f + cu + 0.5f * cu * cu - 1.5f * usqr);
            f[idx(i,j,k, nx, ny)] = (1.0f - omega) * f[idx(i,j,k, nx, ny)] + omega * feq;
        }
    }
}

// Streaming kernel: "pull" from neighbor nodes.
__global__ void streaming_kernel(DTYPE* f_in, DTYPE* f_out, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {
            for (int k = 0; k < numDirs; k++) {
                int ip = i - cx_const[k];
                int jp = j - cy_const[k];
                f_out[idx(i,j,k, nx, ny)] = f_in[idx(ip, jp, k, nx, ny)];
            }
        } else {
            for (int k = 0; k < numDirs; k++) {
                f_out[idx(i,j,k, nx, ny)] = f_in[idx(i,j,k, nx, ny)];
            }
        }
    }
}

// Bounce-back kernel: apply bounce-back at masked (solid) nodes.
__global__ void bounce_back_kernel(DTYPE* f, char* mask, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < nx && j < ny) {
        if (mask[i + j*nx] == 1) {
            // Swap east (1) and west (3)
            int idx1 = idx(i,j,1, nx, ny);
            int idx3 = idx(i,j,3, nx, ny);
            DTYPE tmp = f[idx1];
            f[idx1] = f[idx3];
            f[idx3] = tmp;
            // Swap north (2) and south (4)
            int idx2 = idx(i,j,2, nx, ny);
            int idx4 = idx(i,j,4, nx, ny);
            tmp = f[idx2];
            f[idx2] = f[idx4];
            f[idx4] = tmp;
            // Swap north-east (5) and south-west (7)
            int idx5 = idx(i,j,5, nx, ny);
            int idx7 = idx(i,j,7, nx, ny);
            tmp = f[idx5];
            f[idx5] = f[idx7];
            f[idx7] = tmp;
            // Swap north-west (6) and south-east (8)
            int idx6 = idx(i,j,6, nx, ny);
            int idx8 = idx(i,j,8, nx, ny);
            tmp = f[idx6];
            f[idx6] = f[idx8];
            f[idx8] = tmp;
        }
    }
}

// Moving-lid kernel: impose the moving lid (velocity U) at the top.
__global__ void moving_lid_kernel(DTYPE* f, int nx, int ny, DTYPE U) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nx) {
        int j = ny - 2;  // as in your original code (note: adjust if desired)
        DTYPE rho = f[idx(i,j,0, nx, ny)] + f[idx(i,j,1, nx, ny)] + f[idx(i,j,3, nx, ny)]
                  + 2.0f * ( f[idx(i,j,2, nx, ny)] + f[idx(i,j,5, nx, ny)] + f[idx(i,j,6, nx, ny)] );
        f[idx(i,j,4, nx, ny)] = f[idx(i,j,2, nx, ny)];
        f[idx(i,j,7, nx, ny)] = f[idx(i,j,5, nx, ny)] + 0.5f*(f[idx(i,j,1, nx, ny)] - f[idx(i,j,3, nx, ny)]) - 0.5f*rho*U;
        f[idx(i,j,8, nx, ny)] = f[idx(i,j,6, nx, ny)] - 0.5f*(f[idx(i,j,1, nx, ny)] - f[idx(i,j,3, nx, ny)]) + 0.5f*rho*U;
    }
}

//-----------------------------------------------------
// Host routines for initialization and simulation
//-----------------------------------------------------
void initialize_simulation() {
    // Allocate device memory
    cudaMalloc(&d_f, simSize);
    cudaMalloc(&d_f_new, simSize);
    cudaMalloc(&d_mask, maskSize);

    // Initialize f on host (uniform density = 1, velocity = 0) and copy to device
    DTYPE* h_f = new DTYPE[nx * ny * numDirs];
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            DTYPE usq = 0.0f;  // initial u0x = 0, u0y = 0
            for (int k = 0; k < numDirs; k++) {
                DTYPE cu = 3.0f*(cx[k]*0.0f + cy[k]*0.0f);
                h_f[i + j*nx + k*nx*ny] = w[k] * 1.0f * (1.0f + cu + 0.5f*cu*cu - 1.5f*usq);
            }
        }
    }
    cudaMemcpy(d_f, h_f, simSize, cudaMemcpyHostToDevice);
    delete[] h_f;

    // Initialize mask on host (using same criteria as your Python code)
    char* h_mask = new char[nx * ny];
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            if (i == 1 || i == nx - 2 || j == 1)
                h_mask[i + j*nx] = 1;
            else
                h_mask[i + j*nx] = 0;
        }
    }
    cudaMemcpy(d_mask, h_mask, maskSize, cudaMemcpyHostToDevice);
    delete[] h_mask;
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
    DTYPE* temp = d_f;
    d_f = d_f_new;
    d_f_new = temp;
}



DTYPE* compute_macroscopic(DTYPE* f, int nx, int ny) {
    DTYPE* macro = new DTYPE[nx * ny * 3];
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            DTYPE rho = 0.0f;
            DTYPE u_x = 0.0f;
            DTYPE u_y = 0.0f;
            for (int k = 0; k < numDirs; k++) {
                DTYPE val = f[idx_h(i,j,k, nx, ny)];
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
    int j = ny / 2; int j2 = ny - 2;
    DTYPE* h_f = new DTYPE[nx * ny * numDirs];
    cudaMemcpy(h_f, d_f, nx * ny * numDirs * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    DTYPE hf0 = h_f[i + j*nx + 0*nx*ny];
    printf("f0 at center: %.6f\n", hf0);
    DTYPE hfw = h_f[i + j2*nx + 0*nx*ny];
    printf("f0 at top: %.6f\n", hfw);

    DTYPE* macro = compute_macroscopic(h_f, nx, ny);
    DTYPE uxw = macro[i + j2*nx + 1*nx*ny];
    DTYPE uyw = macro[i + j2*nx + 2*nx*ny];
    printf("u at top: %.6f, %.6f\n", uxw, uyw);

    write_vti("cavity_flow.vti", macro, nx, ny);
    delete[] macro;

    return 0;
}
