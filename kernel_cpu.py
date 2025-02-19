import math
import numpy as np

class D2Q9:
    w = np.array([4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0])
    cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    cs2 = 1.0/3.0
    cs = math.sqrt(cs2)

class LatticeInitializerD2Q9:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.f = np.zeros((nx, ny, 9)).astype(float)  
        self.u = np.zeros((nx, ny, 2)).astype(float)  
        self.rho = np.ones((nx, ny)).astype(float)  

    def initialize(self, rho, u):
        for i in range(self.nx):
            for j in range(self.ny):
                self.u[i, j] = u
                for k in range(9):
                    cu = 3 * (D2Q9.cx[k] * u[0] + D2Q9.cy[k] * u[1])
                    self.f[i, j, k] = D2Q9.w[k] * rho * (1 + cu + 0.5 * cu**2 - 1.5 * (u[0]**2 + u[1]**2))

    def streaming(self):
        f_temp = np.copy(self.f)
        for k in range(9):
            self.f[:, :, k] = np.roll(np.roll(f_temp[:, :, k], D2Q9.cx[k], axis=0), D2Q9.cy[k], axis=1)

    def collision(self, omega):
        for i in range(self.nx):
            for j in range(self.ny):
                rho = np.sum(self.f[i, j, :])
                ux = np.sum(self.f[i, j, :] * D2Q9.cx) / rho
                uy = np.sum(self.f[i, j, :] * D2Q9.cy) / rho
                u_sq = ux**2 + uy**2
                for k in range(9):
                    cu = 3 * (D2Q9.cx[k] * ux + D2Q9.cy[k] * uy)
                    feq = D2Q9.w[k] * rho * (1 + cu + 0.5 * cu**2 - 1.5 * u_sq)
                    self.f[i, j, k] = (1 - omega) * self.f[i, j, k] + omega * feq
                    
    def update_macroscopic(self):
        for i in range(self.nx):
            for j in range(self.ny):
                rho_ij = np.sum(self.f[i, j, :])
                self.rho[i, j] = rho_ij
                if rho_ij > 1e-12:
                    ux_ij = np.sum(self.f[i, j, :] * D2Q9.cx) / rho_ij
                    uy_ij = np.sum(self.f[i, j, :] * D2Q9.cy) / rho_ij
                else:
                    ux_ij, uy_ij = 0.0, 0.0
                self.u[i, j, 0] = ux_ij
                self.u[i, j, 1] = uy_ij

    def bounce_back(self):
        # Assuming bounce-back on all boundaries
        for i in range(self.nx):
            for j in range(self.ny):
                if i == 0 or i == self.nx - 1 or j == 0 or j == self.ny - 1:
                    self.f[i, j, [1, 5, 8]] = self.f[i, j, [3, 7, 6]]
                    self.f[i, j, [3, 6, 7]] = self.f[i, j, [1, 8, 5]]
                    self.f[i, j, [2, 5, 6]] = self.f[i, j, [4, 7, 8]]
                    self.f[i, j, [4, 7, 8]] = self.f[i, j, [2, 5, 6]]