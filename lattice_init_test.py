import unittest
import numpy as np
from kernel_cpu import D2Q9, LatticeInitializerD2Q9

class TestLatticeInitializerD2Q9(unittest.TestCase):
    def test_initialize(self):
        nx, ny = 10, 10
        rho = 1.0
        u = np.array([0.1, 0.2])
        initializer = LatticeInitializerD2Q9(nx, ny)
        initializer.initialize(rho, u)
        
        # Check if the velocity field is initialized correctly
        for i in range(nx):
            for j in range(ny):
                np.testing.assert_array_almost_equal(initializer.u[i, j], u)
        
        # Check if the distribution function is initialized correctly
        for i in range(nx):
            for j in range(ny):
                for k in range(9):
                    cu = 3 * (D2Q9.cx[k] * u[0] + D2Q9.cy[k] * u[1])
                    expected_f = D2Q9.w[k] * rho * (1 + cu + 0.5 * cu**2 - 1.5 * (u[0]**2 + u[1]**2))
                    self.assertAlmostEqual(initializer.f[i, j, k], expected_f)

    def test_streaming(self):
        nx, ny = 2, 2
        initializer = LatticeInitializerD2Q9(nx, ny)
        
        # Manually specify the distribution function
        initializer.f = np.array([
            [[0, 1, 2, 3, 4, 5, 6, 7, 8],
             [9, 10, 11, 12, 13, 14, 15, 16, 17]],
            [[18, 19, 20, 21, 22, 23, 24, 25, 26],
             [27, 28, 29, 30, 31, 32, 33, 34, 35]]
             ])
        
        # Expected distribution function after streaming
        f_expected = np.array([
            [[0, 19, 11, 21, 13, 32, 33, 34, 35],
             [9, 28, 2, 30, 4, 23, 24, 25, 26]],
            [[18, 1, 29, 3, 31, 14, 15, 16, 17],
             [27, 10, 20, 12, 22, 5, 6, 7, 8]]
        ])
        
        #        Before
        # 6  2  5     24 20 23
        # 3  0  1     21 18 19
        # 7  4  8     25 22 26
        
        # 15 11 14    33 29 32
        # 12  9 10    30 27 28
        # 16 13 17    34 31 35
   #-------------------------------  
        #        After   
        # 33 11 32    15 29 14
        # 21 0  19     3 18  1
        # 34 13 35    16 31 17
        
        # 24  2 23    6  20  5
        # 30  9 28    12 27 10
        # 25  4 26    7  22  8
        
        # Print expected distribution function
        print("Expected distribution function after streaming:", f_expected)
        
        # Print the initial state
        print("Initial state:", initializer.f)
        
        # Perform streaming
        initializer.streaming()
        
        # Print the final state
        print("Final state:", initializer.f)
        
        
        # Check if the streaming step is performed correctly
        for i in range(nx):
            for j in range(ny):
                for k in range(9):
                    self.assertAlmostEqual(initializer.f[i, j, k], f_expected[i, j, k])

if __name__ == '__main__':
    unittest.main()