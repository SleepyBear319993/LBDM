import unittest
import numpy as np
from kernel_gpu import LBMSolverD2Q9GPU

class TestLBMSolverD2Q9GPU(unittest.TestCase):
    def test_streaming_periodic(self):
                # Manually specify the distribution function
        f_in = np.array([
            [[0, 1, 2, 3, 4, 5, 6, 7, 8],
             [9, 10, 11, 12, 13, 14, 15, 16, 17]],
            [[18, 19, 20, 21, 22, 23, 24, 25, 26],
             [27, 28, 29, 30, 31, 32, 33, 34, 35]]
             ], dtype=np.float32)
        
        # Expected distribution function after streaming
        f_expected = np.array([
            [[0, 19, 11, 21, 13, 32, 33, 34, 35],
             [9, 28, 2, 30, 4, 23, 24, 25, 26]],
            [[18, 1, 29, 3, 31, 14, 15, 16, 17],
             [27, 10, 20, 12, 22, 5, 6, 7, 8]]
        ], dtype=np.float32)
        
        f_out = np.zeros_like(f_in)
        nx = 2
        ny = 2
        
        lb = LBMSolverD2Q9GPU(nx, ny, 1.0, 0.0)
        lb.f.copy_to_device(f_in)
        lb.stream_periodic()
        f_out = lb.get_distribution()
        
        # Print expected distribution function
        print("Expected distribution function after streaming:", f_expected)
        
        # Print the initial state
        print("Distribution function after streaming:", f_out)
        
        for i in range(nx):
            for j in range(ny):
                for k in range(9):
                    self.assertAlmostEqual(f_out[i, j, k], f_expected[i, j, k])
        
        
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

if __name__ == '__main__':
    unittest.main()