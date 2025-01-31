import unittest
import cupy as cp
from lattice_cons_gpu import D2Q9

class TestExample(unittest.TestCase):
    def test_cx_cy_dot_product(self):
        d2q9 = D2Q9()
        dot_product = cp.dot(d2q9.cx, d2q9.cy)
        self.assertEqual(dot_product.get(), 0)  # Use .get() to transfer data from GPU to CPU

if __name__ == '__main__':
    unittest.main()