import unittest
import numpy as np
from lattice_cons import D2Q9

class TestExample(unittest.TestCase):
    def test_cx_cy_dot_product(self):
        d2q9 = D2Q9()
        dot_product = np.dot(d2q9.cx, d2q9.cy)
        self.assertEqual(dot_product, 0)

if __name__ == '__main__':
    unittest.main()