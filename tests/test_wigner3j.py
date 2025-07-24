import unittest
import numpy as np

from plytrons.wigner3j import Wigner3j, Wigner3jCalculator, clebsch_gordan, gaunt_coeff

class TestWigner3jFunctions(unittest.TestCase):

    def test_known_wigner3j_values(self):
        # Value from documentation
        val = Wigner3j(2, 6, 4, 0, 0, 0)
        self.assertAlmostEqual(val, 0.186989398002, places=9)

        # Sum of m's not zero -> should be zero
        self.assertEqual(Wigner3j(2, 6, 4, 0, 0, 1), 0.0)

        # Triangle condition violated
        self.assertEqual(Wigner3j(2, 1, 4, 0, 0, 0), 0.0)

        # |m_i| > j_i
        self.assertEqual(Wigner3j(2, 6, 4, 0, 0, 5), 0.0)

    def test_cyclic_symmetry(self):
        # Wigner 3j symbol is invariant under cyclic permutation of columns, up to sign
        a = Wigner3j(3, 2, 2, 1, -1, 0)
        b = Wigner3j(2, 2, 3, -1, 0, 1)
        c = Wigner3j(2, 3, 2, 0, 1, -1)
        # All should have the same value (or differ by a sign, depending on parity)
        self.assertAlmostEqual(abs(a), abs(b), places=12)
        self.assertAlmostEqual(abs(a), abs(c), places=12)

    def test_clebsch_gordan(self):
        # Compare to known values, e.g. for j1 = 1, m1 = 1, j2 = 1, m2 = -1, j3 = 1, m3 = 0
        # <1,1,1,-1|1,0> should be +1/√2
        cg = clebsch_gordan(1, 1, 1, -1, 1, 0)
        self.assertAlmostEqual(cg, 1/np.sqrt(2), places=12)

    def test_gaunt_coeff(self):
        # The Gaunt coefficient for l1=l2=l3=0, m1=m2=m3=0 is 1/sqrt(4*pi)
        gaunt = gaunt_coeff(0, 0, 0, 0, 0, 0)
        self.assertAlmostEqual(gaunt, 1/np.sqrt(4*np.pi), places=12)

    def test_calculator_bulk(self):
        # Compare array output to single value
        calc = Wigner3jCalculator(3, 3)
        arr = calc.calculate(2, 2, 0, 0)
        # Should match single value
        self.assertAlmostEqual(arr[2], Wigner3j(2, 2, 2, 0, 0, 0), places=12)
        self.assertAlmostEqual(arr[0], Wigner3j(0, 2, 2, 0, 0, 0), places=12)
        # Asserting out of bounds
        self.assertEqual(arr[5], 0.0)  # outside j_max
        self.assertEqual(arr[6], 0.0)  # outside j_max

if __name__ == '__main__':
    unittest.main()
