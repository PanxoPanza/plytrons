import unittest
import numpy as np
from plytrons import math_utils
from scipy.special import spherical_jn, sph_harm, lpmv, poch

class TestMathUtils(unittest.TestCase):

    def test_nb_meshgrid(self):
        x = np.linspace(0, 1, 3)
        y = np.linspace(0, 2, 4)
        xx, yy = math_utils.nb_meshgrid(x, y)
        # Compare with numpy.meshgrid (with indexing='xy')
        xx_np, yy_np = np.meshgrid(x, y)
        np.testing.assert_allclose(xx, xx_np)
        np.testing.assert_allclose(yy, yy_np)

    def test_js_real_vs_scipy(self):
        # Test a few l, x values vs scipy.special.spherical_jn
        for l in range(0, 4):
            xvals = np.linspace(0.1, 2, 5)
            res = math_utils.js_real(l, xvals)
            ref = spherical_jn(l, xvals)
            np.testing.assert_allclose(res, ref, rtol=1e-7)

    def test_hs_imag_types(self):
        # Just test it runs for a range (complex result, can't test directly with scipy)
        out = math_utils.hs_imag(2, 0.5)
        self.assertTrue(np.iscomplex(out))

    def test_legendre_poly_vs_scipy(self):
        poly = math_utils.Legendre_poly()
        z = 0.5
        lmax = 3
        # Test P_l^0(z) vs scipy
        for l in range(lmax + 1):
            for m in range(0, l + 1):
                our = poly.Plm(l, m, z)
                scale = (-1)**m*np.sqrt((2*l + 1)*poch(l + m + 1, -2*m))
                ref = scale*lpmv(m, l, z)
                np.testing.assert_allclose(our, ref, rtol=1e-7)

    def test_nb_lpmv_vs_scipy(self):
        # Associated Legendre, compare to scipy.special.lpmv
        z = 0.3
        for l in range(0, 4):
            for m in range(0, l + 1):
                our = math_utils.nb_lpmv(l, m, z)
                scale = np.sqrt((2*l + 1)*poch(l + m + 1, -2*m))
                ref = scale*lpmv(m, l, z)
                np.testing.assert_allclose(our, ref, rtol=1e-7)

    def test_qm_sph_harm_vs_scipy(self):
        # Compare quantum spherical harmonics with scipy.special.sph_harm (for m >= 0)
        l, m = 2, 1
        theta = np.pi / 3
        phi = np.pi / 4
        our = math_utils.qm_sph_harm(m, l, theta, phi)
        ref = sph_harm(m, l, phi, theta)  # scipy uses (m, l, phi, theta)
        np.testing.assert_allclose(our, ref, rtol=1e-7)

    def test_qm_sph_harm_negative_m(self):
        # Test symmetry for negative m
        l, m = 2, -1
        theta = np.pi / 3
        phi = np.pi / 4
        our = math_utils.qm_sph_harm(m, l, theta, phi)
        ref = sph_harm(m, l, phi, theta)
        np.testing.assert_allclose(our, ref, rtol=1e-7)

    def test_nb_meshgrid_square(self):
        # Edge case: square grid
        x = np.array([0, 1])
        y = np.array([0, 1])
        xx, yy = math_utils.nb_meshgrid(x, y)
        xx_ref, yy_ref = np.meshgrid(x, y)
        np.testing.assert_allclose(xx, xx_ref)
        np.testing.assert_allclose(yy, yy_ref)

if __name__ == "__main__":
    unittest.main()