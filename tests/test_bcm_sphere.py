import unittest
import numpy as np

# Library under test
import plytrons.bcm_sphere as bcm


def _eps_const(_lambda_nm):
    """Dummy isotropic permittivity – constant over wavelength."""
    return 2.0 + 0.0j


class BCMTestCase(unittest.TestCase):
    """Minimal smoke‑tests for the bcm_sphere API using unittest only."""

    def setUp(self):
        # --- electromagnetic set‑up ----------------------------------------
        self.lambda_nm = 600.0                         # nm
        c0            = 299_792_458.0                  # m/s
        self.w        = 2 * np.pi * c0 / (self.lambda_nm * 1e-9)  # rad/s
        self.eps_h    = 1.0                            # vacuum / air

        # --- BCMObject ------------------------------------------------------
        self.lmax   = 3
        self.sphere = bcm.BCMObject(
            label="dummy",
            diameter=20.0,              # nm
            lmax=self.lmax,
            eps=_eps_const,
            position=np.zeros(3),
        )

        # --- Plane‑wave source ---------------------------------------------
        self.Efield = bcm.EField(
            E0=1.0,
            k_hat=np.array([0.0, 0.0, 1.0]),
            e_hat=np.array([1.0, 0.0, 0.0]),
        )

        # --- Pre‑compute one‑sphere matrices & source -----------------------
        self.Gint = [bcm.Ginternal(self.sphere)]
        self.Gext = [[bcm.Gexternal(self.sphere, self.sphere)]]
        self.Sext = [bcm.Efield_coupling(self.sphere, self.Efield)]

    # ---------------------------------------------------------------------
    # Individual unit tests ------------------------------------------------
    # ---------------------------------------------------------------------

    def test_cached_properties(self):
        # n_coef formula
        self.assertEqual(self.sphere.n_coef, self.lmax * (self.lmax + 2))
        # index_range correctness + caching (same id twice)
        idx_first  = self.sphere.index_range
        idx_second = self.sphere.index_range
        self.assertTrue(np.array_equal(idx_first, np.arange(self.sphere.n_coef)))
        self.assertIs(idx_first, idx_second)

    def test_Ginternal_shape(self):
        Gin = self.Gint[0]
        self.assertEqual(Gin.shape, (self.sphere.n_coef, self.sphere.n_coef))

    def test_Efield_coupling_shape(self):
        Si = self.Sext[0]
        self.assertEqual(Si.shape, (self.sphere.n_coef,))

    def test_solve_BCM_singleSphere(self):
        X, Sw = bcm.solve_BCM(
            self.w,
            self.eps_h,
            [self.sphere],
            self.Efield,
            self.Gint,
            self.Gext,
            self.Sext,
        )

        # Shape sanity checks ---------------------------------------------
        self.assertEqual(len(X), 1)
        self.assertEqual(X[0].shape, (self.sphere.n_coef,))
        self.assertEqual(Sw[0].shape, (self.sphere.n_coef,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
