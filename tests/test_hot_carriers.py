# tests/test_hot_carriers.py
"""
Unit-tests for plytrons.hot_carriers written with the std-lib 'unittest'.

The numerical choices are intentionally tiny so first-run Numba
compilation finishes quickly on most laptops.
"""
# from __future__ import annotations

import unittest
import numpy as np
from plytrons.hot_carriers import _fermi_dirac, idx_to_lm, hot_e_dist
from plytrons.quantum_well import QWLevelSet

# ----------------------------------------------------------------------
# Helper: toy level-set with just two bound states
# ----------------------------------------------------------------------
def _mini_levels() -> list[QWLevelSet]:
    """
    Returns two minimal `QWLevelSet` objects:

    * l = 0  → E₀ = 0.10 eV, amplitude 1
    * l = 1  → E₁ = 0.20 eV, amplitude 1
    """
    return [
        QWLevelSet(Eb=np.array([0.10]), A=np.array([1.0 + 0j])),
        QWLevelSet(Eb=np.array([0.20]), A=np.array([1.0 + 0j])),
    ]


# ----------------------------------------------------------------------
# Main test-case class
# ----------------------------------------------------------------------
class TestHotCarriers(unittest.TestCase):
    # --- idx_to_lm -----------------------------------------------------

    def test_idx_to_lm_lookup(self):
        lookup = {
            0: (1, -1),
            1: (1, 0),
            2: (1, 1),
            3: (2, -2),
            4: (2, -1),
            5: (2, 0),
        }
        for k, expected in lookup.items():
            with self.subTest(k=k):
                self.assertEqual(idx_to_lm(k), expected)

    def test_idx_to_lm_roundtrip(self):
        """k ↔ (l,m) stays bijective for the first 50 indices."""
        for k in range(50):
            l, m = idx_to_lm(k)
            k_back = (l - 1) * (l + 1) + (m + l)
            self.assertEqual(k_back, k)

    def test_idx_to_lm_negative(self):
        with self.assertRaises(IndexError):
            idx_to_lm(-1)

    # --- _fermi_dirac --------------------------------------------------

    def test_fermi_dirac_half_filling(self):
        E_F = 0.0  # eV
        f = _fermi_dirac(np.array([E_F]), E_F)
        self.assertAlmostEqual(f[0], 0.5, places=12)

    def test_fermi_dirac_monotonic(self):
        """At 300 K the distribution must decrease monotonically with E."""
        E = np.linspace(-1, 1, 11)
        f = _fermi_dirac(E, 0.0)
        self.assertTrue(np.all(np.diff(f) < 0))

    # --- end-to-end smoke test ----------------------------------------

    def test_hot_e_dist_smoke(self):
        """
        Quick pass through the full Numba pipeline:

        * shape of the returned arrays is correct
        * all outputs are finite
        * no negative generation / heating rates
        """
        Te, Th = hot_e_dist(
            a_nm     = 5.0,               # sphere radius (nm)
            hv_eV    = 2.0,               # photon energy (eV)
            E_F      = 0.0,               # Fermi level
            tau_e_fs = 10.0,              # electron lifetime (fs)
            e_state  = _mini_levels(),
            X_lm     = np.ones(3, dtype=np.complex128)  # |X₁m| = 1
        )

        self.assertEqual(Te.shape, (2,))
        self.assertEqual(Th.shape, (2,))
        self.assertTrue(np.all(np.isfinite(Te)))
        self.assertTrue(np.all(np.isfinite(Th)))
        self.assertTrue(np.all(Te >= 0))
        self.assertTrue(np.all(Th >= 0))


# ----------------------------------------------------------------------
# Allow `python tests/test_hot_carriers.py` execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    unittest.main(verbosity=2)
