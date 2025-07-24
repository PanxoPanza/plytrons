import unittest
import numpy as np
from plytrons import quantum_well as qw

# -----------------------------------------------------------------------------
# Analytic helpers for an *infinite* spherical well (reference values)
# -----------------------------------------------------------------------------

def infinite_well_E(a_nm: float, l: int, n: int) -> float:
    """Return energy (eV) for the *n*‑th root of j_l inside an infinite well."""
    # j_l zeros: for l=0 => π, 2π, 3π; for l>0 we fetch from scipy if avail.
    # To stay self‑contained, only implement l=0,1 with hard‑coded first zeros.
    j0_roots = np.array([np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    j1_roots = np.array([4.49340946, 7.72525184])  # first two roots of j1
    x_ln = j0_roots[n] if l == 0 else j1_roots[n]

    hbar = qw.hbar  # eV fs
    me   = qw.me    # eV fs² nm⁻²
    return (hbar**2 * x_ln**2) / (2*me*a_nm**2)


class QuantumWellEnergyTests(unittest.TestCase):
    """Compare first j_l zeros against large‑depth quantum_well results."""

    @classmethod
    def setUpClass(cls):
        cls.a_nm   = 1.0
        cls.lmax   = 2
        cls.V0     = 1e3  # effectively infinite
        cls.bound  = qw.get_bound_states(cls.a_nm, cls.lmax, cls.V0,
                                         nE_coarse=5000)  # speed‑up

    # ------------------------------------------------------------------
    # ℓ = 0: first four energies
    # ------------------------------------------------------------------
    def test_l0_energies(self):
        E_calc = self.bound[0]                 # numpy array of roots for l=0
        self.assertGreaterEqual(E_calc.size, 4, "Less than 4 bound states")

        for n in range(4):
            E_ref = infinite_well_E(self.a_nm, l=0, n=n)
            self.assertAlmostEqual(E_calc[n], E_ref, delta=0.03*E_ref)

    # ------------------------------------------------------------------
    # ℓ = 1: first two energies
    # ------------------------------------------------------------------
    def test_l1_energies(self):
        E_calc = self.bound[1]
        self.assertGreaterEqual(E_calc.size, 2, "Less than 2 bound states")

        for n in range(2):
            E_ref = infinite_well_E(self.a_nm, l=1, n=n)
            self.assertAlmostEqual(E_calc[n], E_ref, delta=0.05*E_ref)

    # ------------------------------------------------------------------
    # Characteristic equation must vanish at root
    # ------------------------------------------------------------------
    def test_F_zero(self):
        l0_first = self.bound[0][0]
        val = qw.F(l0_first, 0, self.a_nm)
        self.assertAlmostEqual(val, 0.0, delta=1e-3)

# -----------------------------------------------------------------------------
# Extra tests for the new dataclass + e_state_assembly helper
# -----------------------------------------------------------------------------

class AssemblyHelperTests(unittest.TestCase):
    """Unit‑tests that validate the behaviour of QWLevelSet and e_state_assembly."""

    def test_dataclass_basic(self):
        """QWLevelSet should store energies & norms and expose them as attributes."""
        from plytrons.quantum_well import QWLevelSet  # local import to avoid circulars
        Eb = np.array([1.0, 2.0], dtype=np.float64)
        A  = np.array([0.1, 0.2], dtype=np.complex128)
        level = QWLevelSet(Eb=Eb, A=A)
        # attribute access
        self.assertTrue(np.all(level.Eb == Eb))
        self.assertTrue(np.all(level.A  == A))
        # dataclass generated __repr__ should contain the class name
        self.assertIn('QWLevelSet', repr(level))

    def test_e_state_assembly_padded(self):
        """e_state_assembly should strip NaN/0 padding from matrix input."""
        from plytrons.quantum_well import e_state_assembly
        E_mat = np.array([[1.0, 2.0, np.nan], [3.0, 0.0, 0.0]])
        A_mat = np.array([[0.1, 0.2, np.nan], [0.3, 0.0, 0.0]])
        assembled = e_state_assembly(E_mat, A_mat)
        # After stripping, lengths should be 2 and 1 respectively
        self.assertEqual(assembled[0].Eb.shape[0], 2)
        self.assertEqual(assembled[1].Eb.shape[0], 1)
        # NaNs and zeros should be gone
        self.assertTrue(np.isfinite(assembled[0].Eb).all())
        self.assertTrue((assembled[1].Eb != 0.0).all())

if __name__ == "__main__":
    unittest.main(verbosity=2)
