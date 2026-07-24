import unittest
import tempfile
import yaml

import numpy as np

from xpeak.core import Atom, UnitCell, atomic_scattering_factor, energy_kev_to_wavelength, enumerate_peaks, simulate_detector, wavelength_to_energy_kev, you_detector_frame, you_sample_rotation
from xpeak.diffcalc_backend import MotorSolution, apply_motor_limits, calculate_ub_from_two_bisecting_reflections, hkl_to_motor_solutions, kappa_sample_rotation, motor_in_limits, motors_to_hkl, parse_constraints, parse_motor_limits, peak_surface_geometries, scan_alpha_beta, solve_symmetric_geometry, you_to_kappa
from xpeak.phonons import PhononDataset, PhononMode, assign_irreps, displaced_atoms


class CoreTests(unittest.TestCase):
    def test_cubic_d_spacing(self):
        cell = UnitCell(4, 4, 4)
        peaks = enumerate_peaks(cell, [Atom("C", 0, 0, 0)], 1.0, 2, 1, 90, 0)
        p100 = next(p for p in peaks if (p.h, p.k, p.l) == (1, 0, 0))
        self.assertAlmostEqual(p100.d, 4.0)

    def test_bcc_extinction(self):
        atoms = [Atom("Fe", 0, 0, 0), Atom("Fe", .5, .5, .5)]
        peaks = enumerate_peaks(UnitCell(2.87, 2.87, 2.87), atoms, 1.0, 2, 1, 120, 0.001)
        hkls = {(p.h, p.k, p.l) for p in peaks}
        self.assertNotIn((1, 0, 0), hkls)
        self.assertIn((1, 1, 0), hkls)

    def test_detector_runs(self):
        spots = simulate_detector(UnitCell(5, 5, 5), [Atom("Si", 0, 0, 0)], 1.0, (0, 0, 0, 0), (0, 0), 100, .2, (512, 512), (256, 256), 4, 1.0)
        self.assertIsInstance(spots, list)

    def test_energy_dependent_form_factor(self):
        factor = atomic_scattering_factor("Si", 1.0, 11000.0)
        self.assertIsInstance(factor, complex)
        self.assertGreater(abs(factor), 1)

    def test_you_zero_motor_frames(self):
        np.testing.assert_allclose(you_sample_rotation(0, 0, 0, 0), np.eye(3), atol=1e-12)
        normal, u_axis, v_axis = you_detector_frame(0, 0)
        np.testing.assert_allclose(normal, [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(u_axis, [1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(v_axis, [0, 0, 1], atol=1e-12)

    def test_you_detector_equation_9(self):
        delta, nu = 23.0, -17.0
        normal, _, _ = you_detector_frame(delta, nu)
        d, n = np.radians([delta, nu])
        np.testing.assert_allclose(normal, [np.sin(d), np.cos(n) * np.cos(d), np.sin(n) * np.cos(d)], atol=1e-12)

    def test_diffcalc_round_trip(self):
        cell = UnitCell(4, 4, 4)
        solutions = hkl_to_motor_solutions(cell, (0, 0, 1), 1.0, parse_constraints("nu=0, mu=0, a_eq_b"))
        self.assertTrue(solutions)
        s = solutions[0]
        recovered = motors_to_hkl(cell, (s.mu, s.delta, s.nu, s.eta, s.chi, s.phi), 1.0)
        np.testing.assert_allclose(recovered, (0, 0, 1), atol=1e-6)

    def test_constraint_parser(self):
        self.assertEqual(parse_constraints("nu=0, mu=0, a_eq_b"), {"nu": 0.0, "mu": 0.0, "a_eq_b": True})

    def test_default_motor_limit_parser_and_filter(self):
        limits = parse_motor_limits("delta=0:180, chi=0:180")
        self.assertEqual(limits, {"delta": (0.0, 180.0), "chi": (0.0, 180.0)})
        solutions = hkl_to_motor_solutions(UnitCell(4, 4, 4), (0, 0, 1), 1.0, {"nu": 0, "mu": 0, "a_eq_b": True}, motor_limits=limits)
        self.assertTrue(all(motor_in_limits(solution, limits) for solution in solutions))

    def test_motor_angles_are_cut_into_zero_to_360_range(self):
        solution = MotorSolution(-10, 20, -30, -40, 50, -60, {})
        limits = parse_motor_limits("mu=0:360, nu=0:360, eta=0:360, phi=0:360")
        adjusted = apply_motor_limits(solution, limits)
        self.assertEqual((adjusted.mu, adjusted.nu, adjusted.eta, adjusted.phi), (350, 330, 320, 300))

    def test_ub_from_two_bisecting_reflections(self):
        references = [((0, 0, 1), 30, 90, 0), ((0, 1, 1), 45, 45, 90)]
        u_matrix, ub_matrix = calculate_ub_from_two_bisecting_reflections(UnitCell(1, 1, 1), 1.0, references)
        np.testing.assert_allclose(u_matrix, np.eye(3), atol=1e-7)
        np.testing.assert_allclose(ub_matrix, 2 * np.pi * np.eye(3), atol=1e-7)

    def test_kappa_conversion_reconstructs_you_orientation(self):
        eta, chi, phi = 20.0, 35.0, -14.0
        kp = you_to_kappa(eta, chi, phi, 50.0)
        expected = you_sample_rotation(0, eta, chi, phi)
        actual = kappa_sample_rotation(kp.komega, kp.kappa, kp.kphi, 50.0)
        np.testing.assert_allclose(actual, expected, atol=1e-8)

    def test_kappa_alpha_90_matches_eulerian(self):
        kp = you_to_kappa(12.0, 30.0, -8.0, 90.0)
        actual = kappa_sample_rotation(kp.komega, kp.kappa, kp.kphi, 90.0)
        np.testing.assert_allclose(actual, you_sample_rotation(0, 12, 30, -8), atol=1e-8)

    def test_surface_incidence_exit_geometry(self):
        result = peak_surface_geometries(UnitCell(4, 4, 4), [(1, 1, 1)], 1.0, {"nu": 0, "mu": 0, "a_eq_b": True}, (0, 0, 1), 5.0)
        self.assertIn((1, 1, 1), result)
        solution = result[(1, 1, 1)]
        self.assertAlmostEqual(solution.virtual["alpha"], solution.virtual["beta"])

    def test_alpha_beta_scan(self):
        rows = scan_alpha_beta(UnitCell(4, 4, 4), (1, 1, 1), 1.0, {"nu": 0, "mu": 0}, (0, 0, 1), [5.0, 7.0])
        self.assertTrue(rows)
        for requested_alpha, solution in rows:
            self.assertAlmostEqual(solution.virtual["alpha"], requested_alpha, places=5)

    def test_specular_alpha_scan_returns_bragg_angle(self):
        rows = scan_alpha_beta(UnitCell(4, 4, 4), (0, 0, 1), 1.0, {"nu": 0, "mu": 0}, (0, 0, 1), np.linspace(1, 15, 8))
        self.assertTrue(rows)
        for alpha, solution in rows:
            self.assertAlmostEqual(alpha, solution.virtual["beta"], places=6)
            self.assertAlmostEqual(alpha, 7.180755781458282, places=6)

    def test_specular_equal_alpha_beta_solver(self):
        solutions = solve_symmetric_geometry(UnitCell(4, 4, 4), (0, 0, 1), 1.0, {"nu": 0, "mu": 0}, (0, 0, 1))
        self.assertTrue(solutions)
        self.assertTrue(all(abs(s.virtual["alpha"] - s.virtual["beta"]) < 1e-7 for s in solutions))

    def test_gamma_mode_displacement_mass_weighting(self):
        mode = PhononMode(0, 0, (0, 0, 0), 1.0, np.array([[1 + 0j, 0j, 0j]]))
        dataset = PhononDataset(UnitCell(2, 2, 2), np.diag([2.0, 2.0, 2.0]), [Atom("C", 0, 0, 0)], np.array([4.0]), [mode])
        moved = displaced_atoms(dataset, mode, 1.0)
        self.assertAlmostEqual(moved[0].x, 0.25)

    def test_energy_wavelength_conversion(self):
        wavelength = energy_kev_to_wavelength(11.0)
        self.assertAlmostEqual(wavelength, 1.1271290766654568)
        self.assertAlmostEqual(wavelength_to_energy_kev(wavelength), 11.0)

    def test_irrep_assignment_by_band_and_frequency(self):
        mode = PhononMode(0, 0, (0, 0, 0), 2.5, np.array([[1 + 0j, 0j, 0j]]))
        dataset = PhononDataset(UnitCell(2, 2, 2), np.diag([2.0, 2.0, 2.0]), [Atom("C", 0, 0, 0)], np.array([12.0]), [mode])
        content = {"q-position": [0, 0, 0], "point_group": "m-3m", "normal_modes": [{"band_indices": [1], "frequency": 2.5, "ir_label": "T2g"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as stream:
            yaml.safe_dump(content, stream)
            stream.flush()
            assigned, matched, mismatches = assign_irreps(dataset, stream.name)
        self.assertEqual(matched, 1)
        self.assertFalse(mismatches)
        self.assertEqual(assigned.modes[0].ir_label, "T2g")


if __name__ == "__main__":
    unittest.main()
