"""Thin application adapter for Diamond Light Source's diffcalc-core."""

from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import redirect_stdout
import io
import re

import numpy as np

from .core import UnitCell


@dataclass(frozen=True)
class MotorSolution:
    mu: float
    delta: float
    nu: float
    eta: float
    chi: float
    phi: float
    virtual: dict[str, float]

    @property
    def sample_motors(self) -> tuple[float, float, float, float]:
        return self.mu, self.eta, self.chi, self.phi

    @property
    def detector_motors(self) -> tuple[float, float]:
        return self.delta, self.nu


@dataclass(frozen=True)
class KappaPosition:
    komega: float
    kappa: float
    kphi: float
    residual_deg: float


@dataclass(frozen=True)
class YouPosition:
    eta: float
    chi: float
    phi: float
    residual_deg: float


def _axis_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) * np.cos(angle_rad) + (1 - np.cos(angle_rad)) * np.outer(axis, axis) + np.sin(angle_rad) * cross


def kappa_sample_rotation(komega: float, kappa: float, kphi: float, alpha: float = 50.0) -> np.ndarray:
    """Kappa cradle rotation using the You phi-frame axes.

    komega and kphi are left-handed about +z. The right-handed kappa axis lies
    in the yz plane, tilted alpha degrees from +z toward +y.
    """
    ko, kap, kp, a = np.radians([komega, kappa, kphi, alpha])
    rz_outer = np.array([[np.cos(ko), np.sin(ko), 0], [-np.sin(ko), np.cos(ko), 0], [0, 0, 1]])
    rz_inner = np.array([[np.cos(kp), np.sin(kp), 0], [-np.sin(kp), np.cos(kp), 0], [0, 0, 1]])
    return rz_outer @ _axis_rotation(np.array([0.0, np.sin(a), np.cos(a)]), kap) @ rz_inner


def you_to_kappa(eta: float, chi: float, phi: float, alpha: float = 50.0) -> KappaPosition:
    """Find a kappa position representing the same H(eta) X(chi) Phi(phi)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .core import you_sample_rotation

    if not 0 < alpha <= 90:
        raise ValueError("Kappa-axis alpha must be in (0, 90] degrees.")
    target = you_sample_rotation(0.0, eta, chi, phi)

    def residual(values):
        candidate = kappa_sample_rotation(*values, alpha)
        return Rotation.from_matrix(target.T @ candidate).as_rotvec()

    guesses = [
        (eta % 360.0, min(abs(chi), 180.0), phi % 360.0),
        ((eta + 180.0) % 360.0, min(abs(chi), 180.0), (phi + 180.0) % 360.0),
        (0.0, 0.0, 0.0),
        (180.0, 90.0, 180.0),
    ]
    candidates = [least_squares(residual, guess, bounds=([0, 0, 0], [360, 180, 360]), xtol=1e-12, ftol=1e-12, gtol=1e-12) for guess in guesses]
    best = min(candidates, key=lambda result: np.linalg.norm(result.fun))
    error = float(np.degrees(np.linalg.norm(best.fun)))
    if error > 1e-5:
        raise ValueError(f"Orientation is unreachable with a {alpha:g}° kappa axis (best error {error:.3g}°).")
    angles = tuple(float(v) % 360.0 for v in best.x)
    return KappaPosition(*angles, error)


def kappa_to_you(komega: float, kappa: float, kphi: float, alpha: float = 50.0) -> YouPosition:
    """Find You Eulerian angles representing the same kappa-cradle rotation."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from .core import you_sample_rotation

    if not 0 < alpha <= 90:
        raise ValueError("Kappa-axis alpha must be in (0, 90] degrees.")
    target = kappa_sample_rotation(komega, kappa, kphi, alpha)

    def residual(values):
        candidate = you_sample_rotation(0.0, *values)
        return Rotation.from_matrix(target.T @ candidate).as_rotvec()

    guesses = (
        (komega, kappa, kphi),
        (komega + 180, -kappa, kphi + 180),
        (0, 0, 0),
        (90, 90, 90),
    )
    candidates = [least_squares(residual, guess, bounds=([-720, -180, -720], [720, 180, 720]), xtol=1e-12, ftol=1e-12, gtol=1e-12) for guess in guesses]
    best = min(candidates, key=lambda result: np.linalg.norm(result.fun))
    error = float(np.degrees(np.linalg.norm(best.fun)))
    if error > 1e-5:
        raise ValueError(f"Kappa orientation could not be represented in You angles (best error {error:.3g}°).")
    angles = tuple(((float(v) + 180.0) % 360.0) - 180.0 for v in best.x)
    return YouPosition(*angles, error)


def parse_constraints(text: str) -> dict[str, float | bool]:
    """Parse user-facing constraints into a Diffcalc constraint mapping.

    ``betain=betaout`` is accepted as the readable spelling of Diffcalc's
    ``bin_eq_bout`` constraint. It is intentionally distinct from ``a_eq_b``,
    which refers to Diffcalc's virtual alpha and beta angles.
    """
    result: dict[str, float | bool] = {}
    for item in re.split(r"[,;]", text):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, value = (part.strip() for part in item.split("=", 1))
            if name.lower() == "betain" and value.lower() == "betaout":
                result["bin_eq_bout"] = True
                continue
            if name.lower() == "betaout" and value.lower() == "betain":
                result["bin_eq_bout"] = True
                continue
            result[name] = float(value)
        else:
            result[item] = True
    return result


def parse_motor_limits(text: str) -> dict[str, tuple[float, float]]:
    """Parse comma-separated ``motor=min:max`` limits in degrees."""
    valid = {"mu", "delta", "nu", "eta", "chi", "phi"}
    limits = {}
    for item in re.split(r"[,;]", text):
        item = item.strip()
        if not item:
            continue
        if "=" not in item or ":" not in item:
            raise ValueError(f"Invalid motor range '{item}'; use motor=min:max.")
        name, bounds = (part.strip() for part in item.split("=", 1))
        if name not in valid:
            raise ValueError(f"Unknown You motor '{name}'.")
        lower, upper = (float(value.strip()) for value in bounds.split(":", 1))
        if lower > upper:
            raise ValueError(f"Minimum exceeds maximum for {name}.")
        limits[name] = (lower, upper)
    return limits


def motor_in_limits(solution: MotorSolution, limits: dict[str, tuple[float, float]]) -> bool:
    return all(lower - 1e-9 <= getattr(solution, name) <= upper + 1e-9 for name, (lower, upper) in limits.items())


def apply_motor_limits(solution: MotorSolution, limits: dict[str, tuple[float, float]]):
    """Move periodic angles by 360 degrees into their requested cuts."""
    values = {}
    for name, (lower, upper) in limits.items():
        value = getattr(solution, name)
        equivalents = [value + 360.0 * turn for turn in range(-3, 4)]
        allowed = [candidate for candidate in equivalents if lower - 1e-9 <= candidate <= upper + 1e-9]
        if not allowed:
            return None
        values[name] = min(allowed, key=lambda candidate: (abs(candidate - value), candidate))
    return replace(solution, **values)


def _calculator(cell: UnitCell, constraints: dict[str, float | bool], surface_hkl=None, u_matrix=None):
    try:
        from diffcalc.hkl.calc import HklCalculation
        from diffcalc.hkl.constraints import Constraints
        from diffcalc.ub.calc import UBCalculation
    except ImportError as exc:
        raise RuntimeError("Motor calculations require diffcalc-core. Install requirements.txt.") from exc
    ubcalc = UBCalculation("XPeak")
    with redirect_stdout(io.StringIO()):
        ubcalc.set_lattice("crystal", cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
        ubcalc.set_u(np.eye(3) if u_matrix is None else np.asarray(u_matrix, dtype=float))
    if surface_hkl is not None:
        surface = tuple(float(v) for v in surface_hkl)
        # Diffcalc uses surf_nhkl for betain/betaout and n_hkl for the more
        # general reference-vector angles (alpha/beta/psi). Keep both tied to
        # the crystal surface selected in XPeak.
        ubcalc.surf_nhkl = surface
        ubcalc.n_hkl = surface
    return HklCalculation(ubcalc, Constraints(constraints))


def hkl_to_motor_solutions(cell, hkl, wavelength, constraints, surface_hkl=None, motor_limits=None, u_matrix=None) -> list[MotorSolution]:
    calc = _calculator(cell, constraints, surface_hkl, u_matrix)
    solutions = [MotorSolution(*position.astuple, dict(virtual)) for position, virtual in calc.get_position(*hkl, wavelength)]
    if not motor_limits:
        return solutions
    adjusted = [apply_motor_limits(solution, motor_limits) for solution in solutions]
    return [solution for solution in adjusted if solution is not None]


def peak_surface_geometries(cell, hkls, wavelength, constraints, surface_hkl, target_alpha=0.0):
    """Return the sector nearest target incidence for each reachable HKL."""
    calc = _calculator(cell, constraints, surface_hkl)
    result = {}
    for hkl in hkls:
        try:
            solutions = [MotorSolution(*position.astuple, dict(virtual)) for position, virtual in calc.get_position(*hkl, wavelength)]
            finite = [s for s in solutions if np.isfinite(s.virtual.get("alpha", np.nan)) and np.isfinite(s.virtual.get("beta", np.nan))]
            if finite:
                result[tuple(hkl)] = min(finite, key=lambda s: abs(s.virtual["alpha"] - target_alpha))
        except Exception:
            continue
    return result


DIFFCALC_VALUE_ANGLES = (
    "alpha", "beta",
    "betain", "betaout", "psi",
    "qaz", "naz", "delta", "nu", "mu", "eta", "chi", "phi", "omega",
)
SCANNABLE_ANGLES = tuple(name for name in DIFFCALC_VALUE_ANGLES if name not in {"alpha", "beta"})


def solution_angle(solution: MotorSolution, name: str) -> float:
    """Return a physical or virtual angle from a Diffcalc motor solution."""
    if name in {"mu", "delta", "nu", "eta", "chi", "phi"}:
        return float(getattr(solution, name))
    value = solution.virtual.get(name)
    if value is None or not np.isfinite(value):
        raise ValueError(f"Angle '{name}' is not available for this solution.")
    return float(value)


def scan_reflection_geometry(cell, hkl, wavelength, scan_name, scan_values, base_constraints, surface_hkl, motor_limits=None, u_matrix=None):
    """Scan one Diffcalc angle while holding two other constraints fixed."""
    if scan_name not in DIFFCALC_VALUE_ANGLES:
        raise ValueError(f"Unknown scan angle '{scan_name}'.")
    if scan_name in base_constraints:
        raise ValueError(f"The scanned angle '{scan_name}' must not also be fixed.")
    if len(base_constraints) != 2:
        raise ValueError("Enter exactly two fixed constraints; the scan supplies the third constraint.")
    rows = []
    for requested in scan_values:
        constraints = dict(base_constraints)
        constraints[scan_name] = float(requested)
        try:
            for solution in hkl_to_motor_solutions(cell, hkl, wavelength, constraints, surface_hkl, motor_limits, u_matrix):
                rows.append((float(requested), solution))
        except Exception:
            continue
    return rows


def scan_alpha_beta(cell, hkl, wavelength, base_constraints, surface_hkl, alpha_values, motor_limits=None, u_matrix=None):
    """Backward-compatible incidence scan, including specular reflections."""
    if "alpha" in base_constraints:
        raise ValueError("The scanned angle 'alpha' must not also be fixed.")
    if len(base_constraints) != 2:
        raise ValueError("Enter exactly two fixed constraints; the scan supplies the third constraint.")
    reciprocal = diffcalc_reciprocal_matrix(cell)
    q_vector = reciprocal @ np.asarray(hkl, dtype=float)
    normal_vector = reciprocal @ np.asarray(surface_hkl, dtype=float)
    parallel = np.linalg.norm(np.cross(q_vector, normal_vector)) <= 1e-8 * np.linalg.norm(q_vector) * np.linalg.norm(normal_vector)
    if parallel:
        theta = np.degrees(np.arcsin(np.clip(wavelength * np.linalg.norm(q_vector) / 2.0, -1.0, 1.0)))
        alpha_values = np.asarray(list(alpha_values), dtype=float)
        if alpha_values.size == 0 or theta < alpha_values.min() - 1e-9 or theta > alpha_values.max() + 1e-9:
            return []
        return [(solution.virtual["alpha"], solution) for solution in _specular_fallback_solutions(cell, hkl, wavelength, base_constraints, surface_hkl, motor_limits, u_matrix)]
    return scan_reflection_geometry(cell, hkl, wavelength, "alpha", alpha_values, base_constraints, surface_hkl, motor_limits, u_matrix)


def _specular_fallback_solutions(cell, hkl, wavelength, base_constraints, surface_hkl, motor_limits, u_matrix):
    """Resolve the redundant-reference singularity by fixing a free sample motor."""
    trial_names = ("phi", "chi", "eta", "mu")
    last_error = None
    for name in trial_names:
        if name in base_constraints:
            continue
        constraints = dict(base_constraints)
        constraints[name] = 0.0
        try:
            solutions = hkl_to_motor_solutions(cell, hkl, wavelength, constraints, surface_hkl, motor_limits, u_matrix)
            if solutions:
                return solutions
        except Exception as exc:
            last_error = exc
    if last_error:
        raise last_error
    return []


def solve_symmetric_geometry(cell, hkl, wavelength, base_constraints, surface_hkl, motor_limits=None, u_matrix=None):
    """Solve betain=betaout, including the Q-parallel-surface specular case."""
    reciprocal = diffcalc_reciprocal_matrix(cell)
    q_vector = reciprocal @ np.asarray(hkl, dtype=float)
    normal_vector = reciprocal @ np.asarray(surface_hkl, dtype=float)
    parallel = np.linalg.norm(np.cross(q_vector, normal_vector)) <= 1e-8 * np.linalg.norm(q_vector) * np.linalg.norm(normal_vector)
    if parallel:
        return _specular_fallback_solutions(cell, hkl, wavelength, base_constraints, surface_hkl, motor_limits, u_matrix)
    constraints = dict(base_constraints)
    constraints["bin_eq_bout"] = True
    return hkl_to_motor_solutions(cell, hkl, wavelength, constraints, surface_hkl, motor_limits, u_matrix)


def calculate_ub_from_two_reflections(cell, wavelength, references):
    """Calculate U/UB from two general ``(hkl, six You motors)`` measurements."""
    try:
        from diffcalc.hkl.geometry import Position
        from diffcalc.ub.calc import UBCalculation
    except ImportError as exc:
        raise RuntimeError("UB calculations require diffcalc-core.") from exc
    if len(references) != 2:
        raise ValueError("Exactly two measured reflections are required.")
    ubcalc = UBCalculation("XPeak measured UB")
    energy_kev = 12.398419843320026 / wavelength
    with redirect_stdout(io.StringIO()):
        ubcalc.set_lattice("crystal", cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
        for index, (hkl, motors) in enumerate(references, 1):
            if len(motors) != 6:
                raise ValueError("Each reflection requires six You angles: mu, delta, nu, eta, chi, phi.")
            position = Position(*(float(value) for value in motors))
            ubcalc.add_reflection(tuple(hkl), position, energy_kev, f"ref{index}")
        ubcalc.calc_ub("ref1", "ref2")
    return np.asarray(ubcalc.U, dtype=float), np.asarray(ubcalc.UB, dtype=float)


def calculate_ub_from_two_bisecting_reflections(cell, wavelength, references):
    """Backward-compatible wrapper for ``(hkl, theta, chi, phi)`` references."""
    general = []
    for hkl, theta, chi, phi in references:
        general.append((hkl, (0.0, 2.0 * theta, 0.0, theta, chi, phi)))
    return calculate_ub_from_two_reflections(cell, wavelength, general)


def motors_to_hkl(cell, motors, wavelength, u_matrix=None) -> tuple[float, float, float]:
    try:
        from diffcalc.hkl.geometry import Position
    except ImportError as exc:
        raise RuntimeError("Motor calculations require diffcalc-core. Install requirements.txt.") from exc
    calc = _calculator(cell, {}, u_matrix=u_matrix)
    return tuple(float(v) for v in calc.get_hkl(Position(*motors), wavelength))


def diffcalc_reciprocal_matrix(cell: UnitCell) -> np.ndarray:
    """Return Diffcalc's Busing-Levy B matrix without its 2*pi factor."""
    from diffcalc.ub.calc import UBCalculation
    ubcalc = UBCalculation("XPeak")
    with redirect_stdout(io.StringIO()):
        ubcalc.set_lattice("crystal", cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)
        ubcalc.set_u(np.eye(3))
    return np.asarray(ubcalc.UB, dtype=float) / (2.0 * np.pi)


def orientation_from_surface(cell: UnitCell, surface_hkl, in_plane_angle_deg: float) -> np.ndarray:
    """Construct U from a surface normal and an in-plane mounting rotation.

    The surface normal is aligned with +z in the zero-motor phi frame. At zero
    in-plane angle the projection of a* (or the first usable reciprocal basis
    vector) is aligned with +x; positive rotation is toward +y.
    """
    reciprocal = diffcalc_reciprocal_matrix(cell)
    normal = reciprocal @ np.asarray(surface_hkl, dtype=float)
    if np.linalg.norm(normal) <= 1e-12:
        raise ValueError("Surface-normal HKL must be nonzero.")
    normal /= np.linalg.norm(normal)
    in_plane = None
    for index in range(3):
        candidate = reciprocal[:, index]
        projected = candidate - np.dot(candidate, normal) * normal
        if np.linalg.norm(projected) > 1e-9 * np.linalg.norm(candidate):
            in_plane = projected / np.linalg.norm(projected)
            break
    if in_plane is None:
        raise ValueError("Could not construct an in-plane crystallographic direction.")
    source = np.column_stack((in_plane, np.cross(normal, in_plane), normal))
    angle = np.radians(float(in_plane_angle_deg))
    target_x = np.array([np.cos(angle), np.sin(angle), 0.0])
    target = np.column_stack((target_x, np.cross([0.0, 0.0, 1.0], target_x), [0.0, 0.0, 1.0]))
    u_matrix = target @ source.T
    if np.linalg.det(u_matrix) < 0 or not np.allclose(u_matrix.T @ u_matrix, np.eye(3), atol=1e-10):
        raise ValueError("Surface mounting did not produce a proper rotation matrix.")
    return u_matrix
