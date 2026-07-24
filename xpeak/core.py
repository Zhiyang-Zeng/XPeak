"""Numerical core for peak finding and a simple monochromatic XRD experiment."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from functools import lru_cache

import numpy as np

HC_KEV_ANGSTROM = 12.398419843320026


def energy_kev_to_wavelength(energy_kev: float) -> float:
    if energy_kev <= 0:
        raise ValueError("Photon energy must be positive.")
    return HC_KEV_ANGSTROM / energy_kev


def wavelength_to_energy_kev(wavelength: float) -> float:
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")
    return HC_KEV_ANGSTROM / wavelength


@dataclass(frozen=True)
class UnitCell:
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def direct_matrix(self) -> np.ndarray:
        """Return columns containing the direct lattice vectors in Angstrom."""
        ar, br, gr = np.radians([self.alpha, self.beta, self.gamma])
        sg = math.sin(gr)
        if min(self.a, self.b, self.c) <= 0 or abs(sg) < 1e-10:
            raise ValueError("Cell lengths must be positive and gamma must not be 0 or 180 degrees.")
        va = np.array([self.a, 0.0, 0.0])
        vb = np.array([self.b * math.cos(gr), self.b * sg, 0.0])
        cx = self.c * math.cos(br)
        cy = self.c * (math.cos(ar) - math.cos(br) * math.cos(gr)) / sg
        cz2 = self.c**2 - cx**2 - cy**2
        if cz2 <= 0:
            raise ValueError("The cell angles do not define a physical unit cell.")
        return np.column_stack((va, vb, np.array([cx, cy, math.sqrt(cz2)])))

    def reciprocal_matrix(self) -> np.ndarray:
        """Return reciprocal basis without the 2*pi convention, in 1/Angstrom."""
        return np.linalg.inv(self.direct_matrix()).T


@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    b_iso: float = 0.0


@dataclass(frozen=True)
class Peak:
    h: int
    k: int
    l: int
    d: float
    two_theta: float
    q: float
    intensity: float


# Atomic numbers are a useful, transparent low-angle form-factor approximation.
ATOMIC_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "Na": 11, "Mg": 12, "Al": 13,
    "Si": 14, "P": 15, "S": 16, "Cl": 17, "K": 19, "Ca": 20,
    "Ti": 22, "Cr": 24, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "Se": 34, "Br": 35, "Sr": 38,
    "Zr": 40, "Mo": 42, "Ag": 47, "Sn": 50, "I": 53, "Ba": 56,
    "W": 74, "Pt": 78, "Au": 79, "Pb": 82,
}


@lru_cache(maxsize=4096)
def atomic_scattering_factor(element: str, q: float, energy_ev: float | None = None) -> complex:
    """Return f(q, E), preferring xrayutilities' tabulated complex factors.

    q follows the xrayutilities convention (2*pi/d, in inverse Angstrom). If
    xrayutilities is not installed or does not know the element, atomic number
    is used as a documented low-angle approximation.
    """
    symbol = element.capitalize()
    try:
        import xrayutilities as xu

        table_element = getattr(xu.materials.elements, symbol)
        energy = energy_ev if energy_ev is not None else "config"
        return complex(table_element.f(q, energy))
    except (ImportError, AttributeError, KeyError, TypeError, ValueError):
        return complex(ATOMIC_Z.get(symbol, 1))


def structure_factor(
    hkl: tuple[int, int, int],
    atoms: list[Atom],
    d: float,
    energy_ev: float | None = None,
) -> complex:
    h, k, l = hkl
    total = 0j
    s2 = 1.0 / (4.0 * d * d)
    for atom in atoms:
        q = 2.0 * math.pi / d
        form_factor = atomic_scattering_factor(atom.element, round(q, 10), energy_ev)
        dw = math.exp(-max(0.0, atom.b_iso) * s2)
        phase = 2j * math.pi * (h * atom.x + k * atom.y + l * atom.z)
        total += atom.occupancy * form_factor * dw * np.exp(phase)
    return total


def enumerate_peaks(
    cell: UnitCell,
    atoms: list[Atom],
    wavelength: float,
    max_index: int = 8,
    min_two_theta: float = 1.0,
    max_two_theta: float = 90.0,
    min_relative_intensity: float = 0.1,
) -> list[Peak]:
    if wavelength <= 0:
        raise ValueError("Wavelength must be positive.")
    try:
        from .diffcalc_backend import diffcalc_reciprocal_matrix
        reciprocal = diffcalc_reciprocal_matrix(cell)
    except (ImportError, RuntimeError):
        reciprocal = cell.reciprocal_matrix()
    energy_ev = 12398.419843320026 / wavelength
    raw: list[Peak] = []
    for h, k, l in product(range(-max_index, max_index + 1), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue
        # Keep one member of each Friedel pair for the search table.
        if next(value for value in (h, k, l) if value != 0) < 0:
            continue
        g = reciprocal @ np.array([h, k, l], dtype=float)
        q_norm = float(np.linalg.norm(g))
        d = 1.0 / q_norm
        sin_theta = wavelength / (2.0 * d)
        if sin_theta >= 1.0:
            continue
        two_theta = math.degrees(2.0 * math.asin(sin_theta))
        if not min_two_theta <= two_theta <= max_two_theta:
            continue
        intensity = abs(structure_factor((h, k, l), atoms, d, energy_ev)) ** 2
        # Unpolarized Lorentz-polarization factor; capped near the direct beam.
        theta = math.radians(two_theta / 2.0)
        lp = (1.0 + math.cos(2.0 * theta) ** 2) / max(
            1e-8, math.sin(theta) ** 2 * math.cos(theta)
        )
        raw.append(Peak(h, k, l, d, two_theta, 2 * math.pi / d, intensity * lp))
    if not raw:
        return []
    scale = max(p.intensity for p in raw)
    peaks = [Peak(p.h, p.k, p.l, p.d, p.two_theta, p.q, 100 * p.intensity / scale) for p in raw]
    return sorted((p for p in peaks if p.intensity >= min_relative_intensity), key=lambda p: p.two_theta)


def you_sample_rotation(mu: float, eta: float, chi: float, phi: float) -> np.ndarray:
    """4S sample rotation Z = M(mu) H(eta) X(chi) Phi(phi).

    This follows You, J. Appl. Cryst. 32 (1999) 614-623, equations (5),
    (11). Lab x is vertical, lab y is the incident beam, and lab z completes
    the right-handed frame. eta and phi are left-handed; mu and chi are
    right-handed according to the paper's motor definitions.
    """
    mu, eta, chi, phi = np.radians([mu, eta, chi, phi])
    m = np.array([[1, 0, 0], [0, np.cos(mu), -np.sin(mu)], [0, np.sin(mu), np.cos(mu)]])
    h = np.array([[np.cos(eta), np.sin(eta), 0], [-np.sin(eta), np.cos(eta), 0], [0, 0, 1]])
    x = np.array([[np.cos(chi), 0, np.sin(chi)], [0, 1, 0], [-np.sin(chi), 0, np.cos(chi)]])
    ph = np.array([[np.cos(phi), np.sin(phi), 0], [-np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    return m @ h @ x @ ph


def you_detector_frame(delta: float, nu: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return detector normal and in-plane axes for You (1999) Delta/Pi.

    The normal is k_f/k from equation (9). The two in-plane axes are the
    correspondingly rotated zero-angle detector x and z axes.
    """
    delta, nu = np.radians([delta, nu])
    d = np.array([[np.cos(delta), np.sin(delta), 0], [-np.sin(delta), np.cos(delta), 0], [0, 0, 1]])
    p = np.array([[1, 0, 0], [0, np.cos(nu), -np.sin(nu)], [0, np.sin(nu), np.cos(nu)]])
    transform = p @ d
    return transform @ np.array([0.0, 1.0, 0.0]), transform @ np.array([1.0, 0.0, 0.0]), transform @ np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class DetectorSpot:
    h: int
    k: int
    l: int
    x_px: float
    y_px: float
    intensity: float
    mismatch: float


def simulate_detector(
    cell: UnitCell,
    atoms: list[Atom],
    wavelength: float,
    sample_motors: tuple[float, float, float, float],
    detector_motors: tuple[float, float],
    detector_distance_mm: float,
    pixel_size_mm: float,
    detector_shape: tuple[int, int],
    beam_center: tuple[float, float],
    max_index: int = 8,
    mosaicity_deg: float = 0.25,
    u_matrix: np.ndarray | None = None,
) -> list[DetectorSpot]:
    """Project reflections onto a 4S+2D detector using You (1999) conventions.

    Lab x is vertical and the incident beam is +y. The detector is tangent to
    the direction selected by delta and nu. Reciprocal vectors and wavevectors
    are divided by 2*pi relative to the paper, leaving the Ewald construction
    and all projected directions unchanged.
    """
    try:
        from .diffcalc_backend import diffcalc_reciprocal_matrix
        reciprocal = diffcalc_reciprocal_matrix(cell)
    except (ImportError, RuntimeError):
        reciprocal = cell.reciprocal_matrix()
    energy_ev = 12398.419843320026 / wavelength
    rotation = you_sample_rotation(*sample_motors) @ (np.eye(3) if u_matrix is None else np.asarray(u_matrix, dtype=float))
    k0 = np.array([0.0, 1.0 / wavelength, 0.0])
    detector_normal, detector_u, detector_v = you_detector_frame(*detector_motors)
    width, height = detector_shape
    spots: list[DetectorSpot] = []
    tolerance = math.radians(max(mosaicity_deg, 0.001)) / wavelength
    for h, k, l in product(range(-max_index, max_index + 1), repeat=3):
        if (h, k, l) == (0, 0, 0):
            continue
        g = rotation @ reciprocal @ np.array([h, k, l], dtype=float)
        kout = k0 + g
        mismatch = abs(float(np.linalg.norm(kout)) - 1.0 / wavelength)
        if mismatch > 3.0 * tolerance:
            continue
        ray = kout / np.linalg.norm(kout)
        denominator = float(ray @ detector_normal)
        if denominator <= 0:
            continue
        hit = detector_distance_mm * ray / denominator
        offset = hit - detector_distance_mm * detector_normal
        x_mm = float(offset @ detector_u)
        y_mm = float(offset @ detector_v)
        x_px = beam_center[0] + x_mm / pixel_size_mm
        y_px = beam_center[1] - y_mm / pixel_size_mm
        if not (0 <= x_px < width and 0 <= y_px < height):
            continue
        d = 1.0 / float(np.linalg.norm(g))
        base = abs(structure_factor((h, k, l), atoms, d, energy_ev)) ** 2
        rocking = math.exp(-0.5 * (mismatch / tolerance) ** 2)
        spots.append(DetectorSpot(h, k, l, x_px, y_px, base * rocking, mismatch))
    return spots
