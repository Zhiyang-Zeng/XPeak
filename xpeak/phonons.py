"""Phonopy band.yaml loading and frozen-phonon intensity calculations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import yaml

from .core import Atom, UnitCell, structure_factor


@dataclass(frozen=True)
class PhononMode:
    q_index: int
    mode_index: int
    q: tuple[float, float, float]
    frequency_thz: float
    eigenvector: np.ndarray  # complex Cartesian, shape (natom, 3)
    ir_label: str | None = None


@dataclass(frozen=True)
class PhononDataset:
    cell: UnitCell
    lattice: np.ndarray  # row-vector convention from Phonopy
    atoms: list[Atom]
    masses: np.ndarray
    modes: list[PhononMode]
    point_group: str | None = None


def load_phonopy_yaml(path: str) -> PhononDataset:
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    lattice = np.asarray(data["lattice"], dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)
    def angle(u, v):
        return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
    cell = UnitCell(*lengths, angle(lattice[1], lattice[2]), angle(lattice[0], lattice[2]), angle(lattice[0], lattice[1]))
    points = data["points"]
    atoms = [Atom(p["symbol"], *map(float, p["coordinates"])) for p in points]
    masses = np.asarray([p["mass"] for p in points], dtype=float)
    modes = []
    for qi, phonon in enumerate(data["phonon"]):
        q = tuple(float(x) for x in phonon["q-position"])
        for mi, band in enumerate(phonon["band"]):
            raw = np.asarray(band["eigenvector"], dtype=float)
            eigenvector = raw[..., 0] + 1j * raw[..., 1]
            if eigenvector.shape != (len(atoms), 3):
                raise ValueError(f"q-point {qi + 1}, mode {mi + 1}: unexpected eigenvector shape {eigenvector.shape}")
            modes.append(PhononMode(qi, mi, q, float(band["frequency"]), eigenvector))
    return PhononDataset(cell, lattice, atoms, masses, modes)


def assign_irreps(dataset: PhononDataset, path: str, frequency_tolerance: float = 1e-4):
    """Assign Phonopy irreps.yaml labels by q-point and one-based band index."""
    with open(path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    q = np.asarray(data["q-position"], dtype=float)
    assignments = {}
    mismatches = []
    for normal_mode in data.get("normal_modes", []):
        label = str(normal_mode.get("ir_label", "")).strip() or "unlabeled"
        frequency = float(normal_mode["frequency"])
        for band_index in normal_mode["band_indices"]:
            assignments[int(band_index) - 1] = (label, frequency)
    updated = []
    matched = 0
    for mode in dataset.modes:
        if np.linalg.norm(np.asarray(mode.q) - q) <= 1e-7 and mode.mode_index in assignments:
            label, ir_frequency = assignments[mode.mode_index]
            if abs(mode.frequency_thz - ir_frequency) > frequency_tolerance:
                mismatches.append((mode.q_index + 1, mode.mode_index + 1, mode.frequency_thz, ir_frequency))
            else:
                mode = replace(mode, ir_label=label)
                matched += 1
        updated.append(mode)
    if not matched:
        raise ValueError("No modes matched the irreps q-point, band indices, and frequencies.")
    return replace(dataset, modes=updated, point_group=str(data.get("point_group", "unknown"))), matched, mismatches


def displaced_atoms(dataset: PhononDataset, mode: PhononMode, amplitude: float, phase_deg: float = 0.0) -> list[Atom]:
    if np.linalg.norm(mode.q) > 1e-8:
        raise ValueError("Non-Gamma phonons require a commensurate supercell and are not yet supported.")
    phase = np.exp(1j * math.radians(phase_deg))
    cartesian = amplitude * np.real(mode.eigenvector * phase) / np.sqrt(dataset.masses)[:, None]
    fractional = cartesian @ np.linalg.inv(dataset.lattice)
    return [Atom(a.element, a.x + d[0], a.y + d[1], a.z + d[2], a.occupancy, a.b_iso) for a, d in zip(dataset.atoms, fractional)]


def mode_intensity_curves(dataset, mode, hkls, amplitudes, wavelength, phase_deg=0.0):
    energy_ev = 12398.419843320026 / wavelength
    reciprocal = dataset.cell.reciprocal_matrix()
    curves = {tuple(hkl): [] for hkl in hkls}
    for amplitude in amplitudes:
        atoms = displaced_atoms(dataset, mode, float(amplitude), phase_deg)
        for hkl in curves:
            d = 1.0 / np.linalg.norm(reciprocal @ np.asarray(hkl, dtype=float))
            curves[hkl].append(abs(structure_factor(hkl, atoms, d, energy_ev)) ** 2)
    return {hkl: np.asarray(values) for hkl, values in curves.items()}
