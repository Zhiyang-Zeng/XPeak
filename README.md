# XPeak

XPeak is a Python desktop application for planning single-crystal X-ray
diffraction experiments. It combines structure-factor calculations, phonon-mode
analysis, six-circle diffractometer geometry, and a virtual area detector in one
GUI.

The program is intended as an experiment-planning and visualization tool. It is
not a replacement for a beamline-specific control or resolution package.

## What can XPeak do?

### 1. Find useful Bragg reflections

- Enumerate `(h k l)` reflections for a general triclinic unit cell.
- Calculate `d`, `|Q| = 2π/d`, Bragg angle `θ`, `2θ`, structure factor, and
  relative intensity.
- Filter peaks by reciprocal-space range, scattering angle, and intensity.
- Load crystal structures from CIF or VASP POSCAR/CONTCAR files.
- Enter the incident X-ray beam as wavelength in Å or photon energy in keV.
- Export the reflection table to CSV.

### 2. Study phonon-sensitive reflections

- Load phonon frequencies and eigenvectors from a Phonopy `band.yaml` file.
- Load mode symmetries from a Phonopy `irreps.yaml` file.
- Select several reflections and plot their relative intensity change as a
  function of a frozen-phonon normal coordinate.
- Identify peaks that are especially sensitive to a chosen structural mode.

### 3. Explore incidence and exit geometry

- For a selected reflection and crystal-surface normal, calculate the X-ray
  incidence angle `α` and exit angle `β`.
- Scan `α` and display all accessible `β` branches and motor solutions.
- Solve the symmetric condition `α = β` directly.
- Inspect the sample, crystallographic axes, beams, scattering vector, and
  detector in an interactive 3D view.

Here `α` and `β` are measured relative to the sample surface. For a specular
reflection, exact elastic diffraction fixes `α = β = θ`; `α` cannot be varied
independently while remaining on the same reciprocal-lattice point.

### 4. Run a virtual diffractometer experiment

- Solve `(h k l) →` six-circle motor positions with
  [Diffcalc Core](https://github.com/DiamondLightSource/diffcalc).
- Calculate the inverse mapping from current motor angles to `(h k l)`.
- Display both the You Eulerian angles and the corresponding configurable kappa
  goniometer position.
- Determine `U` and `UB` from two measured reflections.
- Simulate where accessible reflections appear on a flat 2D detector.
- Explore the complete instrument geometry in a second interactive 3D view.

## Quick start

Python 3.10 or newer is recommended. Tkinter is supplied with most Python
installations.

```bash
git clone https://github.com/Zhiyang-Zeng/XPeak.git
cd XPeak
python3 -m pip install -r requirements.txt
python3 run_xpeak.py
```

A typical workflow is:

1. Open **Peak finder** and load a CIF or POSCAR.
2. Enter the photon energy or wavelength and choose the `(h k l)` search range.
3. Calculate and filter the reflection table.
4. Optionally load `band.yaml` and `irreps.yaml`, select a phonon mode, and add
   reflections to the monitored list.
5. Send a reflection to **Alpha-beta geometry** to inspect surface accessibility.
6. Send an accessible solution to **Virtual experiment** and inspect the detector
   image and both motor readouts.

## Scattering and intensity model

For atoms at fractional coordinates `r_j`, XPeak evaluates the kinematic
structure factor

```text
F(hkl) = Σ_j occ_j f_j(Q,E) exp[-B_j |Q|²/(16π²)] exp[2πi (hkl · r_j)]
```

and reports a relative intensity derived from `|F|²`, including the selected
Lorentz-polarization weighting. When available, `xrayutilities` supplies the
energy-dependent complex atomic scattering factors. If no tabulated factor can
be obtained for an element, XPeak falls back to its atomic number as an
approximation.

The energy-wavelength conversion is

```text
E [keV] × λ [Å] = 12.398419843
```

The intensities are useful for comparing reflections within the model. Absolute
intensities require experimental scale factors and corrections for effects such
as absorption, extinction, footprint, detector efficiency, and instrumental
resolution.

## Phonon convention

XPeak interprets Phonopy eigenvectors as complex Cartesian, mass-normalized
vectors. For normal coordinate `Q_mode`, the displacement of atom `j` is

```text
u_j(Q_mode) = Q_mode Re[e_j exp(i phase)] / sqrt(m_j)
```

where `Q_mode` is entered in Å√amu. Gamma-point modes are supported. A
non-Gamma mode requires a commensurate supercell and is therefore rejected
rather than treated incorrectly as a Gamma distortion.

Symmetry labels from `irreps.yaml` are assigned using the q-point and one-based
band indices, with a frequency consistency check. Degenerate bands receive the
same irreducible-representation label.

## Diffractometer convention

The virtual experiment follows H. You, *J. Appl. Cryst.* **32** (1999),
614–623. The laboratory axes are

- `+x`: vertical;
- `+y`: along the incident beam;
- `+z`: completes the right-handed laboratory frame.

The sample transformation is

```text
Z = M(μ) H(η) X(χ) Φ(φ)
```

and the detector-center direction is obtained from

```text
Π(ν) Δ(δ) k_i
```

In this convention, `η`, `φ`, and `δ` are left-handed motor angles, while
`μ`, `χ`, and `ν` are right-handed. The detector's in-plane `u` and `v` axes
rotate with the detector arm.

### Default motor ranges

```text
μ=0:360, δ=0:180, ν=0:360, η=0:360, χ=0:180, φ=0:360
```

The ranges are inclusive and can be edited as `motor=min:max` in degrees.
Periodically equivalent angles are first mapped into the requested interval;
solutions that remain outside a motor range are excluded.

### Diffcalc constraints

Enter a target reflection and three independent Diffcalc constraints, for
example

```text
ν=0, μ=0, a_eq_b
```

then select **Solve HKL → motors**. The returned table may contain several
kinematically equivalent sectors. Select the sector compatible with the actual
beamline limits before applying it to the detector simulation.

### UB matrix from two reflections

For each reference reflection, enter `(h k l)` and its measured `θ`, `χ`, and
`φ`. XPeak currently interprets each reference as a vertical bisecting
measurement:

```text
η = θ, δ = 2θ, μ = 0, ν = 0
```

The resulting orientation matrix is used by motor solving, inverse HKL
calculation, alpha-beta scans, detector simulation, and both 3D views. This
assumption must match the geometry used when the two references were measured.

### Kappa readout

Every applied You position is also converted to

```text
(μ, δ, ν, kω, κ, kφ)
```

The kappa-axis inclination is configurable and defaults to 50°. XPeak solves
the full rotation matrix and verifies the reconstructed orientation. An
orientation outside the chosen kappa cradle's coverage is reported as
unreachable instead of being approximated.

## 3D-view controls

- Drag to rotate the view.
- Use the mouse wheel to zoom.
- **Side view** restores the You-paper laboratory view, with the incident beam
  travelling from right to left and `x` vertical.
- **X-ray incidence view** looks downstream along the incident beam, again with
  `x` vertical.

Changing a peak or motor solution does not reset the camera. The sample's
oriented `a`, `b`, and `c` directions and the positive laboratory axes are
displayed for reference.

## Structure-file support

The structure loader accepts CIF, POSCAR, CONTCAR, `*.vasp`, and extensionless
VASP files. Manual atom input uses one atom per line:

```text
element, fractional_x, fractional_y, fractional_z, occupancy, B_iso
```

Occupancy defaults to 1 and isotropic `B` defaults to 0 when omitted.

## Important limitations

- The detector model is a flat ideal area detector without a complete
  beamline-resolution function.
- Instrument-specific detector tilts, offsets, polarization conventions, and
  motor zero corrections must be calibrated for the target beamline.
- Multiple scattering, absorption, extinction, sample shape, finite mosaic
  distributions, and diffuse scattering are not modeled completely.
- Frozen-phonon curves describe the selected static displacement pattern; they
  are not a calculation of time-resolved populations or dynamical structure
  factors.

Always validate predicted motor positions against the beamline's control
software, collision limits, and local safety procedures before moving hardware.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Acknowledgements

Motor-angle calculations use Diamond Light Source's Apache-2.0-licensed
[Diffcalc](https://github.com/DiamondLightSource/diffcalc). The six-circle
geometry follows H. You, *J. Appl. Cryst.* **32** (1999), 614–623.
