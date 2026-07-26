# XPeak changelog

## Version 2.1 — 26 July 2026

Version 2.1 expands XPeak from a reflection finder into a broader
single-crystal X-ray experiment-planning workstation.

### Peak and phonon analysis

- Load CIF and VASP POSCAR/CONTCAR structures and display the Busing–Levy `B`
  matrix.
- Load Phonopy `band.yaml` eigenvectors and `irreps.yaml` symmetry labels.
- Monitor selected reflections and calculate relative intensity versus a
  frozen-phonon coordinate.
- Enter the incident beam using either wavelength or photon energy.

### Reflection geometry

- Scan any supported Diffcalc motor or virtual angle under two independent
  constraints.
- Calculate incidence `betain` and exit `betaout` relative to a selected crystal
  surface, including direct `betain = betaout` solutions.
- Define `U` from a surface normal and in-plane rotation.
- Inspect the sample, beams, detector, surface normal, and right-handed crystal
  axes in an interactive, zoomable 3D view.

### Virtual experiment

- Solve reflections using the You Eulerian convention and display the
  corresponding kappa-goniometer position.
- Convert entered Eulerian and kappa positions in either direction.
- Calculate `Q = (H,K,L)` from the current motor condition.
- Determine general `U` and `UB` matrices from two measured reflections.
- Simulate accessible reflections on a 512 × 512 pixel area detector.
- Show detector `+x/u` and `+y/v` directions directly in both the detector image
  and 3D instrument view.
- Use a `0.01°` default mosaic acceptance and Diffcalc's unrestricted default
  Eulerian motor positions, with optional beamline-specific limits.

### Interface and documentation

- Add a dedicated You-convention reference panel using the published 4S+2D
  diffractometer geometry.
- Add facility-style controls, larger fonts, scalable 3D views, camera presets,
  grid frames, and persistent camera orientation.
- Expand the README for crystal physicists with conventions, equations, file
  formats, constraints, and experiment-planning guidance.
