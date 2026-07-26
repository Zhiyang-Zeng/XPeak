from __future__ import annotations

import csv
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import matplotlib as mpl
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .core import Atom, UnitCell, energy_kev_to_wavelength, enumerate_peaks, simulate_detector, wavelength_to_energy_kev, you_detector_frame, you_sample_rotation
from .diffcalc_backend import SCANNABLE_ANGLES, calculate_ub_from_two_reflections, diffcalc_reciprocal_matrix, hkl_to_motor_solutions, kappa_to_you, motors_to_hkl, orientation_from_surface, parse_constraints, parse_motor_limits, scan_reflection_geometry, solution_angle, solve_symmetric_geometry, you_to_kappa
from .phonons import assign_irreps, load_phonopy_yaml, mode_intensity_curves


class XPeakApp(tk.Tk):
    BG = "#f3f6fa"
    PANEL = "#ffffff"
    PANEL_2 = "#e7eef5"
    TEXT = "#172334"
    MUTED = "#61758a"
    ACCENT = "#2563eb"
    ACCENT_ACTIVE = "#1d4ed8"
    SUCCESS = "#15803d"

    def __init__(self) -> None:
        super().__init__()
        self.title("XPeak | Beamline Diffraction Workstation")
        self.geometry("1600x1050")
        self.minsize(1180, 760)
        self.configure(background=self.BG)
        self.peaks = []
        self.motor_solutions = []
        self.phonon_dataset = None
        self.monitored_hkls = []
        self.surface_scan_results = []
        # Diffcalc does not impose hardware travel limits by default. A blank
        # field therefore preserves every mathematical motor solution.
        self.motor_limits_text = tk.StringVar(value="")
        self.u_matrix = np.eye(3)
        self.ub_matrix = None
        self._style()
        self._build()
        self.find_peaks()

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.TEXT, bordercolor=self.PANEL_2, lightcolor=self.PANEL_2, darkcolor=self.BG)
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT, font=("TkDefaultFont", 11))
        style.configure("Muted.TLabel", foreground=self.MUTED)
        style.configure("Status.TLabel", background=self.PANEL, foreground=self.SUCCESS, padding=(12, 7), font=("TkFixedFont", 11, "bold"))
        style.configure("TLabelframe", background=self.BG, bordercolor="#b8c7d6", relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=self.BG, foreground=self.ACCENT, font=("TkDefaultFont", 12, "bold"))
        style.configure("TButton", background=self.PANEL_2, foreground=self.TEXT, bordercolor="#aebdca", padding=(12, 9), font=("TkDefaultFont", 11, "bold"))
        style.map("TButton", background=[("active", "#dbeafe"), ("pressed", "#bfdbfe")])
        style.configure("Accent.TButton", background=self.ACCENT, foreground="#ffffff", bordercolor=self.ACCENT, padding=(13, 9))
        style.map("Accent.TButton", background=[("active", self.ACCENT_ACTIVE), ("pressed", "#1e40af")])
        style.configure("TEntry", fieldbackground="#ffffff", foreground=self.TEXT, insertcolor=self.TEXT, bordercolor="#aebdca", padding=7, font=("TkDefaultFont", 11))
        style.configure("TCombobox", fieldbackground="#ffffff", background=self.PANEL_2, foreground=self.TEXT, arrowcolor=self.ACCENT, bordercolor="#aebdca", padding=6, font=("TkDefaultFont", 11))
        style.map("TCombobox", fieldbackground=[("readonly", "#ffffff")], foreground=[("readonly", self.TEXT)])
        style.configure("TRadiobutton", background=self.BG, foreground=self.TEXT, indicatorbackground="#ffffff", indicatormargin=5, font=("TkDefaultFont", 11))
        style.map("TRadiobutton", indicatorcolor=[("selected", self.ACCENT)])
        style.configure("TNotebook", background=self.BG, borderwidth=0, tabmargins=(0, 4, 0, 0))
        style.configure("TNotebook.Tab", background="#e7eef5", foreground=self.MUTED, padding=(20, 12), font=("TkDefaultFont", 12, "bold"))
        style.map("TNotebook.Tab", background=[("selected", "#ffffff"), ("active", "#dbeafe")], foreground=[("selected", self.ACCENT), ("active", self.TEXT)])
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=self.TEXT, rowheight=31, bordercolor="#c4d0dc", font=("TkDefaultFont", 11))
        style.configure("Treeview.Heading", background=self.PANEL_2, foreground="#1e40af", relief="flat", padding=8, font=("TkDefaultFont", 11, "bold"))
        style.map("Treeview", background=[("selected", "#dbeafe")], foreground=[("selected", "#172334")])
        style.map("Treeview.Heading", background=[("active", "#dbeafe")])
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("TkDefaultFont", 23, "bold"))
        style.configure("Badge.TLabel", background="#dbeafe", foreground="#1e40af", padding=(11, 6), font=("TkFixedFont", 10, "bold"))
        mpl.rcParams.update({
            "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
            "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
            "figure.facecolor": self.BG, "axes.facecolor": self.PANEL,
            "axes.edgecolor": "#8fa3b5", "axes.labelcolor": self.TEXT,
            "axes.titlecolor": self.TEXT, "text.color": self.TEXT,
            "xtick.color": self.MUTED, "ytick.color": self.MUTED,
            "grid.color": "#bdc9d4", "grid.alpha": 0.65,
            "legend.facecolor": self.PANEL, "legend.edgecolor": "#aebdca",
            "legend.labelcolor": self.TEXT,
        })

    def _build(self) -> None:
        header = ttk.Frame(self, padding=(20, 14))
        header.pack(fill="x")
        ttk.Label(header, text="XPEAK", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="  SINGLE-CRYSTAL DIFFRACTION WORKSTATION", style="Muted.TLabel").pack(side="left", pady=(7, 0))
        ttk.Label(header, text="BEAMLINE SIMULATION  •  LOCAL", style="Badge.TLabel").pack(side="right")
        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        inputs = ttk.LabelFrame(body, text="Crystal & beam", padding=12)
        body.add(inputs, weight=1)
        self._build_inputs(inputs)
        notebook = ttk.Notebook(body)
        self.notebook = notebook
        body.add(notebook, weight=4)
        peak_tab = ttk.Frame(notebook, padding=10)
        surface_tab = ttk.Frame(notebook, padding=10)
        sim_tab = ttk.Frame(notebook, padding=10)
        convention_tab = ttk.Frame(notebook, padding=10)
        notebook.add(peak_tab, text="1  Peak finder")
        notebook.add(surface_tab, text="2  Reflection geometry")
        notebook.add(sim_tab, text="3  Virtual experiment")
        notebook.add(convention_tab, text="4  You convention")
        self._build_peak_tab(peak_tab)
        self._build_surface_tab(surface_tab)
        self._build_sim_tab(sim_tab)
        self._build_you_convention_tab(convention_tab)
        footer = ttk.Label(self, text="●  CALCULATION ENGINE READY     |     You 4S+2D     |     Diffcalc Core     |     xrayutilities", style="Status.TLabel", anchor="w")
        footer.pack(fill="x", side="bottom")

    def _build_you_convention_tab(self, parent) -> None:
        header = ttk.Frame(parent)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(
            header,
            text="You 4S+2D diffractometer geometry",
            font=("TkDefaultFont", 15, "bold"),
        ).pack(side="left")
        ttk.Label(
            header,
            text="H. You, J. Appl. Cryst. 32 (1999) 614–623",
            style="Muted.TLabel",
        ).pack(side="right", pady=(4, 0))

        caption = (
            "Laboratory axes and motor definitions in You's convention. "
            "Sample circles: μ, η, χ, and φ. Detector circles: δ and ν. "
            "The incident X-ray beam direction and the positive x, y, z axes are shown in the figure."
        )
        ttk.Label(
            parent,
            text=caption,
            style="Muted.TLabel",
            justify="left",
            wraplength=1100,
        ).pack(fill="x", pady=(0, 8))

        image_path = Path(__file__).resolve().parent.parent / "assets" / "you_1999_geometry.png"
        figure = Figure(figsize=(10, 6.5), dpi=100, facecolor=self.BG)
        axis = figure.add_subplot(111)
        axis.imshow(mpimg.imread(image_path))
        axis.set_axis_off()
        figure.tight_layout(pad=0.2)
        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()
        self.you_convention_figure = figure
        self.you_convention_canvas = canvas

    def _entry(self, parent, label, value, row, col=0, width=9):
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", pady=3)
        var = tk.StringVar(value=str(value))
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="ew", padx=(6, 8))
        return var

    def _build_inputs(self, parent) -> None:
        ttk.Label(parent, text="Unit cell (Å / degrees)", font=("TkDefaultFont", 11, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        self.a = self._entry(parent, "a", 5.431, 1)
        self.b = self._entry(parent, "b", 5.431, 2)
        self.c = self._entry(parent, "c", 5.431, 3)
        self.alpha = self._entry(parent, "alpha", 90, 4)
        self.beta = self._entry(parent, "beta", 90, 5)
        self.gamma = self._entry(parent, "gamma", 90, 6)
        self.wavelength = self._entry(parent, "Wavelength (Å)", 1.5406, 7)
        self.energy_kev = self._entry(parent, "Energy (keV)", f"{wavelength_to_energy_kev(1.5406):.7g}", 8)
        self.beam_input = tk.StringVar(value="wavelength")
        beam_mode = ttk.Frame(parent)
        beam_mode.grid(row=9, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Radiobutton(beam_mode, text="Use wavelength", variable=self.beam_input, value="wavelength", command=self._sync_beam_fields).pack(side="left")
        ttk.Radiobutton(beam_mode, text="Use energy", variable=self.beam_input, value="energy", command=self._sync_beam_fields).pack(side="left", padx=5)
        ttk.Separator(parent).grid(row=10, column=0, columnspan=2, sticky="ew", pady=10)
        ttk.Label(parent, text="Atoms (element, x, y, z, occ, Biso)", font=("TkDefaultFont", 11, "bold")).grid(row=11, column=0, columnspan=2, sticky="w")
        self.atom_text = tk.Text(parent, height=12, width=28, font=("TkFixedFont", 10))
        self.atom_text.configure(background="#ffffff", foreground=self.TEXT, insertbackground=self.TEXT, selectbackground="#dbeafe", selectforeground=self.TEXT, relief="flat", padx=8, pady=8, font=("TkFixedFont", 11))
        self.atom_text.grid(row=12, column=0, columnspan=2, sticky="nsew", pady=6)
        self.atom_text.insert("1.0", "Si, 0, 0, 0, 1, 0.5\nSi, .25, .25, .25, 1, 0.5\nSi, 0, .5, .5, 1, 0.5\nSi, .5, 0, .5, 1, 0.5\nSi, .5, .5, 0, 1, 0.5\nSi, .25, .75, .75, 1, 0.5\nSi, .75, .25, .75, 1, 0.5\nSi, .75, .75, .25, 1, 0.5")
        parent.rowconfigure(12, weight=1)
        ttk.Label(parent, text="Fractional coordinates; # starts a comment", style="Muted.TLabel", wraplength=210).grid(row=13, column=0, columnspan=2, sticky="w")
        ttk.Button(parent, text="Load structure (CIF/POSCAR)…", command=self.load_structure).grid(row=14, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.b_matrix_status = ttk.Label(parent, text="B matrix: —", style="Muted.TLabel", wraplength=260, justify="left")
        self.b_matrix_status.grid(row=15, column=0, columnspan=2, sticky="w", pady=(8, 0))

    def _sync_beam_fields(self) -> None:
        try:
            if self.beam_input.get() == "energy":
                self.wavelength.set(f"{energy_kev_to_wavelength(float(self.energy_kev.get())):.9g}")
            else:
                self.energy_kev.set(f"{wavelength_to_energy_kev(float(self.wavelength.get())):.9g}")
        except ValueError:
            pass

    def _beam_wavelength(self) -> float:
        if self.beam_input.get() == "energy":
            wavelength = energy_kev_to_wavelength(float(self.energy_kev.get()))
            self.wavelength.set(f"{wavelength:.9g}")
            return wavelength
        wavelength = float(self.wavelength.get())
        self.energy_kev.set(f"{wavelength_to_energy_kev(wavelength):.9g}")
        return wavelength

    def _build_peak_tab(self, parent) -> None:
        controls = ttk.Frame(parent)
        controls.pack(fill="x", pady=(0, 8))
        self.max_hkl = self._entry(controls, "Max |hkl|", 8, 0)
        self.min_tt = self._entry(controls, "2θ min", 5, 0, 2)
        self.max_tt = self._entry(controls, "2θ max", 90, 0, 4)
        self.min_i = self._entry(controls, "I min (%)", 0.5, 0, 6)
        ttk.Button(controls, text="Find peaks", style="Accent.TButton", command=self.find_peaks).grid(row=0, column=8, padx=8)
        ttk.Button(controls, text="Export CSV", command=self.export_csv).grid(row=0, column=9)
        pane = ttk.Panedwindow(parent, orient="vertical")
        pane.pack(fill="both", expand=True)
        table_frame = ttk.Frame(pane)
        pane.add(table_frame, weight=3)
        cols = ("h", "k", "l", "d", "two_theta", "q", "intensity")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", selectmode="extended")
        labels = {"h": "h", "k": "k", "l": "l", "d": "d (Å)", "two_theta": "2θ (°)", "q": "|Q| (Å⁻¹)", "intensity": "Relative I (%)"}
        for col in cols:
            self.tree.heading(col, text=labels[col], command=lambda c=col: self._sort(c))
            self.tree.column(col, width=85, anchor="center")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        dynamics = ttk.LabelFrame(pane, text="Peak intensity vs phonon coordinate", padding=8)
        pane.add(dynamics, weight=2)
        dyn_controls = ttk.Frame(dynamics)
        dyn_controls.pack(side="left", fill="y", padx=(0, 8))
        ttk.Button(dyn_controls, text="Load Phonopy band.yaml…", command=self.load_phonon).pack(fill="x")
        ttk.Button(dyn_controls, text="Load symmetry irreps.yaml…", command=self.load_irreps).pack(fill="x", pady=(3, 0))
        self.phonon_status = ttk.Label(dyn_controls, text="No phonon file loaded", wraplength=235, style="Muted.TLabel")
        self.phonon_status.pack(fill="x", pady=4)
        self.mode_choice = ttk.Combobox(dyn_controls, state="readonly", width=34)
        self.mode_choice.pack(fill="x", pady=3)
        buttons = ttk.Frame(dyn_controls)
        buttons.pack(fill="x", pady=3)
        ttk.Button(buttons, text="Add selected peaks", command=self.add_monitored_peaks).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.clear_monitored_peaks).pack(side="left", padx=4)
        self.monitored_list = tk.Listbox(dyn_controls, height=3, selectmode="extended")
        self.monitored_list.configure(background="#ffffff", foreground=self.TEXT, selectbackground="#dbeafe", selectforeground=self.TEXT, relief="flat", font=("TkDefaultFont", 11))
        self.monitored_list.pack(fill="x", pady=3)
        amp = ttk.Frame(dyn_controls)
        amp.pack(fill="x")
        self.amp_min = self._entry(amp, "Q min", -0.5, 0)
        self.amp_max = self._entry(amp, "Q max", 0.5, 0, 2)
        self.amp_points = self._entry(amp, "Points", 101, 0, 4)
        self.mode_phase = self._entry(amp, "Phase °", 0, 0, 6)
        ttk.Button(dyn_controls, text="Calculate intensity curves", style="Accent.TButton", command=self.plot_phonon_curves).pack(fill="x", pady=5)
        self.phonon_figure = Figure(figsize=(5, 2.4), dpi=100, facecolor=self.BG)
        self.phonon_ax = self.phonon_figure.add_subplot(111)
        self.phonon_canvas = FigureCanvasTkAgg(self.phonon_figure, master=dynamics)
        self.phonon_canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

    def _build_sim_tab(self, parent) -> None:
        top = ttk.LabelFrame(parent, text="Motor angles and detector", padding=8)
        top.pack(fill="x")
        self.mu = tk.StringVar(value="0")
        self.delta = tk.StringVar(value="0")
        self.nu = tk.StringVar(value="0")
        self.eta = tk.StringVar(value="0")
        self.chi = tk.StringVar(value="0")
        self.phi = tk.StringVar(value="0")
        self.komega = tk.StringVar(value="0")
        self.kappa_angle = tk.StringVar(value="0")
        self.kphi = tk.StringVar(value="0")

        ttk.Label(top, text="You Eulerian").grid(row=0, column=0, sticky="w", padx=(0, 8))
        for index, (label, variable) in enumerate((("μ", self.mu), ("δ", self.delta), ("ν", self.nu), ("η", self.eta), ("χ", self.chi), ("φ", self.phi))):
            column = 1 + 2 * index
            ttk.Label(top, text=label).grid(row=0, column=column, sticky="e")
            ttk.Entry(top, textvariable=variable, width=8).grid(row=0, column=column + 1, padx=(3, 8))
        ttk.Button(top, text="Set Eulerian", command=self.set_eulerian_orientation).grid(row=0, column=13, padx=(6, 0))

        ttk.Label(top, text="Kappa convention").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(5, 0))
        for index, (label, variable) in enumerate((("μ", self.mu), ("δ", self.delta), ("ν", self.nu), ("κω", self.komega), ("κ", self.kappa_angle), ("κφ", self.kphi))):
            column = 1 + 2 * index
            ttk.Label(top, text=label).grid(row=1, column=column, sticky="e", pady=(5, 0))
            ttk.Entry(top, textvariable=variable, width=8).grid(row=1, column=column + 1, padx=(3, 8), pady=(5, 0))
        ttk.Button(top, text="Set Kappa", command=self.set_kappa_orientation).grid(row=1, column=13, padx=(6, 0), pady=(5, 0))

        settings = ttk.Frame(top)
        settings.grid(row=2, column=0, columnspan=14, sticky="w", pady=(7, 0))
        self.distance = self._entry(settings, "Distance mm", 150, 0)
        self.pixel = self._entry(settings, "Pixel mm", 0.172, 0, 2)
        self.mosaic = self._entry(settings, "Mosaic °", 0.01, 0, 4)
        self.kappa_alpha = self._entry(settings, "κ-axis α °", 50, 0, 6)
        ttk.Button(settings, text="Simulate", style="Accent.TButton", command=self.simulate).grid(row=0, column=8, padx=10)
        self.q_hkl_status = ttk.Label(settings, text="Q (r.l.u.) = —", style="Muted.TLabel")
        self.q_hkl_status.grid(row=0, column=9, sticky="w", padx=(12, 5))
        ttk.Button(settings, text="Calculate Q", command=self.current_hkl).grid(row=0, column=10, padx=(0, 4))
        self.sim_status = ttk.Label(top, text="Detector: 512 × 512 px; beam center (256, 256)")
        self.sim_status.grid(row=3, column=13, sticky="e", padx=(8, 0), pady=(4, 0))
        self.kappa_status = ttk.Label(top, text="κ position: —", style="Muted.TLabel")
        self.kappa_status.grid(row=3, column=0, columnspan=13, sticky="w", pady=(4, 0))
        solve = ttk.LabelFrame(parent, text="Diffcalc motor calculator (active U)", padding=8)
        solve.pack(fill="x", pady=(6, 0))
        self.target_h = self._entry(solve, "h", 1, 0)
        self.target_k = self._entry(solve, "k", 1, 0, 2)
        self.target_l = self._entry(solve, "l", 1, 0, 4)
        self.sim_surface_h = self._entry(solve, "Surface h", 0, 0, 6)
        self.sim_surface_k = self._entry(solve, "k", 0, 0, 8)
        self.sim_surface_l = self._entry(solve, "l", 1, 0, 10)
        ttk.Label(solve, text="Constraints").grid(row=1, column=0, sticky="w")
        self.constraints = tk.StringVar(value="nu=0, mu=0, betain=betaout")
        ttk.Entry(solve, textvariable=self.constraints, width=38).grid(row=1, column=1, columnspan=10, sticky="ew", padx=6)
        ttk.Button(solve, text="Solve HKL → motors", command=self.solve_hkl).grid(row=0, column=12, padx=6)
        self.solution_choice = ttk.Combobox(solve, state="readonly", width=76)
        self.solution_choice.grid(row=2, column=0, columnspan=12, sticky="ew", pady=(7, 0))
        ttk.Button(solve, text="Apply solution", command=self.apply_solution).grid(row=2, column=12, padx=6, pady=(7, 0))
        ttk.Label(solve, text="Motor limits").grid(row=3, column=0, sticky="w")
        ttk.Entry(solve, textvariable=self.motor_limits_text, width=64).grid(row=3, column=1, columnspan=10, sticky="ew", padx=6, pady=(5, 0))
        ttk.Label(solve, text="blank = Diffcalc default", style="Muted.TLabel").grid(row=3, column=11, columnspan=2, sticky="w")
        ub_frame = ttk.LabelFrame(parent, text="U and UB from two general measured reflections (kappa motor positions)", padding=6)
        ub_frame.pack(fill="x", pady=(3, 0))
        ttk.Label(ub_frame, text="Ref").grid(row=0, column=0)
        for col, text_label in enumerate(("h", "k", "l", "μ", "δ", "ν", "κω", "κ", "κφ"), 1):
            ttk.Label(ub_frame, text=text_label).grid(row=0, column=col)
        self.ub_reference_vars = []
        defaults = ((1, 0, 0, 0, 20, 0, 10, 0, 0), (0, 1, 0, 0, 30, 0, 15, 0, 90))
        for row, values in enumerate(defaults, 1):
            ttk.Label(ub_frame, text=str(row)).grid(row=row, column=0)
            variables = []
            for col, value in enumerate(values, 1):
                variable = tk.StringVar(value=str(value))
                ttk.Entry(ub_frame, textvariable=variable, width=7).grid(row=row, column=col, padx=2)
                variables.append(variable)
            self.ub_reference_vars.append(variables)
        ttk.Button(ub_frame, text="Calculate U and UB", command=self.calculate_measured_ub).grid(row=1, column=10, rowspan=2, padx=7)
        self.ub_status = ttk.Label(ub_frame, text="U = identity", wraplength=1050, style="Muted.TLabel")
        self.ub_status.grid(row=3, column=0, columnspan=11, sticky="w", pady=(5, 0))
        view_pane = ttk.Panedwindow(parent, orient="horizontal")
        view_pane.pack(fill="both", expand=True, pady=(3, 0))
        detector_frame = ttk.Frame(view_pane)
        virtual_scene_frame = ttk.Frame(view_pane)
        view_pane.add(detector_frame, weight=1)
        view_pane.add(virtual_scene_frame, weight=1)
        ttk.Label(detector_frame, text="Detector coordinates: +x (u) → right    •    +y (v) ↓ down", style="Muted.TLabel").pack(fill="x")
        self.figure = Figure(figsize=(8, 7), dpi=100, facecolor=self.BG)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=detector_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        virtual_camera = ttk.Frame(virtual_scene_frame)
        virtual_camera.pack(fill="x")
        ttk.Button(virtual_camera, text="Side view", command=self.set_virtual_you_camera).pack(side="left")
        ttk.Button(virtual_camera, text="X-ray incidence view", command=self.set_virtual_incidence_camera).pack(side="left", padx=4)
        self.virtual_geometry_figure = Figure(figsize=(7, 7), dpi=100, facecolor=self.BG)
        self.virtual_geometry_ax = self.virtual_geometry_figure.add_subplot(111, projection="3d")
        self.virtual_geometry_canvas = FigureCanvasTkAgg(self.virtual_geometry_figure, master=virtual_scene_frame)
        self.virtual_geometry_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.virtual_geometry_canvas.mpl_connect("scroll_event", lambda event: self._zoom_3d(event, self.virtual_geometry_ax, self.virtual_geometry_canvas))
        self.set_virtual_you_camera(draw=False)
        self.simulate()

    def _build_surface_tab(self, parent) -> None:
        controls = ttk.LabelFrame(parent, text="Constrained reflection calculation", padding=8)
        controls.pack(fill="x")
        ttk.Label(controls, text="betain = incidence angle    •    betaout = exit angle", style="Muted.TLabel").grid(row=0, column=7, sticky="w", padx=8)
        self.ab_h = self._entry(controls, "Peak h", 1, 0)
        self.ab_k = self._entry(controls, "k", 1, 0, 2)
        self.ab_l = self._entry(controls, "l", 1, 0, 4)
        ttk.Button(controls, text="Use selected peak", command=self.use_selected_peak_for_surface).grid(row=0, column=6, padx=5)
        self.ab_surface_h = self._entry(controls, "Surface h", 0, 1)
        self.ab_surface_k = self._entry(controls, "k", 0, 1, 2)
        self.ab_surface_l = self._entry(controls, "l", 1, 1, 4)
        ttk.Label(controls, text="Scan angle").grid(row=2, column=0, sticky="w")
        self.ab_scan_name = tk.StringVar(value="betain")
        ttk.Combobox(controls, textvariable=self.ab_scan_name, values=SCANNABLE_ANGLES, state="readonly", width=10).grid(row=2, column=1, sticky="w", padx=6)
        self.ab_alpha_min = self._entry(controls, "Min °", 0.1, 2, 2)
        self.ab_alpha_max = self._entry(controls, "Max °", 20, 2, 4)
        self.ab_points = self._entry(controls, "Points", 81, 2, 7)
        ttk.Label(controls, text="Plot angle").grid(row=1, column=7, sticky="w")
        self.ab_plot_name = tk.StringVar(value="betaout")
        plot_angles = ("betain", "betaout", "psi", "qaz", "naz", "theta", "ttheta", "mu", "delta", "nu", "eta", "chi", "phi")
        ttk.Combobox(controls, textvariable=self.ab_plot_name, values=plot_angles, state="readonly", width=10).grid(row=1, column=8, sticky="w", padx=6)
        ttk.Label(controls, text="Two fixed constraints").grid(row=3, column=0, sticky="w")
        self.ab_constraints = tk.StringVar(value="nu=0, mu=0")
        ttk.Entry(controls, textvariable=self.ab_constraints, width=30).grid(row=3, column=1, columnspan=4, sticky="ew", padx=6)
        ttk.Button(controls, text="Solve betain = betaout", command=self.solve_equal_betain_betaout).grid(row=1, column=6, padx=5)
        ttk.Button(controls, text="Run angle scan", style="Accent.TButton", command=self.run_surface_scan).grid(row=2, column=6, rowspan=2, padx=5)
        ttk.Label(controls, text="Motor limits").grid(row=4, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.motor_limits_text, width=72).grid(row=4, column=1, columnspan=6, sticky="ew", padx=6, pady=(4, 0))
        mount = ttk.LabelFrame(parent, text="U matrix from surface mounting", padding=7)
        mount.pack(fill="x", pady=(7, 0))
        self.surface_inplane_angle = self._entry(mount, "In-plane rotation °", 0, 0)
        ttk.Button(mount, text="Update U from surface", command=self.update_u_from_surface_panel).grid(row=0, column=2, padx=6)
        self.surface_u_status = ttk.Label(mount, text="U = identity", style="Muted.TLabel", wraplength=800)
        self.surface_u_status.grid(row=0, column=3, sticky="w", padx=8)
        pane = ttk.Panedwindow(parent, orient="vertical")
        pane.pack(fill="both", expand=True, pady=(8, 0))
        plot_frame = ttk.Frame(pane)
        pane.add(plot_frame, weight=3)
        visual_pane = ttk.Panedwindow(plot_frame, orient="horizontal")
        visual_pane.pack(fill="both", expand=True)
        curve_frame = ttk.Frame(visual_pane)
        scene_frame = ttk.Frame(visual_pane)
        visual_pane.add(curve_frame, weight=2)
        visual_pane.add(scene_frame, weight=3)
        self.ab_figure = Figure(figsize=(7, 3), dpi=100, facecolor=self.BG)
        self.ab_ax = self.ab_figure.add_subplot(111)
        self.ab_canvas = FigureCanvasTkAgg(self.ab_figure, master=curve_frame)
        self.ab_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.geometry_figure = Figure(figsize=(7, 4), dpi=100, facecolor=self.BG)
        self.geometry_ax = self.geometry_figure.add_subplot(111, projection="3d")
        camera_buttons = ttk.Frame(scene_frame)
        camera_buttons.pack(fill="x")
        ttk.Button(camera_buttons, text="Side view", command=self.set_you_camera).pack(side="left")
        ttk.Button(camera_buttons, text="X-ray incidence view", command=self.set_incidence_camera).pack(side="left", padx=4)
        self.geometry_canvas = FigureCanvasTkAgg(self.geometry_figure, master=scene_frame)
        self.geometry_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.geometry_canvas.mpl_connect("scroll_event", lambda event: self._zoom_3d(event, self.geometry_ax, self.geometry_canvas))
        initial_ticks = np.linspace(-1.5, 1.5, 5)
        self.geometry_ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5))
        self.geometry_ax.set_xticks(initial_ticks)
        self.geometry_ax.set_xticklabels([])
        self.geometry_ax.set_yticks(initial_ticks)
        self.geometry_ax.set_yticklabels([])
        self.geometry_ax.set_zticks(initial_ticks)
        self.geometry_ax.set_zticklabels([])
        self.geometry_ax.grid(True)
        self.set_you_camera(draw=False)
        table_frame = ttk.Frame(pane)
        pane.add(table_frame, weight=2)
        columns = ("scan", "plot", "betain", "betaout", "mu", "delta", "nu", "eta", "chi", "phi")
        self.ab_tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        self.ab_tree.bind("<<TreeviewSelect>>", self.update_surface_3d)
        for column in columns:
            heading = {"scan": "scan (°)", "plot": "plot (°)", "betain": "betain incidence (°)", "betaout": "betaout exit (°)"}.get(column, column)
            self.ab_tree.heading(column, text=heading)
            self.ab_tree.column(column, width=82, anchor="center")
        self.ab_tree.pack(side="left", fill="both", expand=True)
        ab_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.ab_tree.yview)
        self.ab_tree.configure(yscrollcommand=ab_scroll.set)
        ab_scroll.pack(side="left", fill="y")
        ttk.Button(table_frame, text="Apply selected motors to experiment", command=self.apply_surface_solution).pack(side="left", padx=8)

    def use_selected_peak_for_surface(self) -> None:
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Select a peak", "Select one reflection in the peak finder first.")
            return
        values = self.tree.item(selection[0], "values")
        for variable, value in zip((self.ab_h, self.ab_k, self.ab_l), values[:3]):
            variable.set(value)
        self.notebook.select(1)

    def run_surface_scan(self) -> None:
        try:
            cell, _, wavelength = self._model()
            hkl = tuple(float(v.get()) for v in (self.ab_h, self.ab_k, self.ab_l))
            surface = tuple(float(v.get()) for v in (self.ab_surface_h, self.ab_surface_k, self.ab_surface_l))
            if np.linalg.norm(hkl) == 0 or np.linalg.norm(surface) == 0:
                raise ValueError("Peak HKL and surface HKL must both be nonzero.")
            count = int(self.ab_points.get())
            if not 2 <= count <= 1001:
                raise ValueError("Points must be between 2 and 1001.")
            constraints = parse_constraints(self.ab_constraints.get())
            scan_name = self.ab_scan_name.get()
            plot_name = self.ab_plot_name.get()
            scan_values = np.linspace(float(self.ab_alpha_min.get()), float(self.ab_alpha_max.get()), count)
            self.surface_scan_results = scan_reflection_geometry(cell, hkl, wavelength, scan_name, scan_values, constraints, surface, parse_motor_limits(self.motor_limits_text.get()), self.u_matrix)
            self.ab_tree.delete(*self.ab_tree.get_children())
            valid_results = []
            for requested, solution in self.surface_scan_results:
                try:
                    plotted = solution_angle(solution, plot_name)
                    betain = solution_angle(solution, "betain")
                    betaout = solution_angle(solution, "betaout")
                except ValueError:
                    continue
                valid_results.append((requested, solution))
                self.ab_tree.insert("", "end", values=(f"{requested:.4f}", f"{plotted:.4f}", f"{betain:.4f}", f"{betaout:.4f}", *[f"{v:.4f}" for v in (solution.mu, solution.delta, solution.nu, solution.eta, solution.chi, solution.phi)]))
            self.surface_scan_results = valid_results
            self.ab_ax.clear()
            if self.surface_scan_results:
                scanned = [row[0] for row in self.surface_scan_results]
                plotted = [solution_angle(row[1], plot_name) for row in self.surface_scan_results]
                self.ab_ax.scatter(scanned, plotted, s=13, color="#2563eb", label="reachable sectors")
            self.ab_ax.set(xlabel=f"Scanned {scan_name} (°)", ylabel=f"{plot_name} (°)", title=f"Peak {hkl}, surface normal {surface}")
            self.ab_ax.legend(fontsize=8)
            self.ab_figure.tight_layout()
            self.ab_canvas.draw_idle()
            if not self.surface_scan_results:
                messagebox.showinfo("Angle scan", "No reachable sectors were found for these constraints and angle range.")
            else:
                first = self.ab_tree.get_children()[0]
                self.ab_tree.selection_set(first)
                self.ab_tree.focus(first)
                self.update_surface_3d()
        except Exception as exc:
            messagebox.showerror("Cannot scan reflection geometry", str(exc))

    def solve_equal_betain_betaout(self) -> None:
        try:
            cell, _, wavelength = self._model()
            hkl = tuple(float(v.get()) for v in (self.ab_h, self.ab_k, self.ab_l))
            surface = tuple(float(v.get()) for v in (self.ab_surface_h, self.ab_surface_k, self.ab_surface_l))
            if np.linalg.norm(hkl) == 0 or np.linalg.norm(surface) == 0:
                raise ValueError("Peak HKL and surface HKL must both be nonzero.")
            constraints = parse_constraints(self.ab_constraints.get())
            if len(constraints) != 2:
                raise ValueError("Enter exactly two fixed constraints; betain=betaout is the third constraint.")
            solutions = solve_symmetric_geometry(
                cell,
                hkl,
                wavelength,
                constraints,
                surface,
                parse_motor_limits(self.motor_limits_text.get()),
                self.u_matrix,
            )
            solutions = [solution for solution in solutions if np.isfinite(solution.virtual.get("betain", np.nan)) and np.isfinite(solution.virtual.get("betaout", np.nan))]
            self.surface_scan_results = sorted(((solution.virtual["betain"], solution) for solution in solutions), key=lambda row: row[0])
            self.ab_tree.delete(*self.ab_tree.get_children())
            for betain, solution in self.surface_scan_results:
                self.ab_tree.insert("", "end", values=(f"{betain:.4f}", f"{solution.virtual['betaout']:.4f}", f"{betain:.4f}", f"{solution.virtual['betaout']:.4f}", *[f"{v:.4f}" for v in (solution.mu, solution.delta, solution.nu, solution.eta, solution.chi, solution.phi)]))
            self.ab_ax.clear()
            if self.surface_scan_results:
                betains = [row[0] for row in self.surface_scan_results]
                betaouts = [row[1].virtual["betaout"] for row in self.surface_scan_results]
                self.ab_ax.scatter(betains, betaouts, s=38, color="#2563eb", label="betain = betaout solutions")
                lower = min(betains + betaouts) - 1
                upper = max(betains + betaouts) + 1
                self.ab_ax.plot([lower, upper], [lower, upper], "--", color="#64748b", linewidth=1, label="betain = betaout")
                first = self.ab_tree.get_children()[0]
                self.ab_tree.selection_set(first)
                self.ab_tree.focus(first)
                self.update_surface_3d()
            self.ab_ax.set(xlabel="Incidence betain (°)", ylabel="Exit betaout (°)", title=f"Symmetric geometry: peak {hkl}, surface {surface}")
            self.ab_ax.legend(fontsize=9)
            self.ab_figure.tight_layout()
            self.ab_canvas.draw_idle()
            if not self.surface_scan_results:
                messagebox.showinfo("betain = betaout", "No symmetric-geometry solution satisfies the constraints and motor limits.")
        except Exception as exc:
            messagebox.showerror("Cannot solve betain = betaout", str(exc))

    def update_surface_3d(self, _event=None) -> None:
        selection = self.ab_tree.selection()
        if not selection or not self.surface_scan_results:
            return
        index = self.ab_tree.index(selection[0])
        _, solution = self.surface_scan_results[index]
        cell, _, _ = self._model()
        surface_hkl = np.array([float(v.get()) for v in (self.ab_surface_h, self.ab_surface_k, self.ab_surface_l)])
        sample_rotation = you_sample_rotation(solution.mu, solution.eta, solution.chi, solution.phi)
        surface_phi = self.u_matrix @ diffcalc_reciprocal_matrix(cell) @ surface_hkl
        surface_lab = sample_rotation @ (surface_phi / np.linalg.norm(surface_phi))
        outgoing, detector_u, detector_v = you_detector_frame(solution.delta, solution.nu)
        incident = np.array([0.0, 1.0, 0.0])
        scattering = outgoing - incident
        ax = self.geometry_ax
        camera = (ax.elev, ax.azim, getattr(ax, "roll", 0.0))
        saved_limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()) if getattr(ax, "_xpeak_has_scene", False) else None
        ax.clear()

        # Display coordinates are (-lab y, -lab z, lab x). This is a proper
        # rotation (determinant +1), so right-handed laboratory and a-b-c
        # frames remain right-handed while physical lab x is displayed upward.
        def display(vector):
            vector = np.asarray(vector)
            mapped = vector[..., [1, 2, 0]].copy()
            mapped[..., 0] *= -1
            mapped[..., 1] *= -1
            return mapped

        def arrow(start, vector, color, label, length=1.0):
            vector = np.asarray(vector, dtype=float)
            vector = length * vector / max(np.linalg.norm(vector), 1e-12)
            start_display = display(start)
            vector_display = display(vector)
            ax.quiver(*start_display, *vector_display, color=color, linewidth=2, arrow_length_ratio=0.12)
            endpoint = np.asarray(start) + vector
            if label:
                ax.text(*display(endpoint), label, color=color, fontsize=8)

        arrow(np.array([0.0, -1.35, 0.0]), incident, "#2563eb", "", 1.35)
        ax.text(*display(np.array([0.0, -0.95, 0.0])), "kᵢ", color="#2563eb", fontsize=8)
        arrow(np.zeros(3), outgoing, "#dc2626", "k_f", 1.35)
        # The surface normal follows the positive user-specified (hkl): for an
        # orthogonal cell, (001) therefore points along +c rather than -c.
        arrow(np.zeros(3), surface_lab, "#16a34a", "surface n", 0.9)
        arrow(np.zeros(3), scattering, "#9333ea", "Q", 0.85)
        for lab_axis, label in (([1, 0, 0], "+x"), ([0, 1, 0], "+y"), ([0, 0, 1], "+z")):
            arrow(np.zeros(3), lab_axis, "#64748b", label, 1.38)

        helper = np.array([1.0, 0.0, 0.0]) if abs(surface_lab[0]) < 0.85 else np.array([0.0, 0.0, 1.0])
        plane_u = np.cross(surface_lab, helper)
        plane_u /= np.linalg.norm(plane_u)
        plane_v = np.cross(surface_lab, plane_u)
        grid = np.linspace(-0.65, 0.65, 2)
        uu, vv = np.meshgrid(grid, grid)
        sample_plane = uu[..., None] * plane_u + vv[..., None] * plane_v
        sample_plane = display(sample_plane)
        ax.plot_surface(sample_plane[..., 0], sample_plane[..., 1], sample_plane[..., 2], color="#22c55e", alpha=0.22, shade=False)

        reciprocal_phi = self.u_matrix @ diffcalc_reciprocal_matrix(cell)
        direct_phi = np.linalg.inv(reciprocal_phi.T)
        crystal_axes = sample_rotation @ direct_phi
        for axis_index, (label, color) in enumerate((("a", "#0891b2"), ("b", "#db2777"), ("c", "#ca8a04"))):
            arrow(np.zeros(3), crystal_axes[:, axis_index], color, label, 0.72)

        detector_center = 1.35 * outgoing
        size = np.linspace(-0.38, 0.38, 2)
        du, dv = np.meshgrid(size, size)
        detector_plane = detector_center + du[..., None] * detector_u + dv[..., None] * detector_v
        detector_plane = display(detector_plane)
        ax.plot_surface(detector_plane[..., 0], detector_plane[..., 1], detector_plane[..., 2], color="#f59e0b", alpha=0.35, shade=False)
        detector_label = detector_center + 0.34 * detector_u + 0.34 * detector_v
        ax.text(*display(detector_label), "detector", fontsize=8)
        ax.scatter([0], [0], [0], color="#111827", s=18)
        if saved_limits:
            ax.set(xlim=saved_limits[0], ylim=saved_limits[1], zlim=saved_limits[2])
        else:
            ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5))
        ticks = np.linspace(-1.5, 1.5, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([])
        ax.set_yticks(ticks)
        ax.set_yticklabels([])
        ax.set_zticks(ticks)
        ax.set_zticklabels([])
        ax.grid(True)
        ax._xpeak_has_scene = True
        ax.set_title(f"betain={solution.virtual['betain']:.2f}°, betaout={solution.virtual['betaout']:.2f}°", fontsize=10)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        self.geometry_figure.tight_layout()
        self.geometry_canvas.draw_idle()

    def set_you_camera(self, draw=True) -> None:
        """Camera along -lab z, with lab x vertical as in You's schematic."""
        self.geometry_ax.view_init(elev=12, azim=-90, roll=0)
        if draw:
            self.geometry_canvas.draw_idle()

    def set_incidence_camera(self, draw=True) -> None:
        """Look downstream along +lab y from the incoming-beam side."""
        self.geometry_ax.view_init(elev=0, azim=0, roll=0)
        if draw:
            self.geometry_canvas.draw_idle()

    def _zoom_3d(self, event, ax, canvas) -> None:
        factor = 0.82 if event.button == "up" else 1.22
        for getter, setter in ((ax.get_xlim3d, ax.set_xlim3d), (ax.get_ylim3d, ax.set_ylim3d), (ax.get_zlim3d, ax.set_zlim3d)):
            lower, upper = getter()
            center = (lower + upper) / 2.0
            half = max((upper - lower) * factor / 2.0, 0.08)
            setter(center - half, center + half)
        canvas.draw_idle()

    def apply_surface_solution(self) -> None:
        selection = self.ab_tree.selection()
        if not selection:
            return
        index = self.ab_tree.index(selection[0])
        solution = self.surface_scan_results[index][1]
        try:
            self._set_kappa_from_you(solution.sample_motors)
        except ValueError as exc:
            messagebox.showerror("Kappa position unavailable", str(exc))
            return
        for variable, value in zip((self.delta, self.nu), solution.detector_motors):
            variable.set(f"{value:.8g}")
        self.notebook.select(2)
        self.simulate()

    def solve_hkl(self) -> None:
        try:
            cell, _, wavelength = self._model()
            hkl = tuple(float(v.get()) for v in (self.target_h, self.target_k, self.target_l))
            surface_hkl = tuple(float(v.get()) for v in (self.sim_surface_h, self.sim_surface_k, self.sim_surface_l))
            self.motor_solutions = hkl_to_motor_solutions(cell, hkl, wavelength, parse_constraints(self.constraints.get()), surface_hkl=surface_hkl, motor_limits=parse_motor_limits(self.motor_limits_text.get()), u_matrix=self.u_matrix)
            labels = [
                f"{i}: μ {solution.mu:.3f}  δ {solution.delta:.3f}  ν {solution.nu:.3f}  "
                f"η {solution.eta:.3f}  χ {solution.chi:.3f}  φ {solution.phi:.3f}"
                for i, solution in enumerate(self.motor_solutions, 1)
            ]
            self.solution_choice["values"] = labels
            if labels:
                self.solution_choice.current(0)
            self.sim_status.config(text=f"Diffcalc found {len(labels)} You Eulerian motor solution(s)")
        except Exception as exc:
            messagebox.showerror("Diffcalc could not solve HKL", str(exc))

    def apply_solution(self) -> None:
        index = self.solution_choice.current()
        if not 0 <= index < len(self.motor_solutions):
            return
        solution = self.motor_solutions[index]
        try:
            self._set_kappa_from_you(solution.sample_motors)
        except ValueError as exc:
            messagebox.showerror("Kappa position unavailable", str(exc))
            return
        for variable, value in zip((self.delta, self.nu), solution.detector_motors):
            variable.set(f"{value:.8g}")
        self.simulate()

    def current_hkl(self) -> None:
        try:
            cell, _, wavelength = self._model()
            sample = self._sample_from_kappa_input()
            motors = (sample[0], float(self.delta.get()), float(self.nu.get()), *sample[1:])
            hkl = motors_to_hkl(cell, motors, wavelength, self.u_matrix)
            self.q_hkl_status.config(text=f"Q (r.l.u.) = ({hkl[0]:.5f}, {hkl[1]:.5f}, {hkl[2]:.5f})")
        except Exception as exc:
            messagebox.showerror("Diffcalc could not calculate HKL", str(exc))

    def _model(self):
        cell = UnitCell(*[float(v.get()) for v in (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)])
        atoms = []
        for number, line in enumerate(self.atom_text.get("1.0", "end").splitlines(), 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            fields = [x.strip() for x in line.split(",")]
            if len(fields) < 4:
                raise ValueError(f"Atom line {number}: expected element, x, y, z[, occupancy, Biso]")
            atoms.append(Atom(fields[0], *map(float, fields[1:4]), float(fields[4]) if len(fields) > 4 else 1.0, float(fields[5]) if len(fields) > 5 else 0.0))
        if not atoms:
            raise ValueError("Enter at least one atom.")
        return cell, atoms, self._beam_wavelength()

    def find_peaks(self) -> None:
        try:
            cell, atoms, wavelength = self._model()
            self._update_b_display(cell)
            self.peaks = enumerate_peaks(cell, atoms, wavelength, int(self.max_hkl.get()), float(self.min_tt.get()), float(self.max_tt.get()), float(self.min_i.get()))
            self.tree.delete(*self.tree.get_children())
            for p in self.peaks:
                self.tree.insert("", "end", values=(p.h, p.k, p.l, f"{p.d:.5f}", f"{p.two_theta:.3f}", f"{p.q:.4f}", f"{p.intensity:.4f}"))
        except Exception as exc:
            messagebox.showerror("Cannot find peaks", str(exc))

    @staticmethod
    def _matrix_text(name: str, matrix: np.ndarray) -> str:
        rows = ["[" + "  ".join(f"{value: .5f}" for value in row) + "]" for row in np.asarray(matrix)]
        return f"{name}:\n" + "\n".join(rows)

    @staticmethod
    def _matrix_inline(name: str, matrix: np.ndarray) -> str:
        rows = ["[" + " ".join(f"{value:.4f}" for value in row) + "]" for row in np.asarray(matrix)]
        return f"{name}: " + "  ".join(rows)

    def _update_b_display(self, cell: UnitCell) -> None:
        b_matrix = 2.0 * np.pi * diffcalc_reciprocal_matrix(cell)
        self.b_matrix_status.config(text=self._matrix_text("B (Å⁻¹)", b_matrix))

    def _update_u_displays(self, source: str) -> None:
        text = self._matrix_inline(f"U — {source}", self.u_matrix)
        if hasattr(self, "surface_u_status"):
            self.surface_u_status.config(text=text)
        if hasattr(self, "ub_status"):
            self.ub_status.config(text=text)

    def _set_surface_orientation(self, surface_hkl, angle: float, source: str) -> None:
        cell, _, _ = self._model()
        self.u_matrix = orientation_from_surface(cell, surface_hkl, angle)
        self.ub_matrix = self.u_matrix @ (2.0 * np.pi * diffcalc_reciprocal_matrix(cell))
        self._update_u_displays(source)

    def update_u_from_surface_panel(self) -> None:
        try:
            surface = tuple(float(v.get()) for v in (self.ab_surface_h, self.ab_surface_k, self.ab_surface_l))
            angle = float(self.surface_inplane_angle.get())
            self._set_surface_orientation(surface, angle, f"surface {surface}, in-plane {angle:g}°")
            for variable, value in zip((self.sim_surface_h, self.sim_surface_k, self.sim_surface_l), surface):
                variable.set(f"{value:g}")
            if self.surface_scan_results:
                self.update_surface_3d()
        except Exception as exc:
            messagebox.showerror("Cannot update U", str(exc))

    def load_structure(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Crystal structures", "*.cif *.vasp *.poscar"), ("VASP POSCAR/CONTCAR", "POSCAR CONTCAR"), ("All files", "*")])
        if not path:
            return
        try:
            from pymatgen.core import Structure

            structure = Structure.from_file(path)
            lattice = structure.lattice
            values = (lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma)
            for variable, value in zip((self.a, self.b, self.c, self.alpha, self.beta, self.gamma), values):
                variable.set(f"{value:.8g}")
            lines = []
            for site in structure:
                x, y, z = site.frac_coords
                for species, occupancy in site.species.items():
                    symbol = getattr(species, "symbol", str(species))
                    lines.append(f"{symbol}, {x:.10g}, {y:.10g}, {z:.10g}, {float(occupancy):.8g}, 0")
            self.atom_text.delete("1.0", "end")
            self.atom_text.insert("1.0", "\n".join(lines))
            self.title(f"XPeak — {path.rsplit('/', 1)[-1]}")
            self.find_peaks()
        except ImportError:
            messagebox.showerror("Cannot load structure", "Structure import requires pymatgen. Install it with: pip install pymatgen")
        except Exception as exc:
            messagebox.showerror("Cannot load structure", str(exc))

    def load_phonon(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Phonopy YAML", "*.yaml *.yml"), ("All files", "*")])
        if not path:
            return
        try:
            dataset = load_phonopy_yaml(path)
            self.phonon_dataset = dataset
            cell = dataset.cell
            for variable, value in zip((self.a, self.b, self.c, self.alpha, self.beta, self.gamma), (cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma)):
                variable.set(f"{value:.9g}")
            self.atom_text.delete("1.0", "end")
            self.atom_text.insert("1.0", "\n".join(f"{a.element}, {a.x:.10g}, {a.y:.10g}, {a.z:.10g}, 1, 0" for a in dataset.atoms))
            self._refresh_mode_labels()
            labels = self.mode_choice["values"]
            if labels:
                self.mode_choice.current(0)
            self.phonon_status.config(text=f"{path.rsplit('/', 1)[-1]}: {len(dataset.atoms)} atoms, {len(dataset.modes)} modes")
            self.find_peaks()
        except Exception as exc:
            messagebox.showerror("Cannot load phonon YAML", str(exc))

    def _refresh_mode_labels(self) -> None:
        if self.phonon_dataset is None:
            self.mode_choice["values"] = []
            return
        labels = []
        for mode in self.phonon_dataset.modes:
            symmetry = f" [{mode.ir_label}]" if mode.ir_label else ""
            labels.append(f"q{mode.q_index + 1} {mode.q} — mode {mode.mode_index + 1}{symmetry}: {mode.frequency_thz:.5g} THz")
        current = self.mode_choice.current()
        self.mode_choice["values"] = labels
        if labels:
            self.mode_choice.current(max(0, min(current, len(labels) - 1)))

    def load_irreps(self) -> None:
        if self.phonon_dataset is None:
            messagebox.showinfo("Load phonons first", "Load the matching band.yaml before assigning mode symmetries.")
            return
        path = filedialog.askopenfilename(filetypes=[("Phonopy irreps YAML", "*.yaml *.yml"), ("All files", "*")])
        if not path:
            return
        try:
            self.phonon_dataset, matched, mismatches = assign_irreps(self.phonon_dataset, path)
            self._refresh_mode_labels()
            self.phonon_status.config(text=f"Point group {self.phonon_dataset.point_group}: symmetry assigned to {matched} mode entries" + (f"; {len(mismatches)} frequency mismatches" if mismatches else ""))
            if mismatches:
                messagebox.showwarning("Some modes were not assigned", f"{len(mismatches)} band entries differed in frequency by more than 0.0001 THz.")
        except Exception as exc:
            messagebox.showerror("Cannot load irreps YAML", str(exc))

    def add_monitored_peaks(self) -> None:
        for item in self.tree.selection():
            values = self.tree.item(item, "values")
            hkl = tuple(int(values[i]) for i in range(3))
            if hkl not in self.monitored_hkls:
                self.monitored_hkls.append(hkl)
                self.monitored_list.insert("end", f"({hkl[0]} {hkl[1]} {hkl[2]})")

    def clear_monitored_peaks(self) -> None:
        self.monitored_hkls.clear()
        self.monitored_list.delete(0, "end")

    def plot_phonon_curves(self) -> None:
        try:
            if self.phonon_dataset is None:
                raise ValueError("Load a Phonopy band.yaml file first.")
            if not self.monitored_hkls:
                raise ValueError("Select reflections in the peak table and click Add selected peaks.")
            mode_index = self.mode_choice.current()
            if mode_index < 0:
                raise ValueError("Select a phonon mode.")
            count = int(self.amp_points.get())
            if not 3 <= count <= 2001:
                raise ValueError("Points must be between 3 and 2001.")
            amplitudes = np.linspace(float(self.amp_min.get()), float(self.amp_max.get()), count)
            wavelength = self._beam_wavelength()
            mode = self.phonon_dataset.modes[mode_index]
            curves = mode_intensity_curves(self.phonon_dataset, mode, self.monitored_hkls, amplitudes, wavelength, float(self.mode_phase.get()))
            self.phonon_ax.clear()
            zero_index = int(np.argmin(np.abs(amplitudes)))
            for hkl, intensity in curves.items():
                reference = intensity[zero_index]
                if reference > max(float(np.max(intensity)), 1.0) * 1e-12:
                    plotted = 100.0 * (intensity / reference - 1.0)
                    label = f"({hkl[0]} {hkl[1]} {hkl[2]})"
                else:
                    scale = max(float(np.max(intensity)), 1e-30)
                    plotted = 100.0 * intensity / scale
                    label = f"({hkl[0]} {hkl[1]} {hkl[2]}) forbidden: % max"
                self.phonon_ax.plot(amplitudes, plotted, label=label)
            self.phonon_ax.axvline(0, color="#94a3b8", linewidth=0.8)
            symmetry = f" [{mode.ir_label}]" if mode.ir_label else ""
            self.phonon_ax.set(xlabel="Normal coordinate Q (Å√amu)", ylabel="ΔI/I₀ (%)", title=f"Mode {mode.mode_index + 1}{symmetry}, {mode.frequency_thz:.5g} THz, q={mode.q}")
            self.phonon_ax.legend(fontsize=7)
            self.phonon_figure.tight_layout()
            self.phonon_canvas.draw_idle()
        except Exception as exc:
            messagebox.showerror("Cannot calculate phonon response", str(exc))

    def calculate_measured_ub(self) -> None:
        try:
            cell, _, wavelength = self._model()
            references = []
            for variables in self.ub_reference_vars:
                values = [float(variable.get()) for variable in variables]
                hkl = tuple(values[:3])
                mu, delta, nu, komega, kappa, kphi = values[3:]
                if not 0.0 <= komega <= 360.0 or not 0.0 <= kappa <= 180.0 or not 0.0 <= kphi <= 360.0:
                    raise ValueError("Reference κω and κφ must be 0–360°; κ must be 0–180°.")
                you = kappa_to_you(komega, kappa, kphi, float(self.kappa_alpha.get()))
                references.append((hkl, (mu, delta, nu, you.eta, you.chi, you.phi)))
            self.u_matrix, self.ub_matrix = calculate_ub_from_two_reflections(cell, wavelength, references)
            self._update_u_displays("two measured reflections")
            self.ub_status.config(text=self._matrix_inline("U — two measured reflections", self.u_matrix) + "    " + self._matrix_inline("UB (Å⁻¹)", self.ub_matrix))
            self.sim_status.config(text="Measured UB active for motor and detector calculations")
            self.simulate()
        except Exception as exc:
            messagebox.showerror("Cannot calculate UB", str(exc))

    def update_virtual_3d(self, sample, detector) -> None:
        ax = self.virtual_geometry_ax
        camera = (ax.elev, ax.azim, getattr(ax, "roll", 0.0))
        saved_limits = (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()) if getattr(ax, "_xpeak_has_scene", False) else None
        ax.clear()
        rotation = you_sample_rotation(*sample) @ self.u_matrix
        outgoing, detector_u, detector_v = you_detector_frame(*detector)

        def display(vector):
            vector = np.asarray(vector)
            mapped = vector[..., [1, 2, 0]].copy()
            mapped[..., 0] *= -1
            mapped[..., 1] *= -1
            return mapped

        def arrow(start, vector, color, label, length):
            vector = length * np.asarray(vector, dtype=float) / max(np.linalg.norm(vector), 1e-12)
            ax.quiver(*display(start), *display(vector), color=color, linewidth=2.5, arrow_length_ratio=0.13)
            if label:
                ax.text(*display(np.asarray(start) + vector), label, color=color, fontsize=9)

        incident = np.array([0.0, 1.0, 0.0])
        arrow(np.array([0.0, -1.4, 0.0]), incident, "#2563eb", "", 1.4)
        ax.text(*display(np.array([0.0, -0.98, 0.0])), "kᵢ", color="#2563eb", fontsize=9)
        arrow(np.zeros(3), outgoing, "#dc2626", "k_f", 1.4)
        arrow(np.zeros(3), outgoing - incident, "#9333ea", "Q", 0.85)
        for lab_axis, label in (([1, 0, 0], "+x"), ([0, 1, 0], "+y"), ([0, 0, 1], "+z")):
            arrow(np.zeros(3), lab_axis, "#64748b", label, 1.42)
        cell, _, _ = self._model()
        direct_phi = np.linalg.inv(diffcalc_reciprocal_matrix(cell).T)
        crystal_axes = rotation @ direct_phi
        for axis_index, (label, color) in enumerate((("a", "#0891b2"), ("b", "#db2777"), ("c", "#ca8a04"))):
            arrow(np.zeros(3), crystal_axes[:, axis_index], color, label, 0.75)
        a_axis = crystal_axes[:, 0] / np.linalg.norm(crystal_axes[:, 0])
        b_axis = crystal_axes[:, 1] / np.linalg.norm(crystal_axes[:, 1])
        grid = np.linspace(-0.6, 0.6, 2)
        aa, bb = np.meshgrid(grid, grid)
        crystal_plane = display(aa[..., None] * a_axis + bb[..., None] * b_axis)
        ax.plot_surface(crystal_plane[..., 0], crystal_plane[..., 1], crystal_plane[..., 2], color="#22c55e", alpha=0.22, shade=False)
        center = 1.4 * outgoing
        du, dv = np.meshgrid(np.linspace(-0.4, 0.4, 2), np.linspace(-0.4, 0.4, 2))
        detector_plane = display(center + du[..., None] * detector_u + dv[..., None] * detector_v)
        ax.plot_surface(detector_plane[..., 0], detector_plane[..., 1], detector_plane[..., 2], color="#f59e0b", alpha=0.4, shade=False)
        arrow(center, detector_u, "#06b6d4", "+detector x / u", 0.58)
        arrow(center, detector_v, "#22c55e", "+detector y / v", 0.58)
        detector_label = center - 0.34 * detector_u - 0.34 * detector_v
        ax.text(*display(detector_label), "detector", fontsize=9)
        if saved_limits:
            ax.set(xlim=saved_limits[0], ylim=saved_limits[1], zlim=saved_limits[2])
        else:
            ax.set(xlim=(-1.55, 1.55), ylim=(-1.55, 1.55), zlim=(-1.55, 1.55))
        ticks = np.linspace(-1.5, 1.5, 5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([])
        ax.set_yticks(ticks)
        ax.set_yticklabels([])
        ax.set_zticks(ticks)
        ax.set_zticklabels([])
        ax.grid(True)
        ax._xpeak_has_scene = True
        ax.set_title("Virtual diffractometer", fontsize=11)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=camera[0], azim=camera[1], roll=camera[2])
        self.virtual_geometry_figure.tight_layout()
        self.virtual_geometry_canvas.draw_idle()

    def set_virtual_you_camera(self, draw=True) -> None:
        self.virtual_geometry_ax.view_init(elev=12, azim=-90, roll=0)
        if draw:
            self.virtual_geometry_canvas.draw_idle()

    def set_virtual_incidence_camera(self, draw=True) -> None:
        self.virtual_geometry_ax.view_init(elev=0, azim=0, roll=0)
        if draw:
            self.virtual_geometry_canvas.draw_idle()

    def simulate(self) -> None:
        try:
            cell, atoms, wavelength = self._model()
            sample = self._sample_from_kappa_input()
            detector = (float(self.delta.get()), float(self.nu.get()))
            try:
                kp = you_to_kappa(sample[1], sample[2], sample[3], float(self.kappa_alpha.get()))
                displayed_kappa = tuple(float(v.get()) for v in (self.komega, self.kappa_angle, self.kphi))
                self.kappa_status.config(text=f"You: μ={sample[0]:.3f} δ={detector[0]:.3f} ν={detector[1]:.3f} η={sample[1]:.3f} χ={sample[2]:.3f} φ={sample[3]:.3f}    |    κ: μ={sample[0]:.3f} δ={detector[0]:.3f} ν={detector[1]:.3f} κω={displayed_kappa[0]:.3f} κ={displayed_kappa[1]:.3f} κφ={displayed_kappa[2]:.3f}")
            except ValueError as exc:
                self.kappa_status.config(text=f"You position shown above | κ position unavailable: {exc}")
            spots = simulate_detector(cell, atoms, wavelength, sample, detector, float(self.distance.get()), float(self.pixel.get()), (512, 512), (256, 256), int(self.max_hkl.get()), float(self.mosaic.get()), self.u_matrix)
            try:
                current_hkl = motors_to_hkl(cell, (sample[0], detector[0], detector[1], sample[1], sample[2], sample[3]), wavelength, self.u_matrix)
                self.q_hkl_status.config(text=f"Q (r.l.u.) = ({current_hkl[0]:.5f}, {current_hkl[1]:.5f}, {current_hkl[2]:.5f})")
            except Exception as exc:
                self.q_hkl_status.config(text=f"Q (r.l.u.) unavailable: {exc}")
            self.update_virtual_3d(sample, detector)
            image = np.zeros((512, 512), dtype=float)
            yy, xx = np.mgrid[-4:5, -4:5]
            kernel = np.exp(-(xx * xx + yy * yy) / 3.0)
            for spot in spots:
                x, y = int(round(spot.x_px)), int(round(spot.y_px))
                xa, xb, ya, yb = max(0, x - 4), min(512, x + 5), max(0, y - 4), min(512, y + 5)
                image[ya:yb, xa:xb] += spot.intensity * kernel[ya - (y - 4):yb - (y - 4), xa - (x - 4):xb - (x - 4)]
            self.ax.clear()
            self.ax.imshow(np.log1p(image), cmap="magma", origin="upper", extent=(0, 512, 512, 0))
            self.ax.scatter([256], [256], marker="+", s=80, color="#38bdf8", linewidth=1)
            arrow_style = dict(arrowstyle="-|>", lw=2.2)
            self.ax.annotate("", xy=(145, 55), xytext=(55, 55), arrowprops={**arrow_style, "color": "#22d3ee"})
            self.ax.text(152, 55, "+x / u", color="#22d3ee", va="center", fontsize=10, fontweight="bold")
            self.ax.annotate("", xy=(55, 145), xytext=(55, 55), arrowprops={**arrow_style, "color": "#4ade80"})
            self.ax.text(55, 153, "+y / v", color="#4ade80", ha="center", va="top", fontsize=10, fontweight="bold")
            brightest = sorted(spots, key=lambda s: s.intensity, reverse=True)[:20]
            for s in brightest:
                self.ax.annotate(f"{s.h}{s.k}{s.l}", (s.x_px, s.y_px), xytext=(4, -4), textcoords="offset points", color="white", fontsize=7)
            self.ax.set(
                title=f"You 4S+2D detector: δ={detector[0]:g}°, ν={detector[1]:g}°",
                xlabel="Detector x / u (pixel)  → +x",
                ylabel="Detector y / v (pixel)  → +y downward",
                xlim=(0, 512),
                ylim=(512, 0),
            )
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.sim_status.config(text=f"{len(spots)} reflections on detector")
        except Exception as exc:
            messagebox.showerror("Cannot simulate", str(exc))

    def _sample_from_kappa_input(self) -> tuple[float, float, float, float]:
        """Resolve the physical kappa fields to the equivalent You sample motors."""
        mu = float(self.mu.get())
        komega, kappa, kphi = self._validated_kappa_angles()
        position = kappa_to_you(
            komega,
            kappa,
            kphi,
            float(self.kappa_alpha.get()),
        )
        for variable, value in zip((self.eta, self.chi, self.phi), (position.eta, position.chi, position.phi)):
            variable.set(f"{value:.8g}")
        return mu, float(self.eta.get()), float(self.chi.get()), float(self.phi.get())

    def _validated_kappa_angles(self) -> tuple[float, float, float]:
        komega = float(self.komega.get())
        kappa = float(self.kappa_angle.get())
        kphi = float(self.kphi.get())
        if not 0.0 <= komega <= 360.0:
            raise ValueError("κω must be between 0° and 360°.")
        if not 0.0 <= kappa <= 180.0:
            raise ValueError("κ must be between 0° and 180°.")
        if not 0.0 <= kphi <= 360.0:
            raise ValueError("κφ must be between 0° and 360°.")
        return komega, kappa, kphi

    def set_eulerian_orientation(self) -> None:
        """Use the Eulerian row and update the equivalent kappa row."""
        try:
            sample = tuple(float(v.get()) for v in (self.mu, self.eta, self.chi, self.phi))
            self._set_kappa_from_you(sample)
            self.simulate()
        except Exception as exc:
            messagebox.showerror("Cannot set Eulerian orientation", str(exc))

    def set_kappa_orientation(self) -> None:
        """Use the kappa row and update the equivalent Eulerian row."""
        try:
            self._sample_from_kappa_input()
            self.simulate()
        except Exception as exc:
            messagebox.showerror("Cannot set kappa orientation", str(exc))

    def _set_kappa_from_you(self, sample_motors) -> None:
        """Populate the physical kappa controls from a Diffcalc You solution."""
        mu, eta, chi, phi = (float(value) for value in sample_motors)
        kp = you_to_kappa(eta, chi, phi, float(self.kappa_alpha.get()))
        self.mu.set(f"{mu:.8g}")
        for variable, value in zip((self.eta, self.chi, self.phi), (eta, chi, phi)):
            variable.set(f"{value:.8g}")
        for variable, value in zip((self.komega, self.kappa_angle, self.kphi), (kp.komega, kp.kappa, kp.kphi)):
            variable.set(f"{value:.8g}")

    def export_csv(self) -> None:
        if not self.peaks:
            messagebox.showinfo("Export", "Run the peak finder first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["h", "k", "l", "d_A", "two_theta_deg", "q_A-1", "relative_intensity_percent"])
                for p in self.peaks:
                    writer.writerow([p.h, p.k, p.l, p.d, p.two_theta, p.q, p.intensity])

    def _sort(self, column: str) -> None:
        rows = []
        for item in self.tree.get_children(""):
            value = self.tree.set(item, column)
            try:
                key = (0, float(value))
            except ValueError:
                key = (1, value)
            rows.append((key, item))
        rows.sort(key=lambda row: row[0])
        for index, (_, item) in enumerate(rows):
            self.tree.move(item, "", index)


def main() -> None:
    XPeakApp().mainloop()
