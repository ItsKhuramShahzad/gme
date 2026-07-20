"""
__init__.py


hubbard_gme
===========
Genuine Multipartite Entanglement analysis for the Hubbard model AND
for small molecular active spaces, using n-body Reduced Density Matrices.

Modules
-------
hamiltonian  : Build and diagonalize the Hubbard Hamiltonian (k eigenstates)
rdm          : Compute n-body RDMs for arbitrary site clusters
metrics      : Purity, entropy, Hilbert-Schmidt and trace distances
gme          : GME analysis, bipartitions, C_GME concurrence
               Includes BOTH paper and supervisor lower bound methods.
convex_roof  : Convex-roof extension for mixed-state RDMs
               5 methods: paper bound, supervisor bound, upper bound,
               numerical optimization (exp-map), Stiefel manifold,
               optional CVXPY SDP relaxation.
landscape    : Landscape analysis (discrete scan, continuous 1D/2D)
               Answers: is the optimum unique? Single peak or plateau?
spin         : Spin observables (Sz, S², double occupancy)
scan         : Parameter scan engine (U scan, U/t scan)
plots        : Comparison plots and scan PDF reports
visualization: Heatmaps and single-point PDF report generation
pipeline     : End-to-end Hubbard analysis drivers
utils        : Basis labels, helper plots

Molecular extension
-------------------
molecules_data        : Database of molecules and CAS active spaces
molecular             : Molecular Hamiltonian, ground state, RDMs, spin/D
molecular_visualization : Molecular PDF report pages
molecular_pipeline    : End-to-end molecular analysis (RDMs + convex roof)
"""

# ── Hubbard model (existing) ────────────────────────────────────────────────
from .hamiltonian import (
    build_hubbard_hamiltonian,
    compute_ground_and_excited_states,
    compute_k_eigenstates,
    compute_all_eigenstates,
)
from .rdm import compute_general_nRDM, compute_all_pure_rdms, generate_site_clusters
from .metrics import compute_purity, compute_entropy, hilbert_schmidt_distance, trace_distance
from .gme import (
    analyze_one_rdm_for_GME,
    check_GME_for_all_rdms,
    compute_C_GME_pure,
    Cgme_lower_bound,
    I_function,
    supervisor_gme_lower_bound,
    supervisor_compute_A,
    supervisor_compute_Bk,
    supervisor_compute_I_phi,
    supervisor_test_pure_state,
)
from .convex_roof import (
    convex_roof_lower_bound,
    convex_roof_lower_bound_supervisor,
    convex_roof_upper_bound,
    convex_roof_numerical,
    convex_roof_stiefel,
    full_convex_roof_analysis,
    convex_roof_analysis_complete,
    convex_roof_sdp_relaxation,
)
from .landscape import (
    landscape_discrete_scan,
    landscape_continuous_1d,
    landscape_surface_2d,
    landscape_full_report,
)
from .scan import run_U_scan, run_Ut_scan
from .plots import plot_cgme_vs_U, plot_cgme_panel, generate_scan_pdf
from .pipeline import run_pure_analysis, run_full_GME_pipeline, run_scan_pipeline
from .visualization import generate_gme_pdf, generate_n_equals_N_pdf

# ── Molecular extension (new) ───────────────────────────────────────────────
from .molecules_data import MOLECULES, list_molecules, get_molecule
from .molecular import (
    build_molecular_hamiltonian,
    compute_molecular_ground_state,
    compute_molecular_spin_docc,
    compute_molecular_nrdm,
    compute_all_molecular_rdms,
    compute_rdm_properties,
    build_metrics_dict,
)
from .molecular_visualization import (
    generate_molecular_gme_pdf,
    generate_molecular_rdm_only_pdf,
)
from .molecular_pipeline import (
    run_convex_roof_on_rdm,
    run_molecular_analysis,
    run_all_molecules_analysis,
)

# ── Thermal / parameter-scan analyses (new) ─────────────────────────────────
from .thermal_pipeline import (
    run_thermal_gme_analysis,
    find_first_excited_singlet,
    run_excited_singlet_gme,
    run_t_dependence_gme,
    run_U_dependence_gme,
)

__version__ = "0.3.0"
__author__  = "Khuram Shahzad"

__all__ = [
    # Hamiltonian
    "build_hubbard_hamiltonian",
    "compute_ground_and_excited_states",
    "compute_k_eigenstates",
    "compute_all_eigenstates",
    # RDM
    "compute_general_nRDM",
    "compute_all_pure_rdms",
    "generate_site_clusters",
    # Metrics
    "compute_purity",
    "compute_entropy",
    "hilbert_schmidt_distance",
    "trace_distance",
    # GME analysis
    "analyze_one_rdm_for_GME",
    "check_GME_for_all_rdms",
    "compute_C_GME_pure",
    "Cgme_lower_bound",
    "I_function",
    "supervisor_gme_lower_bound",
    "supervisor_compute_A",
    "supervisor_compute_Bk",
    "supervisor_compute_I_phi",
    "supervisor_test_pure_state",
    # Convex roof
    "convex_roof_lower_bound",
    "convex_roof_lower_bound_supervisor",
    "convex_roof_upper_bound",
    "convex_roof_numerical",
    "convex_roof_stiefel",
    "full_convex_roof_analysis",
    "convex_roof_analysis_complete",
    "convex_roof_sdp_relaxation",
    # Landscape
    "landscape_discrete_scan",
    "landscape_continuous_1d",
    "landscape_surface_2d",
    "landscape_full_report",
    # Scan
    "run_U_scan",
    "run_Ut_scan",
    # Plotting
    "plot_cgme_vs_U",
    "plot_cgme_panel",
    "generate_scan_pdf",
    # Hubbard pipeline
    "run_pure_analysis",
    "run_full_GME_pipeline",
    "run_scan_pipeline",
    # Visualization (Hubbard)
    "generate_gme_pdf",
    "generate_n_equals_N_pdf",
    # Molecules (new)
    "MOLECULES",
    "list_molecules",
    "get_molecule",
    "build_molecular_hamiltonian",
    "compute_molecular_ground_state",
    "compute_molecular_spin_docc",
    "compute_molecular_nrdm",
    "compute_all_molecular_rdms",
    "compute_rdm_properties",
    "build_metrics_dict",
    "generate_molecular_gme_pdf",
    "generate_molecular_rdm_only_pdf",
    "run_convex_roof_on_rdm",
    "run_molecular_analysis",
    "run_all_molecules_analysis",
    # Thermal / parameter-scan
    "run_thermal_gme_analysis",
    "find_first_excited_singlet",
    "run_excited_singlet_gme",
    "run_t_dependence_gme",
    "run_U_dependence_gme",
]