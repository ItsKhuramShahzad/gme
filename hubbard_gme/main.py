#!/usr/bin/env python3
"""
main.py — Hubbard GME command-line interface

TOPOLOGIES
----------
  uniform   1-D ring, all bonds = t, PBC  (default)
  open      1-D open chain, no PBC
  nnn       Ring + next-nearest-neighbour hopping t'  (--t-prime)
  ssh       Su-Schrieffer-Heeger alternating t1/t2    (--t2)
  ladder    Two-leg ladder, leg=t rung=t_perp         (--t-perp, even L)

EXAMPLES
--------
# Standard ring with full analysis
python main.py --L 6 --U 4.0

# Include landscape analysis
python main.py --L 6 --U 4.0 --landscape

# Include continuous optimization
python main.py --L 6 --U 4.0 --continuous

# Full analysis with all features
python main.py --L 6 --U 4.0 --landscape --continuous

# U scan with landscape at each point
python main.py --scan-U --L 4 --mode nnn --t-prime 0.3 --U-min 0.5 --U-max 16 --U-steps 20 --landscape

# Open chain topology
python main.py --L 6 --U 4.0 --mode open

# SSH model (topological)
python main.py --L 6 --U 4.0 --mode ssh --t2 0.3

# Two-leg ladder
python main.py --L 6 --U 4.0 --mode ladder --t-perp 1.5

# Parallel convex-roof on all RDMs (8 worker processes)
python main.py --L 6 --U 4.0 --max-n 6 --min-n 3 --convex-roof-all --workers 8

# Enable PKL caching: reuse system-state + RDM PKLs and cache convex-roof results.
# A second identical run reuses cached cluster results and skips recomputation.
python main.py --L 6 --U 4.0 --min-n 3 --use-pkl-cache

# Force a fresh convex-roof recompute but keep state/RDM caches; use the
# meyerwallach cost measure with the Powell optimizer inside the Stiefel solver.
python main.py --L 6 --U 4.0 --use-pkl-cache --force-gme-recompute \
               --method meyerwallach --optimizer Powell

# Thermal nBRDM analysis with terminal parameters and PKL reuse.
python main.py --thermal --L 4 --U 4.0 --T-max 2 --T-steps 11 \
               --thermal-solver supervisor --use-pkl-cache

# Extended Hubbard: on-site U + nearest-neighbour V (V Σ_j n_j n_{j+1}).
# Single point with V (V=0 reproduces the plain Hubbard model exactly):
python main.py --L 6 --U 4.0 --V 5.0

# U-scan at a FIXED V (supervisor plot: C_GME vs U/t at V/t = 5, 10, 20).
# Run once per V; each auto-saves a PKL (V is in the filename) + a PDF.
python main.py --L 6 --Ne 6 --t 1.0 --u-scan \
               --u-scan-min 0 --u-scan-max 15 --u-scan-steps 16 \
               --V 5 --full-hamiltonian --n-restarts 8 --k-max-extra 4 \
               --use-pkl-cache --output-prefix V5

# V-scan: FIX t and U, sweep V — shows the Mott -> CDW transition at V ~ U/2.
python main.py --L 6 --Ne 6 --t 1.0 --U 10 --v-scan \
               --v-scan-min 0 --v-scan-max 15 --v-scan-steps 16 \
               --full-hamiltonian --n-restarts 8 --k-max-extra 4 \
               --use-pkl-cache --output-prefix Vscan_U10
"""

import argparse
import sys
import os

# ── BLAS thread pinning ──────────────────────────────────────────────────────
# CRITICAL for multiprocessing: each worker re-imports NumPy/SciPy and would
# spawn N internal BLAS threads by default.  With M parallel processes that
# becomes M*N threads competing for cores — slower than serial.  Pin each
# worker to one BLAS thread.  These environment variables MUST be set before
# numpy is imported.
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",  "1")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from hubbard_gme.pipeline import run_full_GME_pipeline, run_pure_analysis, run_scan_pipeline
from hubbard_gme.hamiltonian import VALID_MODES
from hubbard_gme.thermal_pipeline import (
    run_thermal_gme_analysis,
    run_excited_singlet_gme,
    run_t_dependence_gme,
    run_U_dependence_gme,
    run_V_dependence_gme,
    run_tT_surface_gme,
    render_surface_from_pkl,
    merge_tT_surface_pkls,
    build_or_load_rdm_bank,
    plot_scan_finalized,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hubbard-gme",
        description="GME analysis for the Hubbard model — 5 topologies supported.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── System parameters ──────────────────────────────────────────────────
    sys_grp = p.add_argument_group("System parameters")
    sys_grp.add_argument("--L",    type=int,   default=4,
                         metavar="INT",   help="Number of sites (default: 4)")
    sys_grp.add_argument("--t",    type=float, default=1.0,
                         metavar="FLOAT", help="Primary hopping amplitude (default: 1.0)")
    sys_grp.add_argument("--U",    type=float, default=4.0,
                         metavar="FLOAT", help="On-site repulsion (default: 4.0)")
    sys_grp.add_argument("--V", type=float, default=0.0, 
                         metavar="FLOAT", help="Nearest-neighbor repulsion (default: 0.0)")
    sys_grp.add_argument("--mode", type=str,   default="uniform",
                         choices=list(VALID_MODES),
                         help="Topology (default: uniform)")
    sys_grp.add_argument("--Ne",   type=int,   default=None,
                         metavar="INT",   help="Total electrons (default: L, half-filling)")
    sys_grp.add_argument("--max-n",type=int,   default=4, dest="max_n",
                         metavar="INT",   help="Max n-body RDM order (default: 4)")
    sys_grp.add_argument("--min-n",type=int,   default=1, dest="min_n",
                         metavar="INT",   help="Min n-body RDM order (default: 1)")
    sys_grp.add_argument("--output-dir", type=str, default="outputs",
                         dest="output_dir", metavar="PATH",
                         help="Output directory (default: ./outputs)")
    sys_grp.add_argument("--state-idx", type=int, default=0, dest="state_idx",
                         metavar="INT",
                         help="Eigenstate index, 0-indexed (default: 0 = ground state). "
                              "For supervisor's task, use --state-idx 4 for the "
                              "5th state (= 1st excited singlet for L=6, U=4).")
    
    sys_grp.add_argument("--method", type=str, default="cgme",
                        choices=["cgme", "meyerwallach"],
                        help="Cost measure minimised by the convex-roof solvers "
                             "(default: cgme). Passed to all three solvers "
                             "(stiefel, gcg, stiefel_gcg). Aliases c_gme/mw also "
                             "accepted by the pipeline.")
    sys_grp.add_argument("--optimizer", type=str, default="L-BFGS-B",
                        choices=["L-BFGS-B", "Powell", "SLSQP", "TNC", "trust-constr"],
                        help="Optimizer used inside the Stiefel solver only "
                             "(default: L-BFGS-B). The gcg and stiefel_gcg solvers "
                             "use their own internal conjugate-gradient and ignore it.")
    sys_grp.add_argument("--reload", action="store_true",
                        help="Reload existing data from .npz and just regenerate the plot.")
    
    sys_grp.add_argument("--n-restarts", type=int, default=10,
                        help="Restarts per run (default: 10).")
    sys_grp.add_argument("--k-max-extra", type=int, default=2,
                        help="k_max_extra (default: 2).")
    sys_grp.add_argument("--max-iter", type=int, default=None, dest="max_iter",
                        metavar="INT",
                        help="Max optimizer iterations per restart (default: None = "
                             "let each solver use its own default).")
    sys_grp.add_argument("--tol", type=float, default=None, metavar="FLOAT",
                        help="Convergence tolerance recorded with cached results "
                             "(default: None = solver default).")
    sys_grp.add_argument("--input-npz", type=str, default=None,
                        help="Custom input .npz path (for --reload).")
    

    # ── PDF and analysis options ───────────────────────────────────────────
    pdf_grp = p.add_argument_group("Output and analysis options")
    pdf_grp.add_argument("--no-pdf", action="store_true",
                         help="Skip main PDF generation")
    pdf_grp.add_argument("--no-verify", action="store_true",
                         help="Skip n=N verification even when Ne == max-n")
    pdf_grp.add_argument("--continuous", action="store_true",
                         help="Run continuous optimization of I_Phi over Bloch sphere "
                              "(supervisor suggestion, gives tighter lower bound)")
    pdf_grp.add_argument("--real-only", action="store_true", dest="real_only",
                         help="With --continuous: use real states only (phi_k=0). "
                              "Faster intermediate step.")
    pdf_grp.add_argument("--landscape", action="store_true",
                         help="Generate landscape analysis (discrete + continuous 1D + 2D surface). "
                              "For n=N pure states only. Creates separate landscape PDF.")
    pdf_grp.add_argument("--landscape-no-2d", action="store_true", dest="landscape_no_2d",
                         help="Landscape analysis: skip 2D surface (faster if only checking discrete+1D)")
    pdf_grp.add_argument("--convex-roof-all", action="store_true", dest="convex_roof_all",
                         help="Compute convex roof for ALL RDMs (pure n=N and mixed n<N). "
                              "Default: only for mixed RDMs.")
    pdf_grp.add_argument("--workers", type=int, default=8, dest="n_workers",
                         metavar="INT",
                         help="Number of parallel worker processes for convex-roof "
                              "analysis and thermal RDM cluster jobs (default: 8). "
                              "Each cluster is processed independently. Use 1 to "
                              "disable parallelism.")

    # ── Topology extra parameters ──────────────────────────────────────────
    topo_grp = p.add_argument_group("Topology parameters")
    topo_grp.add_argument("--t-prime", type=float, default=None,
                          dest="t_prime", metavar="FLOAT",
                          help="NNN hopping for --mode nnn (default: 0.3*t)")
    topo_grp.add_argument("--t2",      type=float, default=None,
                          metavar="FLOAT",
                          help="Weak bond hopping for --mode ssh (default: 0.5*t)")
    topo_grp.add_argument("--t-perp",  type=float, default=None,
                          dest="t_perp", metavar="FLOAT",
                          help="Rung hopping for --mode ladder (default: t)")

    # ── Scan mode ──────────────────────────────────────────────────────────
    scan_grp = p.add_argument_group("Parameter scan")
    scan_grp.add_argument("--scan-U",   action="store_true",
                          help="Scan over U values (fixed t)")
    scan_grp.add_argument("--scan-Ut",  action="store_true",
                          help="Scan over U/t ratio")
    scan_grp.add_argument("--U-min",    type=float, default=0.5,  metavar="FLOAT",
                          help="U scan start  (default: 0.5)")
    scan_grp.add_argument("--U-max",    type=float, default=16.0, metavar="FLOAT",
                          help="U scan end    (default: 16.0)")
    scan_grp.add_argument("--U-steps",  type=int,   default=20,   metavar="INT",
                          help="U scan steps  (default: 20)")
    scan_grp.add_argument("--Ut-min",   type=float, default=0.1,  metavar="FLOAT",
                          help="U/t scan start (default: 0.1)")
    scan_grp.add_argument("--Ut-max",   type=float, default=20.0, metavar="FLOAT",
                          help="U/t scan end   (default: 20.0)")
    scan_grp.add_argument("--Ut-steps", type=int,   default=25,   metavar="INT",
                          help="U/t scan steps (default: 25)")
    scan_grp.add_argument("--n-body",   type=int,   default=3,    dest="n_body",
                          metavar="INT",   help="RDM body order for scan (default: 3)")
    scan_grp.add_argument("--k-states", type=int,   default=4,    dest="k_states",
                          metavar="INT",   help="Eigenstates per scan point (default: 4)")

    # ── Thermal nBRDM pipeline ──────────────────────────────────────────────
    therm_grp = p.add_argument_group("Thermal nBRDM analysis")
    therm_grp.add_argument("--thermal", action="store_true",
                           help="Run thermal nBRDM/GME analysis from thermal_pipeline.py")
    therm_grp.add_argument("--T-min", type=float, default=0.0, dest="T_min",
                           metavar="FLOAT", help="Thermal sweep start (default: 0)")
    therm_grp.add_argument("--T-max", type=float, default=2.0, dest="T_max",
                           metavar="FLOAT", help="Thermal sweep end (default: 2)")
    therm_grp.add_argument("--T-steps", type=int, default=11, dest="T_steps",
                           metavar="INT", help="Number of T points (default: 11)")
    therm_grp.add_argument("--thermal-solver", type=str, default="gcg",
                           choices=["gcg", "supervisor"],
                           help="Thermal mixed-state solver (default: gcg)")
    therm_grp.add_argument("--thermal-energy-cutoff", type=float, default=10.0,
                           dest="thermal_energy_cutoff", metavar="FLOAT",
                           help="Include states with (E_k-E_ref)/T below this cutoff "
                                "(default: 10)")
    therm_grp.add_argument("--thermal-all-states", action="store_true",
                           help="Ignore --thermal-energy-cutoff and include all eigenstates")
    therm_grp.add_argument("--thermal-representative-cluster", action="store_false",
                           dest="thermal_average_site_clusters",
                           help="Use only sites (0,..,n-1) instead of averaging over "
                                "all contiguous site clusters")
    therm_grp.add_argument("--output-prefix", type=str, default="",
                           dest="output_prefix", metavar="STR",
                           help="Prefix for thermal output filenames")
    therm_grp.add_argument("--force-thermal-recompute", action="store_true",
                           help="With --thermal --use-pkl-cache, ignore existing thermal PKL")
    therm_grp.add_argument("--no-save-thermal-pkl", action="store_false",
                           dest="save_thermal_pkl",
                           help="With --thermal --use-pkl-cache, do not write thermal PKL")

    # ── Excited singlet analysis ───────────────────────────────────────────
    therm_grp.add_argument("--excited-singlet", action="store_true",
                           dest="excited_singlet",
                           help="Run GME analysis of the first singlet excited state. "
                                "Produces a bar chart of C_GME(n) at T=0. "
                                "Add --T-sweep to also plot C_GME(n, T).")
    therm_grp.add_argument("--singlet-k", type=int, default=12,
                           dest="singlet_k", metavar="INT",
                           help="Eigenstates to compute for singlet search (default: 12). "
                                "Increase if no excited singlet is found.")
    therm_grp.add_argument("--T-sweep", action="store_true", dest="T_sweep",
                           help="With --excited-singlet: also run a thermal sweep "
                                "using --T-min/--T-max/--T-steps as the grid.")

    # ── t-dependence scan (T=0, vary t) ───────────────────────────────────
    therm_grp.add_argument("--t-scan", action="store_true", dest="t_scan",
                           help="Scan C_GME(n) vs t at T=0, fixing U. "
                                "Use --t-scan-min/max/steps for the t grid.")
    therm_grp.add_argument("--t-scan-min", type=float, default=0.05,
                           dest="t_scan_min", metavar="FLOAT",
                           help="t-scan start (default: 0.05)")
    therm_grp.add_argument("--t-scan-max", type=float, default=2.0,
                           dest="t_scan_max", metavar="FLOAT",
                           help="t-scan end   (default: 2.0)")
    therm_grp.add_argument("--t-scan-steps", type=int, default=20,
                           dest="t_scan_steps", metavar="INT",
                           help="t-scan steps (default: 20)")
    # ── 3D surface: C_GME(t/U, T)  (standalone task) ──────────────────────
    therm_grp.add_argument("--tT-surface", action="store_true",
                           dest="tT_surface",
                           help="Standalone 3D surface C_GME(t/U, T): one 3D "
                                "panel per body-order n. Uses --t-scan-min/max/"
                                "steps for the t grid and --T-min/--T-max with "
                                "--tT-surface-T-steps for the temperature grid.")
    therm_grp.add_argument("--tT-surface-T-steps", type=int, default=15,
                           dest="tT_surface_T_steps", metavar="INT",
                           help="Temperature grid points for the 3D surface "
                                "(default: 15). A --tT-surface run writes BOTH "
                                "the 3D surface and the 2D heatmap PDF.")
    therm_grp.add_argument("--tT-surface-max-n", type=int, default=None,
                           dest="tT_surface_max_n", metavar="INT",
                           help="Cap the largest body-order n for the surface "
                                "(default: L). The RDM-bank build AND the GCG "
                                "convex-roof cost blow up for large n (n=7,8 are "
                                "128x128 / 256x256 mixed states), so e.g. "
                                "--tT-surface-max-n 4 makes L=8 feasible. Also "
                                "shrinks the cached RDM bank accordingly.")
    therm_grp.add_argument("--render-surface", type=str, default=None,
                           dest="render_surface", metavar="PKL",
                           help="Re-render the 3D surface + 2D heatmap from a "
                                "cached tTsurface_*.pkl (no recompute). Output "
                                "goes to --output-dir (default: outputs/).")
    therm_grp.add_argument("--replot", type=str, nargs="+", default=None,
                           dest="replot", metavar="PKL",
                           help="Regenerate t-scan / U-scan figures from cached "
                                "scan PKLs (no recompute). Uses the same "
                                "plot_scan_finalized() the scans call, so a "
                                "replot always matches a fresh scan. Accepts "
                                "paths and globs; pass 'all' for every t-scan "
                                "and U-scan PKL under results/.../thermal_runs. "
                                "Output goes to --output-dir/replots.")
    therm_grp.add_argument("--merge-surface", type=str, nargs="+", default=None,
                           dest="merge_surface", metavar="PKL",
                           help="Merge several per-n surface PKLs (each from a "
                                "single-n job: --min-n N --tT-surface-max-n N) "
                                "into ONE combined 3D+2D figure over all n, plus "
                                "a re-usable merged PKL. Accepts paths and globs, "
                                "e.g. --merge-surface 'results/.../tTsurface_*n*'.")
    therm_grp.add_argument("--spectrum", action="store_true",
                           help="Print the low-energy eigenstate spectrum with "
                                "spin classification (idx, E, <S^2>, S, label) "
                                "and save it as a CSV. Uses --L/--Ne/--t/--U/"
                                "--mode and --spectrum-k.")
    therm_grp.add_argument("--spectrum-k", type=int, default=12,
                           dest="spectrum_k", metavar="INT",
                           help="Number of lowest eigenstates for --spectrum "
                                "(default: 12).")

    # ── Standalone RDM-bank generation ────────────────────────────────────
    therm_grp.add_argument("--build-bank", action="store_true",
                           dest="build_bank",
                           help="Build (or load) ONLY the generic RDM bank for "
                                "the given Hamiltonian (--L/--Ne/--t/--U/--mode) "
                                "and cache it under results/rdm_pkl_results/"
                                "rdm_bank/. The bank covers n = --min-n .. L and "
                                "is reused by every later analysis. No plots.")
    therm_grp.add_argument("--bank-max-n", type=int, default=None,
                           dest="bank_max_n", metavar="INT",
                           help="With --build-bank: cap the largest body-order n "
                                "(default: L, i.e. uncapped). n=7,8 banks at L=8 "
                                "are ~18GB each (all eigenstates x 128x128/256x256 "
                                "RDMs) — use e.g. --bank-max-n 4 for L>=7 to avoid "
                                "filling disk. Does not affect existing L<=6 caches.")
    # Optional grid sweep: --build-bank normally builds the single point given
    # by --L/--Ne/--t/--U. Supplying any of the lists below turns it into a
    # cartesian sweep, so widening the ML dataset needs no separate script.
    therm_grp.add_argument("--bank-L", nargs="+", type=int, default=None,
                           dest="bank_L", metavar="INT",
                           help="With --build-bank: sweep these ring sizes "
                                "instead of the single --L.")
    therm_grp.add_argument("--bank-U", nargs="+", type=float, default=None,
                           dest="bank_U", metavar="FLOAT",
                           help="With --build-bank: sweep these U values "
                                "instead of the single --U.")
    therm_grp.add_argument("--bank-t", nargs="+", type=float, default=None,
                           dest="bank_t", metavar="FLOAT",
                           help="With --build-bank: sweep these hoppings "
                                "instead of the single --t.")
    therm_grp.add_argument("--bank-Ne", nargs="+", type=int, default=None,
                           dest="bank_Ne", metavar="INT",
                           help="With --build-bank: sweep these fillings "
                                "(default: --Ne, else half filling Ne=L).")
    therm_grp.add_argument("--bank-sectors", default=None,
                           choices=["sz0", "full", "both"], dest="bank_sectors",
                           help="With --build-bank: which S_z sector(s) to build "
                                "(default: follows --full-hamiltonian).")
    therm_grp.add_argument("--bank-dry-run", action="store_true",
                           dest="bank_dry_run",
                           help="With --build-bank: print the grid and the record "
                                "yield, build nothing.")
    # --- chain straight into GCG labelling so one command stores everything ---
    therm_grp.add_argument("--bank-label", action="store_true",
                           dest="bank_label",
                           help="With --build-bank: after building, run the GCG "
                                "convex roof on every bank RDM and store the "
                                "labels (C_GME, Wootters for n=2), the full "
                                "convergence history and the RDM metrics into "
                                "the reserved ML store. Resumable: anything "
                                "already labelled is skipped, never recomputed.")
    therm_grp.add_argument("--bank-label-T", nargs="+", type=float,
                           default=[0.0], dest="bank_label_T", metavar="FLOAT",
                           help="With --bank-label: temperatures to label from "
                                "each bank (default: 0.0 = ground state only).")
    therm_grp.add_argument("--bank-label-outdir", default=None,
                           dest="bank_label_outdir", metavar="DIR",
                           help="With --bank-label: dataset root "
                                "(default: <repo>/gme_qml/data_for_ML_v2). "
                                "Relative paths are anchored to the repo.")

    # ── U-dependence scan (T=0, vary U) ───────────────────────────────────
    therm_grp.add_argument("--u-scan", action="store_true", dest="u_scan",
                           help="Scan C_GME(n) vs U at T=0, fixing t. "
                                "Use --u-scan-min/max/steps for the U grid.")
    therm_grp.add_argument("--u-scan-min", type=float, default=1.0,
                           dest="u_scan_min", metavar="FLOAT",
                           help="U-scan start (default: 1.0)")
    therm_grp.add_argument("--u-scan-max", type=float, default=20.0,
                           dest="u_scan_max", metavar="FLOAT",
                           help="U-scan end   (default: 20.0)")
    therm_grp.add_argument("--u-scan-steps", type=int, default=20,
                           dest="u_scan_steps", metavar="INT",
                           help="U-scan steps (default: 20)")
    therm_grp.add_argument("--v-scan", action="store_true", dest="v_scan",
                           help="Scan C_GME(n) vs V at T=0, fixing t and U "
                                "(extended-Hubbard V-scan; shows the CDW "
                                "transition). Use --v-scan-min/max/steps and "
                                "--U to set the fixed U.")
    therm_grp.add_argument("--v-scan-min", type=float, default=0.0,
                           dest="v_scan_min", metavar="FLOAT",
                           help="V-scan start (default: 0.0)")
    therm_grp.add_argument("--v-scan-max", type=float, default=15.0,
                           dest="v_scan_max", metavar="FLOAT",
                           help="V-scan end   (default: 15.0)")
    therm_grp.add_argument("--v-scan-steps", type=int, default=16,
                           dest="v_scan_steps", metavar="INT",
                           help="V-scan steps (default: 16)")
    therm_grp.add_argument("--full-hamiltonian", action="store_true",
                           dest="full_hamiltonian",
                           help="Use complete Hamiltonian (all Sz sectors). "
                                "Default: Sz=0 sector only (spin_preserving=True).")
    therm_grp.add_argument("--T-view-max", type=float, default=None, dest="T_view_max",
                       help="Cap the thermal-plot x-axis at this T (view only; "
                            "does not change the data). Default: auto-clip.")

    # ── PKL persistence / reuse ──────────────────────────────────────────────
    cache_grp = p.add_argument_group("PKL persistence and reuse")
    cache_grp.add_argument("--use-pkl-cache", action="store_true", dest="use_pkl_cache",
                           help="Enable central PKL caching: reuse/save the "
                                "system-state, RDM, thermal, and excited-singlet "
                                "PKLs, and cache convex-roof results under "
                                "results/rdm_pkl_results/. "
                                "Plain topologies only (uniform/open).")
    cache_grp.add_argument("--cache-version", type=str, default="v1", dest="cache_version",
                           metavar="STR", help="Cache version tag (default: v1).")
    cache_grp.add_argument("--force-state-recompute", action="store_true",
                           dest="force_state_recompute",
                           help="Recompute the system state even if a cached PKL exists.")
    cache_grp.add_argument("--force-rdm-recompute", action="store_true",
                           dest="force_rdm_recompute",
                           help="Recompute RDMs even if cached PKLs exist.")
    cache_grp.add_argument("--no-reuse-gme-pkl", action="store_false", dest="reuse_gme_pkl",
                           help="Do not load cached convex-roof cluster results "
                                "(recompute all clusters). Default: reuse.")
    cache_grp.add_argument("--force-gme-recompute", action="store_true",
                           dest="force_gme_recompute",
                           help="Recompute convex-roof results even if cached "
                                "cluster PKLs exist. This is a hard global "
                                "override for all solver methods.")
    cache_grp.add_argument("--gme-cache-default", type=str, default="reload",
                           choices=["reload", "recompute"],
                           dest="gme_cache_default",
                           help="Default cache action for each convex-roof solver "
                                "(default: reload). 'reload' means reuse the solver "
                                "result from PKL when present, otherwise compute it.")
    cache_grp.add_argument("--stiefel-cache", type=str, default="default",
                           choices=["default", "reload", "recompute"],
                           dest="stiefel_cache_action",
                           help="Per-method cache override for the Stiefel solver. "
                                "Use 'default' to follow --gme-cache-default.")
    cache_grp.add_argument("--gcg-cache", type=str, default="default",
                           choices=["default", "reload", "recompute"],
                           dest="gcg_cache_action",
                           help="Per-method cache override for the GCG solver. "
                                "Use 'default' to follow --gme-cache-default.")
    cache_grp.add_argument("--stiefel-gcg-cache", type=str, default="default",
                           choices=["default", "reload", "recompute"],
                           dest="stiefel_gcg_cache_action",
                           help="Per-method cache override for the hybrid Stiefel-GCG solver. "
                                "Use 'default' to follow --gme-cache-default.")
    cache_grp.add_argument("--no-save-gme-pkl", action="store_false", dest="save_gme_pkl",
                           help="Do not write convex-roof job/full PKLs.")
    cache_grp.add_argument("--no-load-from-full-runs", action="store_true",
                           dest="no_load_from_full_runs",
                           help="Do not attempt recovery from older full-run PKLs "
                                "(reserved; per-cluster job_cache reuse still applies).")

    return p


def _topo_kwargs(args) -> dict:
    """Collect topology extra params into a dict for pipeline calls."""
    return dict(t_prime=args.t_prime, t2=args.t2, t_perp=args.t_perp)


def _solver_kwargs(args) -> dict:
    """Convex-roof solver settings threaded into the pipeline."""
    return dict(
        method=args.method,
        optimizer=args.optimizer,
        n_restarts=args.n_restarts,
        k_max_extra=args.k_max_extra,
        max_iter=args.max_iter,
        tol=args.tol,
    )


def _cache_kwargs(args) -> dict:
    """PKL persistence / reuse settings threaded into the pipeline."""
    return dict(
        use_pkl_cache=args.use_pkl_cache,
        cache_version=args.cache_version,
        force_state_recompute=args.force_state_recompute,
        force_rdm_recompute=args.force_rdm_recompute,
        reuse_gme_pkl=args.reuse_gme_pkl,
        force_gme_recompute=args.force_gme_recompute,
        gme_cache_default=args.gme_cache_default,
        stiefel_cache_action=args.stiefel_cache_action,
        gcg_cache_action=args.gcg_cache_action,
        stiefel_gcg_cache_action=args.stiefel_gcg_cache_action,
        no_load_from_full_runs=args.no_load_from_full_runs,
        save_gme_pkl=args.save_gme_pkl,
    )


def main():
    parser = build_parser()
    args   = parser.parse_args()

    # ── Print topology reminder ────────────────────────────────────────────
    topo_info = {
        "uniform": "1-D ring, PBC, all bonds = t",
        "open":    "1-D open chain, no PBC",
        "nnn":     f"Ring + NNN hopping t'={args.t_prime or f'0.3*t={0.3*args.t:.3f}'}",
        "ssh":     f"SSH chain, t1={args.t}, t2={args.t2 or f'0.5*t={0.5*args.t:.3f}'} (open BC)",
        "ladder":  f"Two-leg ladder, t={args.t}, t_perp={args.t_perp or args.t} (even L required)",
        "dimerized":"Dimerized (legacy)",
        "heisenberg":"Heisenberg limit (legacy)",
    }
    print(f"\n  Topology : {args.mode}  —  {topo_info.get(args.mode, '')}")
    if args.n_workers and args.n_workers > 1:
        print(f"  Workers  : {args.n_workers}  (parallel cluster jobs)")
    else:
        print(f"  Workers  : 1  (serial)")

    # ── Validate landscape option ──────────────────────────────────────────
    ne_actual = args.Ne if args.Ne is not None else args.L
    if args.landscape and ne_actual != args.max_n:
        print(f"\n  WARNING: --landscape requires n=N (Ne={ne_actual} == max-n={args.max_n})")
        print(f"  Disabling landscape analysis. Set --Ne {args.max_n} to enable.\n")
        args.landscape = False

    # ── Eigenstate energy spectrum + spin classification ───────────────────
    if args.spectrum:
        from hubbard_gme.hamiltonian import (
            compute_k_eigenstates, classify_eigenstates_spin,
        )
        ne = args.Ne if args.Ne is not None else args.L
        sp = not args.full_hamiltonian
        k_disp = args.spectrum_k
        # S_z is only well-defined once a FULL spin multiplet is captured: a
        # degenerate level of total spin S spans (2S+1) rows, and the S_z
        # block-diagonalisation needs ALL of them. A raw --spectrum-k cutoff can
        # slice through such a multiplet (e.g. a quintet = 5 rows), leaving a
        # partial block whose "S_z" is the meaningless expectation value of an
        # arbitrary superposition (the spurious non-integer S_z, e.g. +0.71). So
        # compute a buffer of extra states — the largest possible multiplicity
        # is Ne+1 (S=Ne/2) — then display only the first k_disp rows, which
        # guarantees every shown level is a complete multiplet.
        k_buf = k_disp + (ne + 2)
        energies, eigvecs, n_so = compute_k_eigenstates(
            args.L, args.t, args.U, args.mode, ne, k=k_buf,
            spin_preserving=sp,
        )
        n_have = len(energies)
        n_disp = min(k_disp, n_have)
        table = classify_eigenstates_spin(
            eigvecs=eigvecs, energies=energies, L=args.L,
            n_electrons=ne, n_spin_orbitals=n_so, spin_preserving=sp,
        )
        # Resolve S_z by diagonalising S_z inside each degenerate energy block.
        # A block that runs to the last COMPUTED state may itself be truncated,
        # so its S_z is left unresolved (NaN); with the buffer above this only
        # ever affects rows beyond n_disp.
        from openfermion.hamiltonians import sz_operator
        from openfermion.linalg import get_number_preserving_sparse_operator
        import numpy as _np, math as _math
        # openfermion sz_operator already IS S_z (eigenvalues -S..+S), no 0.5.
        Sz_op = get_number_preserving_sparse_operator(
            sz_operator(args.L), n_so, ne, spin_preserving=sp)
        Sz_clean = _np.full(n_have, _np.nan)
        _i = 0
        while _i < n_have:
            _j = _i
            while _j < n_have and abs(energies[_j] - energies[_i]) < 1e-6:
                _j += 1
            if _j == n_have:
                break  # block touches the computed boundary → may be incomplete
            blk = eigvecs[:, _i:_j]
            Szsub = blk.conj().T @ (Sz_op @ blk)
            Sz_clean[_i:_j] = _np.sort(
                _np.real(_np.linalg.eigvalsh((Szsub + Szsub.conj().T) / 2)))
            _i = _j
        for k_, r in enumerate(table):
            r["Sz"] = float(Sz_clean[k_])
        sector = "Sz=0 sector" if sp else "full Hamiltonian (all Sz)"
        print(f"\n  Eigenstate spectrum — L={args.L}, t={args.t}, U={args.U} "
              f"(t/U={args.t/args.U:.4g}), Ne={ne}, mode={args.mode} [{sector}]")
        print(f"  {'idx':>3}  {'E':>12}  {'<S^2>':>8}  {'S':>5}  {'Sz':>6}  label")
        print("  " + "-" * 60)
        def _z(x):  # clamp tiny negatives so they don't print as -0.0000
            return 0.0 if abs(x) < 1e-9 else x
        def _fmt_sz(x):  # unresolved S_z prints as '?'
            return "   ?  " if (x is None or _math.isnan(x)) else f"{_z(x):>+6.2f}"
        unresolved = False
        for r in table[:n_disp]:
            if r["Sz"] is None or _math.isnan(r["Sz"]):
                unresolved = True
            print(f"  {r['idx']:>3}  {r['energy']:>12.5f}  {_z(r['S2']):>8.4f}  "
                  f"{_z(r['S']):>5.3f}  {_fmt_sz(r['Sz'])}  {r['spin_label']}")
        if unresolved:
            print("\n  ('?' = S_z unresolved: this spin multiplet extends past the "
                  "computed states; raise --spectrum-k to capture it fully.)")
        singlets = [r for r in table[:n_disp] if abs(r["S"]) < 0.05]
        if len(singlets) > 1:
            fx = singlets[1]
            print(f"\n  First excited singlet: idx={fx['idx']}, E={fx['energy']:.5f}")
        # save CSV (only the displayed rows)
        import csv as _csv
        os.makedirs(args.output_dir, exist_ok=True)
        sec_tag = "sz0" if sp else "full"
        csv_path = os.path.join(
            args.output_dir,
            f"spectrum_L{args.L}_Ne{ne}_t{args.t}_U{args.U}_{sec_tag}.csv")
        with open(csv_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["idx", "energy", "S2", "S", "Sz", "multiplicity",
                        "spin_label"])
            for r in table[:n_disp]:
                sz_csv = ("" if (r["Sz"] is None or _math.isnan(r["Sz"]))
                          else r["Sz"])
                w.writerow([r["idx"], r["energy"], r["S2"], r["S"], sz_csv,
                            r["multiplicity"], r["spin_label"]])
        print(f"  CSV → {csv_path}\n")
        return

    # ── Re-plot t-scan / U-scan figures from cached PKLs (no recompute) ─────
    if args.replot:
        import glob as _glob
        import pickle as _pickle
        from hubbard_gme.rdm import get_rdm_pkl_results_dir

        thermal_dir = get_rdm_pkl_results_dir() / "thermal_runs"
        files: list = []
        for spec in args.replot:
            if spec == "all":
                for pat in ("tscan*.pkl", "uscan*.pkl"):
                    files += sorted(str(p) for p in thermal_dir.glob(pat))
            else:
                hits = sorted(_glob.glob(spec))
                if not hits and not os.path.isabs(spec):
                    # bare filename or glob: look inside the reserved dir
                    hits = sorted(str(p) for p in thermal_dir.glob(spec))
                if not hits:
                    # Report a miss as a miss. Passing the pattern through as a
                    # literal path only produces a confusing FileNotFoundError.
                    print(f"  [no match] {spec}")
                    continue
                files += hits
        # de-duplicate, preserve order
        files = list(dict.fromkeys(files))
        if not files:
            avail = sorted(p.name for p in thermal_dir.glob("[tu]scan*.pkl"))
            msg = (f"--replot: no matching scan PKLs.\n"
                   f"  looked in: {thermal_dir}\n"
                   f"  note: --output-prefix names the FIGURE, not the PKL; the\n"
                   f"        PKL is named from parameters. Excited-singlet runs\n"
                   f"        carry an '_es_' tag, e.g. 'uscan*_es_*.pkl'.\n"
                   f"  {len(avail)} scan PKL(s) available; first few:")
            for a in avail[:6]:
                msg += f"\n    {a}"
            parser.error(msg)

        outdir = os.path.join(args.output_dir, "replots")
        os.makedirs(outdir, exist_ok=True)
        ok = fail = 0
        for pkl_path in files:
            try:
                with open(pkl_path, "rb") as fh:
                    d = _pickle.load(fh)
                r = d.get("result", d)
                base = os.path.splitext(os.path.basename(pkl_path))[0]
                out = os.path.join(outdir, f"REPLOT_{base}.pdf")
                # sector is encoded in the filename: '..._sz0_...' vs '..._full_...'
                plot_scan_finalized(r, out,
                                    spin_preserving="sz0" in os.path.basename(pkl_path),
                                    save_png=True)
                kind = "t-scan" if "t_values" in r else "U-scan"
                print(f"  {kind}  n={sorted(r['C_GME'])} → {os.path.basename(out)}")
                ok += 1
            except Exception as e:
                print(f"  [skip] {os.path.basename(pkl_path)}: "
                      f"{type(e).__name__}: {e}")
                fail += 1
        print(f"\n  regenerated {ok} figures ({fail} skipped) → {outdir}/\n")
        return

    # ── Re-render surface + heatmap from a cached PKL (no recompute) ────────
    if args.render_surface:
        if not os.path.exists(args.render_surface):
            parser.error(f"--render-surface: file not found: {args.render_surface}")
        paths = render_surface_from_pkl(args.render_surface,
                                        output_dir=args.output_dir)
        print(f"\n  3D surface  → {paths['surface_path']}")
        print(f"  2D heatmap  → {paths['heatmap_path']}\n")
        return

    # ── Merge per-n surface PKLs into one combined figure (no recompute) ────
    if args.merge_surface:
        out = merge_tT_surface_pkls(args.merge_surface,
                                    output_dir=args.output_dir)
        print(f"\n  merged n = {out['n_values']}")
        print(f"  3D surface  → {out['surface_path']}")
        print(f"  2D heatmap  → {out['heatmap_path']}")
        if out.get("merged_pkl"):
            print(f"  merged PKL  → {out['merged_pkl']}")
        print()
        return

    # ── Standalone RDM-bank build mode ─────────────────────────────────────
    if args.build_bank:
        import time as _time

        # A sweep is requested if ANY of the --bank-{L,U,t,Ne,sectors} lists is
        # given; otherwise fall back to the single point from --L/--U/--t/--Ne
        # so existing commands behave exactly as before.
        sweeping = any(x is not None for x in
                       (args.bank_L, args.bank_U, args.bank_t,
                        args.bank_Ne, args.bank_sectors))
        L_vals = args.bank_L if args.bank_L is not None else [args.L]
        U_vals = args.bank_U if args.bank_U is not None else [args.U]
        t_vals = args.bank_t if args.bank_t is not None else [args.t]
        sec = args.bank_sectors or ("full" if args.full_hamiltonian else "sz0")
        sectors = {"sz0": [True], "full": [False], "both": [True, False]}[sec]

        jobs = []
        for L in L_vals:
            if args.bank_Ne is not None:
                fillings = args.bank_Ne
            else:
                fillings = [args.Ne if args.Ne is not None else L]
            for Ne in fillings:
                if not 0 < Ne <= 2 * L:
                    print(f"  [skip] L={L} Ne={Ne}: filling outside 0 < Ne <= 2L")
                    continue
                if max(2, args.min_n) > L:
                    print(f"  [skip] L={L}: --min-n {args.min_n} exceeds L")
                    continue
                for U in U_vals:
                    for t in t_vals:
                        for sp in sectors:
                            if sp and Ne % 2:
                                continue      # odd Ne has no S_z=0 sector
                            jobs.append((L, Ne, float(U), float(t), sp))
        if not jobs:
            parser.error("--build-bank: the requested grid is empty")

        if sweeping:
            n_per = {L: max(0, (min(args.bank_max_n or L, L) - max(2, args.min_n) + 1))
                     for L in L_vals}
            print(f"\n  bank grid   : {len(L_vals)} L x {len(U_vals)} U x "
                  f"{len(t_vals)} t x {len(sectors)} sector(s), mode={args.mode}")
            print(f"  banks to do : {len(jobs)}")
            print(f"  records/T   : {sum(n_per[L] for L, *_ in jobs)}"
                  f"   (x number of --bank-T values at ingestion)")
            if args.bank_dry_run:
                for L, Ne, U, t, sp in jobs:
                    print(f"    L={L} Ne={Ne} U={U} t={t} "
                          f"{'sz0' if sp else 'full'}")
                print()
                return

        t0 = _time.time()
        built = failed = 0
        built_paths: list = []          # exactly the banks this run touched
        for i, (L, Ne, U, t, sp) in enumerate(jobs, 1):
            if sweeping:
                print(f"\n[{_time.strftime('%H:%M:%S')}] ({i}/{len(jobs)}) "
                      f"L={L} Ne={Ne} U={U} t={t} "
                      f"{'sz0' if sp else 'full'} mode={args.mode}", flush=True)
            try:
                info = build_or_load_rdm_bank(
                    L=L, t=t, U=U, mode=args.mode,
                    n_electrons=Ne, n_min=max(2, args.min_n),
                    t_prime=args.t_prime, t2=args.t2, t_perp=args.t_perp,
                    average_site_clusters=args.thermal_average_site_clusters,
                    n_workers=args.n_workers,
                    spin_preserving=sp,
                    use_pkl_cache=True,
                    cache_version=args.cache_version,
                    force_recompute=(args.force_state_recompute
                                     or args.force_rdm_recompute),
                    save_pkl=True,
                    verbose=not sweeping,
                    n_max=args.bank_max_n,
                )
            except Exception as e:                  # keep a long sweep alive
                failed += 1
                print(f"    [FAIL] {type(e).__name__}: {e}", flush=True)
                continue
            built += 1
            built_paths.append(info["cache_path"])
            status = ("loaded from cache" if info["loaded_from_cache"]
                      else "built + saved")
            if sweeping:
                print(f"    {status}  ({built} ok, {failed} failed, "
                      f"{(_time.time()-t0)/60:.1f} min)", flush=True)
            else:
                print(f"\n  RDM bank {status}")
                print(f"  n range     : {info['n_values'][0]}..{info['n_values'][-1]}")
                print(f"  eigenstates : {len(info['energies'])}")
                print(f"  bank PKL    : {info['cache_path']}")
        if sweeping:
            print(f"\n  DONE: {built} banks ({failed} failed) in "
                  f"{(_time.time()-t0)/60:.1f} min")

        # ── optional stage 2: GCG-label every bank RDM and store everything ──
        if args.bank_label:
            from hubbard_gme.rdm import get_project_root, get_rdm_pkl_results_dir
            _repo = str(get_project_root())
            _qml = os.path.join(_repo, "gme_qml")
            if _qml not in sys.path:
                sys.path.insert(0, _qml)
            from ml_ingest_pkls import (ingest_banks, _load_done,
                                        _append_manifest, _anchor)

            ds_root = _anchor(args.bank_label_outdir
                              or os.path.join(_repo, "gme_qml", "data_for_ML_v2"),
                              _repo)
            out_ing = ds_root / "ingested"
            records_dir = out_ing / "records"
            records_dir.mkdir(parents=True, exist_ok=True)
            manifest = out_ing / "manifest.csv"
            done = _load_done(manifest)
            n_before = len(done)   # ingest_banks mutates `done` as it goes

            print(f"\n  ── labelling stage ─────────────────────────────────")
            print(f"  dataset root : {out_ing}")
            print(f"  already have : {n_before} labelled records (skipped)")
            print(f"  temperatures : {args.bank_label_T}")
            # --max-iter/--tol default to None in main.py (the solver picks its
            # own); ingest_banks needs concrete numbers, so fall back to the
            # same defaults ml_ingest_pkls.py uses.
            _max_iter = args.max_iter if args.max_iter is not None else 200
            _tol = args.tol if args.tol is not None else 1e-9
            print(f"  solver       : n_restarts={args.n_restarts} "
                  f"max_iter={_max_iter} tol={_tol} "
                  f"k_max_extra={args.k_max_extra}")
            rows = ingest_banks(
                get_rdm_pkl_results_dir(), records_dir, out_ing, done,
                T_list=list(args.bank_label_T),
                cutoff=args.thermal_energy_cutoff,
                n_restarts=args.n_restarts, max_iter=_max_iter,
                tol=_tol, k_max_extra=args.k_max_extra, verbose=True,
                # label exactly the grid this command built, not the whole cache
                only_banks=built_paths,
            )
            _append_manifest(rows, manifest)
            print(f"\n  labelled {len(rows)} NEW records "
                  f"(total now {n_before + len(rows)})")
            print(f"  records   → {records_dir}")
            print(f"  manifest  → {manifest}")
            print(f"  convergence sidecars → {out_ing / 'convergence'}")
        elif sweeping:
            print("\n  Banks hold RDMs only — no GME labels. To label them, "
                  "re-run with --bank-label, or:")
            print("    python gme_qml/ml_ingest_pkls.py --source banks "
                  "--bank-T 0 0.1 0.25 0.5 1 2 "
                  "--n-restarts 8 --max-iter 150 --k-max-extra 2")
        print()
        return

    # ── Thermal nBRDM mode ─────────────────────────────────────────────────
    if args.thermal:
        if args.T_steps < 1:
            parser.error("--T-steps must be at least 1")
        if args.T_max < args.T_min:
            parser.error("--T-max must be greater than or equal to --T-min")
        if not args.thermal_all_states and args.thermal_energy_cutoff <= 0:
            parser.error("--thermal-energy-cutoff must be positive")

        n_min_thermal = max(2, args.min_n)
        if n_min_thermal != args.min_n:
            print("  Thermal analysis uses n >= 2; raising --min-n to 2.")
        if n_min_thermal > args.L:
            parser.error("thermal --min-n cannot exceed --L")

        T_values = np.linspace(args.T_min, args.T_max, args.T_steps)
        thermal_cutoff = None if args.thermal_all_states else args.thermal_energy_cutoff
        thermal_max_iter = 300 if args.max_iter is None else args.max_iter
        thermal_tol = 1e-9 if args.tol is None else args.tol

        result = run_thermal_gme_analysis(
            L=args.L,
            t=args.t,
            U=args.U,
            T_values=T_values,
            mode=args.mode,
            n_electrons=args.Ne,
            t_prime=args.t_prime,
            t2=args.t2,
            t_perp=args.t_perp,
            n_min=n_min_thermal,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            solver=args.thermal_solver,
            n_restarts=args.n_restarts,
            max_iter=thermal_max_iter,
            tol=thermal_tol,
            k_max_extra=args.k_max_extra,
            verbose=True,
            thermal_energy_cutoff=thermal_cutoff,
            use_pkl_cache=args.use_pkl_cache,
            cache_version=args.cache_version,
            force_recompute=args.force_thermal_recompute,
            save_pkl=args.save_thermal_pkl,
            n_workers=args.n_workers,
            average_site_clusters=args.thermal_average_site_clusters,
            spin_preserving=not args.full_hamiltonian,
            T_view_max=args.T_view_max
        )

        print(f"\n  Thermal plot : {result['plot_path']}")
        if result.get("cache_path"):
            status = "loaded" if result.get("loaded_from_cache") else "saved"
            print(f"  Thermal PKL  : {result['cache_path']} ({status})")
        print()
        return

    # ── Excited singlet mode ───────────────────────────────────────────────
    # Combined with --t-scan / --u-scan, --excited-singlet is a MODIFIER that
    # tells the scan which state to use, so let the scan handler take over.
    if args.excited_singlet and not (args.t_scan or args.u_scan or args.v_scan):
        n_min_es = max(2, args.min_n)
        if n_min_es > args.L:
            parser.error("--min-n cannot exceed --L for excited-singlet analysis")
        es_max_iter = 300 if args.max_iter is None else args.max_iter
        es_tol      = 1e-9 if args.tol     is None else args.tol
        es_cutoff   = None if args.thermal_all_states else args.thermal_energy_cutoff

        T_values_es = None
        if args.T_sweep:
            if args.T_steps < 1:
                parser.error("--T-steps must be at least 1")
            T_values_es = np.linspace(args.T_min, args.T_max, args.T_steps)

        result = run_excited_singlet_gme(
            L=args.L, t=args.t, U=args.U,
            mode=args.mode, n_electrons=args.Ne,
            n_min=n_min_es,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            T_values=T_values_es,
            k=args.singlet_k,
            solver=args.thermal_solver,
            n_restarts=args.n_restarts,
            max_iter=es_max_iter,
            tol=es_tol,
            k_max_extra=args.k_max_extra,
            verbose=True,
            thermal_energy_cutoff=es_cutoff,
            use_pkl_cache=args.use_pkl_cache,
            cache_version=args.cache_version,
            force_recompute=args.force_thermal_recompute,
            save_pkl=args.save_thermal_pkl,
            n_workers=args.n_workers,
            spin_preserving=not args.full_hamiltonian,
            T_view_max=args.T_view_max
        )
        print(f"\n  Singlet bar plot : {result['bar_path']}")
        if result.get("pkl_path"):
            status = "loaded" if result.get("loaded_from_cache") else "saved"
            print(f"  Singlet PKL      : {result['pkl_path']} ({status})")
        if result.get("T_results", {}).get("plot_path"):
            print(f"  Singlet T plot   : {result['T_results']['plot_path']}")
        print()
        return

    # ── t-dependence scan mode ─────────────────────────────────────────────
    if args.t_scan:
        ts_max_iter = 300 if args.max_iter is None else args.max_iter
        ts_tol      = 1e-9 if args.tol     is None else args.tol
        if args.t_scan_min <= 0:
            parser.error("--t-scan-min must be positive")
        if args.t_scan_max <= args.t_scan_min:
            parser.error("--t-scan-max must be greater than --t-scan-min")
        t_values = np.linspace(args.t_scan_min, args.t_scan_max, args.t_scan_steps)

        result = run_t_dependence_gme(
            L=args.L, t_values=t_values, U=args.U,
            V= args.V,
            mode=args.mode, n_electrons=args.Ne,
            n_min=max(2, args.min_n),
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            solver=args.thermal_solver,
            n_restarts=args.n_restarts,
            max_iter=ts_max_iter,
            tol=ts_tol,
            k_max_extra=args.k_max_extra,
            verbose=True,
            use_pkl_cache=args.use_pkl_cache,
            cache_version=args.cache_version,
            force_recompute=args.force_thermal_recompute,
            save_pkl=args.save_thermal_pkl,
            spin_preserving=not args.full_hamiltonian,
            # --excited-singlet turns the scan into a first-excited-singlet
            # scan; the singlet is re-located at every grid point
            excited_singlet=args.excited_singlet,
            singlet_k=args.singlet_k,
        )
        print(f"\n  t-scan plot : {result['plot_path']}")
        if result.get("pkl_path"):
            status = "loaded" if result.get("loaded_from_cache") else "saved"
            print(f"  t-scan PKL  : {result['pkl_path']} ({status})")
        print()
        return

    # ── 3D surface mode: C_GME(t/U, T) ─────────────────────────────────────
    if args.tT_surface:
        surf_max_iter = 300 if args.max_iter is None else args.max_iter
        surf_tol      = 1e-9 if args.tol     is None else args.tol
        if args.t_scan_min <= 0:
            parser.error("--t-scan-min must be positive")
        if args.t_scan_max <= args.t_scan_min:
            parser.error("--t-scan-max must be greater than --t-scan-min")
        if args.tT_surface_T_steps < 2:
            parser.error("--tT-surface-T-steps must be at least 2")
        t_values = np.linspace(args.t_scan_min, args.t_scan_max, args.t_scan_steps)
        T_surf   = np.linspace(max(0.0, args.T_min), args.T_max,
                               args.tT_surface_T_steps)

        surf = run_tT_surface_gme(
            L=args.L, t_values=t_values, T_values=T_surf, U=args.U,
            mode=args.mode, n_electrons=args.Ne,
            n_min=max(2, args.min_n),
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            solver=args.thermal_solver,
            n_restarts=args.n_restarts,
            max_iter=surf_max_iter,
            tol=surf_tol,
            k_max_extra=args.k_max_extra,
            thermal_energy_cutoff=(
                None if args.thermal_all_states
                else args.thermal_energy_cutoff
            ),
            verbose=True,
            n_workers=args.n_workers,
            average_site_clusters=args.thermal_average_site_clusters,
            spin_preserving=not args.full_hamiltonian,
            use_pkl_cache=args.use_pkl_cache,
            cache_version=args.cache_version,
            force_recompute=args.force_thermal_recompute,
            save_pkl=args.save_thermal_pkl,
            n_max=args.tT_surface_max_n,
        )
        print(f"\n  t-T 3D surface plot : {surf['plot_path']}")
        if surf.get("pkl_path"):
            status = "loaded" if surf.get("loaded_from_cache") else "saved"
            print(f"  t-T surface PKL     : {surf['pkl_path']} ({status})")
        print()
        return

    # ── U-dependence scan mode ─────────────────────────────────────────────
    if args.u_scan:
        us_max_iter = 300 if args.max_iter is None else args.max_iter
        us_tol      = 1e-9 if args.tol     is None else args.tol
        # Negative U (attractive Hubbard) is physically valid, so it is allowed.
        if args.u_scan_max <= args.u_scan_min:
            parser.error("--u-scan-max must be greater than --u-scan-min")
        U_values = np.linspace(args.u_scan_min, args.u_scan_max, args.u_scan_steps)

        result = run_U_dependence_gme(
            L=args.L, U_values=U_values, t=args.t,
            V= args.V,
            mode=args.mode, n_electrons=args.Ne,
            n_min=max(2, args.min_n),
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            solver=args.thermal_solver,
            n_restarts=args.n_restarts,
            max_iter=us_max_iter,
            tol=us_tol,
            k_max_extra=args.k_max_extra,
            verbose=True,
            use_pkl_cache=args.use_pkl_cache,
            cache_version=args.cache_version,
            force_recompute=args.force_thermal_recompute,
            save_pkl=args.save_thermal_pkl,
            spin_preserving=not args.full_hamiltonian,
            # --excited-singlet turns the scan into a first-excited-singlet
            # scan; the singlet is re-located at every grid point
            excited_singlet=args.excited_singlet,
            singlet_k=args.singlet_k,
        )
        print(f"\n  U-scan plot : {result['plot_path']}")
        if result.get("pkl_path"):
            status = "loaded" if result.get("loaded_from_cache") else "saved"
            print(f"  U-scan PKL  : {result['pkl_path']} ({status})")
        print()
        return

    # ── V-dependence scan mode (extended Hubbard: fix t & U, sweep V) ───────
    if args.v_scan:
        vs_max_iter = 300 if args.max_iter is None else args.max_iter
        vs_tol      = 1e-9 if args.tol     is None else args.tol
        if args.v_scan_max <= args.v_scan_min:
            parser.error("--v-scan-max must be greater than --v-scan-min")
        V_values = np.linspace(args.v_scan_min, args.v_scan_max, args.v_scan_steps)

        result = run_V_dependence_gme(
            L=args.L, V_values=V_values, t=args.t, U=args.U,
            mode=args.mode, n_electrons=args.Ne,
            n_min=max(2, args.min_n),
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            solver=args.thermal_solver,
            n_restarts=args.n_restarts,
            max_iter=vs_max_iter,
            tol=vs_tol,
            k_max_extra=args.k_max_extra,
            verbose=True,
            use_pkl_cache=args.use_pkl_cache,
            cache_version=args.cache_version,
            force_recompute=args.force_thermal_recompute,
            save_pkl=args.save_thermal_pkl,
            spin_preserving=not args.full_hamiltonian,
            excited_singlet=args.excited_singlet,
            singlet_k=args.singlet_k,
        )
        print(f"\n  V-scan plot : {result['plot_path']}")
        if result.get("pkl_path"):
            status = "loaded" if result.get("loaded_from_cache") else "saved"
            print(f"  V-scan PKL  : {result['pkl_path']} ({status})")
        print()
        return

    # ── Scan mode ──────────────────────────────────────────────────────────
    if args.scan_U or args.scan_Ut:
        U_values  = list(np.linspace(args.U_min,  args.U_max,  args.U_steps))  if args.scan_U  else None
        Ut_values = list(np.linspace(args.Ut_min, args.Ut_max, args.Ut_steps)) if args.scan_Ut else None

        run_scan_pipeline(
            L=args.L, t=args.t,
            U_values=U_values, Ut_values=Ut_values,
            mode=args.mode, n_electrons=args.Ne,
            n_body=args.n_body, k_states=args.k_states,
            output_dir=args.output_dir,
            landscape_analysis=args.landscape,
            landscape_no_2d=args.landscape_no_2d,
            **_topo_kwargs(args),
        )
        return

    # ── Single-point mode ──────────────────────────────────────────────────
    if ne_actual == args.max_n and not args.no_verify:
        print(f"\n  *** n = N detected: Ne={ne_actual} == max-n={args.max_n}")
        print(f"  *** Supervisor verification test will run automatically.")
        if not args.landscape:
            print(f"  *** Use --landscape to also generate landscape analysis.")
        print(f"  *** Use --no-verify to skip verification.\n")

    if args.no_pdf:
        result = run_pure_analysis(
            L=args.L, t=args.t, U=args.U, mode=args.mode,
            n_electrons=args.Ne, max_n_body=args.max_n,
            min_n_body=args.min_n,
            output_dir=args.output_dir, generate_pdf=False,
            verify_n_equals_N=not args.no_verify,
            continuous_optimization=args.continuous,
            continuous_real_only=args.real_only,
            landscape_analysis=args.landscape,
            landscape_no_2d=args.landscape_no_2d,
            convex_roof_all=args.convex_roof_all,
            n_workers=args.n_workers,
            state_idx=args.state_idx,
            **_solver_kwargs(args),
            **_cache_kwargs(args),
            **_topo_kwargs(args),
        )
        si = result["system_info"]
        print(f"\n{'─'*50}")
        print(f"  E0 = {result['E0']:.8f}")
        print(f"  E1 = {result['E1']:.8f}   gap = {result['gap']:.6f}")
        print(f"  <S2> = {si['S2']:.4f}   S = {si['S']:.3f}")
        print(f"  Spin state : {si['spin_label']}")
        print(f"  <D>/site   : {si['double_occ_per_site']:.4f}")

        # Print n=N verification results if they ran
        nN = result.get("n_equals_N_results", {})
        if nN:
            print(f"\n{'─'*50}")
            print(f"  n = N verification results:")
            for n_ord, vr in nN.items():
                status = "VERIFIED ✓" if vr["is_valid"] else "FAILED ✗"
                print(f"    n={n_ord}  C_exact={vr['C_exact']:.6f}  "
                      f"C_LB={vr['C_LB']:.6f}  diff={vr['difference']:.2e}  {status}")

        # Print continuous optimization results if they ran
        cont = result.get("continuous_results", {})
        if cont:
            print(f"\n{'─'*50}")
            print(f"  Continuous optimization results:")
            for sites, cr in cont.items():
                print(f"    sites={sites}  C_LB={cr['C_LB']:.6f}  "
                      f"method={cr['method']}  gain={cr['C_LB']-cr['C_LB_basis']:+.6f}")

        print(f"{'─'*50}\n")
    else:
        run_full_GME_pipeline(
            L=args.L, t=args.t, U=args.U, mode=args.mode,
            n_electrons=args.Ne, max_n_body=args.max_n,
            min_n_body=args.min_n,
            output_dir=args.output_dir,
            verify_n_equals_N=not args.no_verify,
            continuous_optimization=args.continuous,
            continuous_real_only=args.real_only,
            landscape_analysis=args.landscape,
            landscape_no_2d=args.landscape_no_2d,
            convex_roof_all=args.convex_roof_all,
            n_workers=args.n_workers,
            state_idx=args.state_idx,
            **_solver_kwargs(args),
            **_cache_kwargs(args),
            **_topo_kwargs(args),
        )


if __name__ == "__main__":
    main()
