# hubbard_gme

**Genuine Multipartite Entanglement (GME) analysis for the Hubbard model — and for small
molecular active spaces — using n-body Reduced Density Matrices.**

Developed by Khuram Shahzad.
Package version: **0.3.0**

---

## Overview

This package implements the full pipeline from:

1. **Exact diagonalisation** of the 1-D Hubbard Hamiltonian (5 topologies)
2. **n-body RDM extraction** for arbitrary site clusters
3. **GME analysis** — pure-state `C_GME`, bipartition purities, paper/supervisor lower
   bounds, and **convex-roof extensions** for mixed RDMs (3 solvers)
4. **Temperature- and parameter-dependent studies** — thermal GME `C_GME(n,T)`,
   excited-singlet GME, and `t`-/`U`-scans
5. **PDF report generation** with heatmaps, metric tables, and solver provenance
6. **Auditable PKL caching** for system states, RDMs, and convex-roof solver runs

The theoretical foundation is based on:
> F. Troiani, C. Angeli, A. Secchi, and S. Pittalis,
> *"Genuine multipartite entanglement from many-electron systems,"*
> Phys. Rev. A **99**, 022323 (2019).

The thermal convex-roof solver (GCG) follows Röthlisberger, Lehmann, and Loss (2009).

---

## Project Structure

```
hubbard_gme/
├── main.py                      # Hubbard CLI entry point (single-point, scan, thermal, scans)
├── main_mol.py                  # Molecular GME CLI entry point
├── run_all_tasks.sh             # Regenerate Task 1–4 PDFs (thermal / singlet / t-scan / u-scan)
├── setup.py
├── requirements.txt
├── enviornment.yml              # conda env (general)
├── enviornment_qchem.yml        # conda env used for runs (`qchem`)
│
├── Plan.md                      # Project plan
├── Progress.md                  # Progress log
├── Verify.md                    # Verification protocol
├── README.md
│
├── src/
│   └── hubbard_gme/
│       ├── __init__.py
│       ├── hamiltonian.py       # Hubbard H (5 topologies) + diagonalisation
│       ├── rdm.py               # n-body RDM computation + RDM PKL cache
│       ├── metrics.py           # Purity, entropy, HS/trace distances
│       ├── gme.py               # Pure-state C_GME, paper + supervisor lower bounds
│       ├── convex_roof.py       # Convex-roof solvers: stiefel, gcg, stiefel_gcg (+SDP)
│       ├── landscape.py         # Optimum-landscape analysis (discrete / 1D / 2D)
│       ├── thermal_pipeline.py  # Thermal GME, excited-singlet, t-/U-scans
│       ├── spin.py              # ⟨Sz⟩, ⟨S²⟩, double occupancy
│       ├── scan.py              # U and U/t parameter-scan engine
│       ├── plots.py             # Scan comparison plots / scan PDF
│       ├── visualization.py     # Heatmaps and multi-page PDF reports
│       ├── pipeline.py          # End-to-end Hubbard drivers
│       ├── utils.py             # Basis labels, helper plots
│       │
│       ├── molecules_data.py    # Molecule / CAS active-space database
│       ├── molecular.py         # Molecular Hamiltonian, GS, RDMs, spin/D
│       ├── molecular_pipeline.py# End-to-end molecular analysis
│       └── molecular_visualization.py
│
├── scripts/
│   ├── verify_project.py        # Verification ladder (static → smoke → cache checks)
│   └── inspect_gme_pkl.py       # Inspect a cached GME PKL
│
├── tests/                       # pytest suite (unit + convex-roof integration)
├── notebooks/                   # Interactive walkthroughs
├── docs/
│   └── theory.md                # Concise theory reference
├── outputs/                     # Generated PDF/PKL reports land here
└── results/
    └── rdm_pkl_results/         # Central PKL cache (see "Caching" below)
```

---

## Installation

```bash
git clone <your-repo>
cd hubbard_gme
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

The runs in this project use the conda `qchem` environment:

```bash
conda env create -f enviornment_qchem.yml
conda activate qchem
```

If the package is not installed with `pip install -e .`, run with `PYTHONPATH=src`.

---

## Usage

### Single-point analysis (CLI)

```bash
# Default: L=4, t=1.0, U=4.0, half-filling, uniform ring
python main.py

# Custom parameters with n=3..6 RDMs
python main.py --L 6 --t 1.0 --U 8.0 --mode uniform --Ne 6 --min-n 3 --max-n 6

# Parallel convex roof on mixed RDMs (8 workers), reusing PKL cache
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache --workers 8

# Add the optimum-landscape analysis (n=N pure states only)
python main.py --L 6 --U 4.0 --landscape

# Add continuous I_Phi optimisation for a tighter lower bound
python main.py --L 6 --U 4.0 --continuous
```

### Topologies

| Mode        | Description                                            | Extra flag        |
|-------------|--------------------------------------------------------|-------------------|
| `uniform`   | 1-D ring, PBC, all bonds = t (default)                 | —                 |
| `open`      | 1-D open chain, no PBC                                  | —                 |
| `nnn`       | Ring + next-nearest-neighbour hopping t′               | `--t-prime`       |
| `ssh`       | Su–Schrieffer–Heeger alternating t1/t2 (open BC)       | `--t2`            |
| `ladder`    | Two-leg ladder, leg = t, rung = t⊥ (even L)            | `--t-perp`        |
| `dimerized`, `heisenberg` | Legacy modes                             | —                 |

Caching (`--use-pkl-cache`) currently supports the plain topologies (uniform/open).

### Temperature- and parameter-dependent analyses

These are driven by `thermal_pipeline.py`:

```bash
# Task 1 — Thermal GME: C_GME(n, T)
python main.py --thermal --L 6 --U 10.0 --t 1.0 --T-min 0 --T-max 5 --T-steps 40 \
               --min-n 2 --thermal-solver gcg --use-pkl-cache --workers 6

# Task 2 — Excited-singlet GME at T=0 (bar chart), optionally with a T-sweep
python main.py --excited-singlet --L 6 --U 10.0 --t 1.0 --min-n 2 \
               --thermal-solver gcg --use-pkl-cache --workers 6
python main.py --excited-singlet --T-sweep --L 6 --U 10.0 --t 1.0 \
               --T-min 0.01 --T-max 5 --T-steps 30 --min-n 2 --use-pkl-cache

# Task 3 — t-scan: C_GME(n) vs t at T=0 (fixed U)
python main.py --t-scan --L 6 --U 10.0 --t-scan-min 0.05 --t-scan-max 2.0 \
               --t-scan-steps 20 --min-n 2 --use-pkl-cache

# Task 4 — U-scan: C_GME(n) vs U at T=0 (fixed t)
python main.py --u-scan --L 6 --t 1.0 --u-scan-min 1.0 --u-scan-max 20.0 \
               --u-scan-steps 20 --min-n 2 --use-pkl-cache

# Task 4b — Extended Hubbard: on-site U + nearest-neighbour V (V Σ_j n_j n_{j+1}).
#   U-scan at a FIXED V/t (C_GME vs U/t at V/t = 5, 10, 20; run once per V):
python main.py --u-scan --L 6 --Ne 6 --t 1.0 --u-scan-min 0 --u-scan-max 15 \
               --u-scan-steps 16 --V 5 --full-hamiltonian --min-n 2 \
               --use-pkl-cache --output-prefix V5

# Task 4c — V-scan: fix t and U, sweep V — shows the Mott -> CDW transition
#   at V ~ U/2 (short-range C_GME collapses across the transition):
python main.py --v-scan --L 6 --Ne 6 --t 1.0 --U 10 --v-scan-min 0 \
               --v-scan-max 15 --v-scan-steps 16 --full-hamiltonian --min-n 2 \
               --use-pkl-cache --output-prefix Vscan_U10
```

`--V` defaults to 0.0 (plain Hubbard); it is a nearest-neighbour density-density
term `V Σ_j n_j n_{j+1}` (n_j = n_{j↑} + n_{j↓}, PBC ring). V is saved in every
scan PKL (in `system_info` and the filename) and shown in the plot title.

`run_all_tasks.sh` runs all four tasks for L=6, U=10, t=1 and writes the PDFs to
`outputs/`. With `--use-pkl-cache` a second run loads the cached results and
regenerates the PDFs almost instantly.

### Parameter scans (eigenstate sweep)

```bash
# Scan over U (fixed t), n=3 RDMs, 4 eigenstates per point
python main.py --scan-U --L 4 --U-min 0.5 --U-max 16 --U-steps 20 --n-body 3

# Scan over U/t with landscape analysis at each point
python main.py --scan-Ut --L 4 --mode nnn --t-prime 0.3 --landscape
```

### Molecular analysis

```bash
python main_mol.py --mol Benzene --cas 0 --cr-method stiefel
python main_mol.py --mol Methylene --cr-method none   # RDMs + metrics only
```

### Python API

```python
import sys; sys.path.insert(0, 'src')
from hubbard_gme.pipeline import run_full_GME_pipeline

result = run_full_GME_pipeline(
    L=6, t=1.0, U=4.0, mode='uniform',
    n_electrons=6, min_n_body=3, max_n_body=6,
    use_pkl_cache=True, n_workers=8, output_dir='outputs',
)
# result contains: E0, E1, gap, psi0, psi1, rdm_pure, rdm_ex, metrics, system_info
```

### Notebook

See `notebooks/` for interactive walkthroughs.

---

## Caching

With `--use-pkl-cache`, artifacts are stored under `results/rdm_pkl_results/`:

```
state_cache/                 # system state (Hamiltonian + eigenstates)
rdm_cache/                   # n-body RDMs (+ trace/purity/entropy/rank metadata)
gme_runs/job_cache/          # per-cluster convex-roof solver results
thermal_runs/                # thermal / excited-singlet result PKLs
thermal_runs/job_cache/      # per (T, n) thermal GME jobs
thermal_runs/rdm_bank_cache/ # thermal RDM banks
```

**Cache version is `v1`** (the current default; cache filenames are tagged `..._v1_...`).
The three convex-roof solvers — `stiefel`, `gcg`, `stiefel_gcg` — can be reloaded or
recomputed independently:

```bash
--gme-cache-default {reload,recompute}     # default: reload
--stiefel-cache     {default,reload,recompute}
--gcg-cache         {default,reload,recompute}
--stiefel-gcg-cache {default,reload,recompute}
--force-gme-recompute                      # hard global recompute
--force-rdm-recompute / --force-state-recompute
```

Each cached GME PKL stores full audit metadata: RDM integrity (hash, trace, purity,
rank, eigenvalues, Hermiticity error), effective solver settings (`n_restarts`,
`k_max_extra`, `tol`, `max_iter`, k-convergence), solver runtime/success, and the
pure-state decomposition vectors. See `scripts/inspect_gme_pkl.py`.

---

## Running Tests

```bash
# Fast unit tests
PYTHONPATH=src pytest tests/test_metrics.py tests/test_hamiltonian.py \
    tests/test_rdm.py tests/test_gme.py tests/test_meyerwallach.py \
    tests/test_verification_contract.py -q

# Full suite (includes slow convex-roof integration tests — minutes)
PYTHONPATH=src pytest tests/ -q
```

Verification ladder:

```bash
python scripts/verify_project.py --mode static
python scripts/verify_project.py --mode imports
python scripts/verify_project.py --mode cache-rdm --cache-version v1
python scripts/verify_project.py --mode all      --cache-version v1
```

---

## Key Physics Parameters

| Parameter   | Symbol | Typical range          |
|-------------|--------|------------------------|
| Sites       | L      | 2–8                    |
| Hopping     | t      | 1.0 (energy unit)      |
| Interaction | U      | 0 → 20                 |
| Electrons   | Ne     | L (half-filling)       |
| RDM order   | n      | min-n .. max-n (1–L)   |
| Temperature | T      | 0 → 5 (thermal mode)   |

---

## Module Summary

| Module | Responsibility |
|--------|----------------|
| `hamiltonian` | Build H (5 topologies), diagonalise with `eigsh` |
| `rdm` | Compute ⟨ψ\|c†…c†c…c\|ψ⟩ for all clusters; RDM PKL cache |
| `metrics` | Purity γ, entropy S(ρ), D_HS, D_T |
| `gme` | Pure-state C_GME, bipartitions, paper + supervisor lower bounds |
| `convex_roof` | Convex-roof solvers (stiefel, gcg, stiefel_gcg, optional SDP) |
| `landscape` | Optimum-landscape analysis (discrete / continuous 1D / 2D) |
| `thermal_pipeline` | Thermal GME, excited-singlet GME, t-/U-scans |
| `spin` | ⟨Sz⟩, ⟨S²⟩, double occupancy → DataFrame |
| `scan` | U and U/t parameter-scan engine |
| `visualization` / `plots` | Heatmaps, multi-page and scan PDF reports |
| `pipeline` | `run_pure_analysis`, `run_full_GME_pipeline`, `run_scan_pipeline` |
| `molecular*` | Molecular Hamiltonian, RDMs, convex roof, PDF reports |
| `utils` | Basis labels, eigenvector probability plots |

---

## Reference

F. Troiani, C. Angeli, A. Secchi, and S. Pittalis,
*"Genuine multipartite entanglement from many-electron systems,"*
Phys. Rev. A **99**, 022323 (2019).
