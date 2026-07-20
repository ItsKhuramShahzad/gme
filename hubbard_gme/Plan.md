# Plan.md — Hubbard GME Project Plan

Generated/updated: 2026-06-29
Project: `hubbard_gme`
Main goal: compute and verify genuine multipartite entanglement (GME) for Hubbard-model
reduced density matrices (RDMs) — for ground states, excited singlets, thermal states,
and along `t`/`U` scans — with reliable caching, reproducibility, and auditable
convex-roof solver metadata.

---

## 1. Scientific Objective

The project studies the 1-D Hubbard model and computes GME-related quantities for site
clusters (n-body RDMs). There are now four analysis families:

1. **Single-point / ground-state GME** (`main.py`, `pipeline.py`)
2. **Thermal GME** `C_GME(n, T)` (`--thermal`, `thermal_pipeline.py`)
3. **Excited-singlet GME** at T=0 and over a T-sweep (`--excited-singlet`)
4. **Parameter scans** `C_GME(n)` vs `t` or `U` at T=0 (`--t-scan`, `--u-scan`)

Reference target for the ground-state case:

```bash
L = 6
Ne = 6
U = 4.0
t = 1.0
mode = uniform
state_idx = 0  # ground state
n = 3, 4, 5, 6
```

Reference target for the thermal/scan tasks (see `run_all_tasks.sh`):

```bash
L = 6, Ne = 6, t = 1.0, U = 10.0, mode = uniform
solver = gcg, n_restarts = 10, max_iter = 200, k_max_extra = 4, min_n = 2
```

Scientific rule for every RDM:

```text
If Tr(rho^2) ≈ 1:
    use pure-state C_GME formula / n=N verification

If Tr(rho^2) < 1:
    use convex-roof methods
```

Important: do **not** run convex roof on pure `n=N` states unless explicitly testing the
convex-roof solvers. For the normal ground-state calculation, `n=6` is the full pure
state and should use the pure-state formula.

---

## 2. Topologies

`main.py --mode` supports 5 topologies (plus 2 legacy):

```text
uniform   1-D ring, all bonds = t, PBC  (default)
open      1-D open chain, no PBC
nnn       ring + next-nearest-neighbour hopping t'   (--t-prime)
ssh       Su-Schrieffer-Heeger alternating t1/t2     (--t2, open BC)
ladder    two-leg ladder, leg=t rung=t_perp          (--t-perp, even L)
dimerized / heisenberg  (legacy)
```

PKL caching is currently wired for the plain topologies (uniform/open).

---

## 3. Caching Features Expected in the Code

Cache root: `results/rdm_pkl_results/`. **Current cache version is `v1`** (the default;
filenames are tagged `..._v1_...`).

### 3.1 System-state cache  →  `state_cache/`

```text
First run : compute Hamiltonian/eigenstate, save system-state PKL
Second run: reload system-state PKL unless --force-state-recompute
```

### 3.2 RDM cache  →  `rdm_cache/`

```text
First run : compute RDMs and save
Second run: load existing RDMs, do not recompute unless --force-rdm-recompute
```

For `python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache` expect 24 RDMs:

```text
n=3: 6 clusters   n=4: 6 clusters   n=5: 6 clusters   n=6: 6 clusters   Total: 24
```

### 3.3 Per-method convex-roof solver cache  →  `gme_runs/job_cache/`

Three solvers, controlled independently:

```text
stiefel        Stiefel-manifold optimisation
gcg            Generalized Conjugate Gradient on U(k)
stiefel_gcg    hybrid (Stiefel warm-start + GCG)
```

CLI controls:

```bash
--gme-cache-default {reload,recompute}     # default: reload
--stiefel-cache     {default,reload,recompute}
--gcg-cache         {default,reload,recompute}
--stiefel-gcg-cache {default,reload,recompute}
--force-gme-recompute                      # hard global recompute override
--no-reuse-gme-pkl                         # recompute all clusters
--no-save-gme-pkl                          # do not write GME PKLs
```

Examples:

```bash
# Recompute all three solvers
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute

# Reload all by default, but recompute only GCG
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default reload --gcg-cache recompute
```

### 3.4 Thermal / excited-singlet / scan caches  →  `thermal_runs/`

```text
thermal_runs/                 thermal + excited-singlet + scan result PKLs
thermal_runs/job_cache/       per (T, n) thermal GME solver jobs
thermal_runs/rdm_bank_cache/  thermal RDM banks (per T)
```

Controls:

```bash
--thermal-solver {gcg,supervisor}   # default: gcg
--force-thermal-recompute           # ignore existing thermal PKL
--no-save-thermal-pkl               # do not write thermal PKL
```

### 3.5 Effective solver settings must be saved

The saved GME PKLs must not store vague values such as `iterauto`/`tolauto`. They must
store resolved/effective values:

```text
n_restarts_requested / n_restarts_effective
k_max_extra / k_values_tried / k_converged
tol / tol_k
max_iter_input / max_iter_effective / max_iter_policy / max_iter_by_k
optimizer / measure
```

Default expected values if the user does not override:

```text
n_restarts = 10
k_max_extra = 2
tol = 1e-9
tol_k = 1e-6

max_iter:
    stiefel      = adaptive by k / n_params
    gcg          = 300
    stiefel_gcg  = 300
```

(Thermal mode raises the gcg `max_iter` default to 300; `run_all_tasks.sh` passes 200.)

### 3.6 Full GME audit metadata

Each per-cluster GME PKL should save enough metadata to audit and reconstruct the run:

```text
meta / system_state_ref / rdm_ref / integrity / cluster_audit_summary
elapsed / result_full / decompositions
```

Required RDM integrity data:

```text
rho_hash_sha256 / rho_shape / rho_dtype / trace / purity / rank / rank_threshold
eigvals / eigvals_nonzero / lambda_min / lambda_max
hermitian_max_error / trace_error / metrics_snapshot
```

Required solver audit data:

```text
solver_runtime / solver_success / elapsed_by_stage / solver_cache_status
active_solvers / solver_effective_settings
```

Required decomposition audit data:

```text
component_index / p / C_GME / psi / psi_shape / psi_dtype
psi_hash_sha256 / psi_norm
```

---

## 4. Correct Run Strategy

### 4.1 Normal ground-state run

Pure formula for `n=N`, convex roof only for mixed RDMs:

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute
```

Expected convex-roof message:

```text
[3/5] Running convex-roof analysis on 18 MIXED clusters
```

Why 18? `n=3,4,5` each contribute 6 mixed clusters; `n=6` is the pure full state and is
skipped by convex roof.

### 4.2 Reload test

After one complete successful run:

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default reload
```

Expected: system-state loaded_from_cache=True, RDM cache loaded count > 0,
GME solver source `S:PKL, G:PKL, SG:PKL`.

### 4.3 Per-method recompute test

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default reload --gcg-cache recompute
```

Expected: `stiefel -> PKL`, `gcg -> NEW`, `stiefel_gcg -> PKL`.

### 4.4 Thermal / excited / scan tasks (Tasks 1–4)

Driven by `run_all_tasks.sh` for L=6, U=10, t=1, gcg solver. Individual commands:

```bash
# Task 1 — thermal GME C_GME(n,T)
python main.py --thermal --L 6 --U 10.0 --t 1.0 --T-min 0 --T-max 5 --T-steps 40 \
  --min-n 2 --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4 \
  --workers 6 --use-pkl-cache --output-prefix L6_thermal_t1_U10

# Task 2 — excited-singlet GME (bar chart + optional T-sweep)
python main.py --excited-singlet --L 6 --U 10.0 --t 1.0 --min-n 2 \
  --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4 \
  --workers 6 --use-pkl-cache --output-prefix L6_singlet_t1_U10

# Task 3 — t-scan C_GME(n) vs t at T=0
python main.py --t-scan --L 6 --U 10.0 --t-scan-min 0.05 --t-scan-max 2.0 \
  --t-scan-steps 20 --min-n 2 --thermal-solver gcg --use-pkl-cache

# Task 4 — U-scan C_GME(n) vs U at T=0
python main.py --u-scan --L 6 --t 1.0 --u-scan-min 1.0 --u-scan-max 20.0 \
  --u-scan-steps 20 --min-n 2 --thermal-solver gcg --use-pkl-cache
```

---

## 5. Output Plan

### 5.1 PDF outputs

Single-point run — one PDF per RDM order `n=3,4,5,6`:

```text
n=3,4,5 : RDM heatmap, RDM metrics, convex-roof solver values,
          solver source status, k convergence, restart information
n=6     : RDM heatmap, RDM metrics, pure-state C_GME, n=N verification
          (no convex-roof block unless n=6 is mixed)
```

Thermal/scan runs — one PDF per task, prefixed by `--output-prefix`:

```text
Task 1 : C_GME(n, T) vs T
Task 2 : C_GME(n) bar chart at T=0 (+ C_GME(n, T) if --T-sweep)
Task 3 : C_GME(n) vs t (T=0)
Task 4 : C_GME(n) vs U (T=0)
```

### 5.2 PKL outputs

```text
results/rdm_pkl_results/state_cache/
results/rdm_pkl_results/rdm_cache/
results/rdm_pkl_results/gme_runs/job_cache/
results/rdm_pkl_results/thermal_runs/{,job_cache/,rdm_bank_cache/}
```

---

## 6. Verification Ladder

Run verification from light to heavy:

```bash
python scripts/verify_project.py --mode static
python scripts/verify_project.py --mode deps
python scripts/verify_project.py --mode imports
python scripts/verify_project.py --mode pytest-light
python scripts/verify_project.py --mode artifacts --artifact-dir outputs
python scripts/verify_project.py --mode cache-rdm --cache-version v1
python scripts/verify_project.py --mode cache-gme --cache-version v1
python scripts/verify_project.py --mode all      --cache-version v1
```

Note: `verify_project.py` still defaults `--cache-version` to `v2`; pass
`--cache-version v1` explicitly so the cache checks find the current PKLs.

Numerical smoke tests:

```bash
python scripts/verify_project.py --mode smoke --smoke-size tiny
python scripts/verify_project.py --mode smoke --smoke-size small
```

---

## 7. Milestones

### Milestone 1 — Code structure verified

```text
main.py, src/hubbard_gme exist
pipeline.py / convex_roof.py / rdm.py / visualization.py / thermal_pipeline.py import
```

### Milestone 2 — RDM cache verified

```text
First run computes and saves RDMs; second run loads them
--force-rdm-recompute recomputes
RDM PKLs contain trace, purity, entropy, rank, eigenvalues
```

### Milestone 3 — Per-method GME cache verified

```text
Each solver reloads/recomputes independently; PDF shows solver source
PKL records solver_cache_status; partial recompute works
```

### Milestone 4 — Effective settings verified

```text
No ambiguous iterauto/tolauto in v1 metadata
tol, tol_k, max_iter_effective, k_values_tried, n_restarts_effective saved
```

### Milestone 5 — Thermal / scan tasks verified

```text
Tasks 1–4 produce their PDFs and result PKLs
Thermal job_cache + rdm_bank_cache populated
C_GME(n=L) → pure ground-state value as T → 0
```

### Milestone 6 — Scientific output verified

```text
RDM trace ≈ 1; Hermiticity error small; eigenvalues ≥ -tol
Pure n=N via pure formula; mixed n<N via convex roof
C_LB ≤ C_solver ≤ C_UB where applicable; C_GME values non-negative
PDF and PKL outputs generated
```

---

## 8. Known Risks and Mitigations

### Risk: interrupted run before GME PKL is saved

```text
RDM cache may already be saved.
GME job cache is saved only after a cluster/solver finishes.
Rerun with --gme-cache-default reload to reuse completed parts.
```

### Risk: old cache conflicts with new metadata

Bump the cache version tag (`--cache-version`); the current convention is `v1`.

### Risk: multiprocessing hides progress until a full cluster finishes

```text
tqdm updates per completed cluster, not per restart or per solver.
0/18 for several minutes can be normal.
```

### Risk: too-heavy full run

For debugging, reduce work: `--n-restarts 3 --k-max-extra 1 --max-iter 150`, then use
stronger settings for final runs.

---

## 9. Recommended Commands

### Debug (fast)

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute \
  --n-restarts 3 --k-max-extra 1 --max-iter 150 --workers 8 \
  --output-dir outputs/debug_L6_U4
```

### Production

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute \
  --n-restarts 10 --k-max-extra 2 --workers 8 \
  --output-dir outputs/prod_L6_U4
```

### All thermal / scan tasks

```bash
bash run_all_tasks.sh
```
