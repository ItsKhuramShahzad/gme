# Verify.md — Hubbard GME Verification Protocol

Generated/updated: 2026-06-29
Project: `hubbard_gme`  (package version 0.3.0)

---

## 1. Purpose

This document defines how to verify the Hubbard GME project across all its modes:

```text
single-point ground-state GME (+ n=N verification)
RDM cache load/recompute
per-method GME solver cache (stiefel / gcg / stiefel_gcg)
effective settings + full GME audit metadata
thermal GME, excited-singlet GME, t-scan, U-scan
PDF source visualization
parallel convex-roof execution
```

The goal is not only that the code runs, but that the scientific outputs are reliable,
reproducible, and auditable.

Cache convention is **v1**. NOTE: `scripts/verify_project.py` still defaults
`--cache-version` to `v2`; pass `--cache-version v1` to the cache-mode checks so they
find the current PKLs.

Environment: runs use the conda `qchem` env. If the package is not installed
(`pip install -e .`), prefix pytest / verify commands with `PYTHONPATH=src`.

---

## 2. Verification Levels

```text
Level 0   File structure check
Level 1   Python syntax check
Level 2   Dependency check
Level 3   Import check
Level 4   Lightweight pytest (fast unit tests)
Level 5   RDM cache verification
Level 6   GME PKL verification
Level 7   CLI smoke test
Level 8   PDF/artifact verification
Level 9   Per-method cache reload/recompute test
Level 10  Thermal / excited-singlet / scan task verification
Level 11  Scientific numerical consistency
Level 12  Parallel worker verification
Level 13  Full production run + slow integration tests
```

---

## 3. Level 0 — File Structure Check

```bash
python scripts/verify_project.py --mode static
```

Required files:

```text
main.py
src/hubbard_gme/pipeline.py
src/hubbard_gme/convex_roof.py
src/hubbard_gme/rdm.py
src/hubbard_gme/visualization.py
src/hubbard_gme/thermal_pipeline.py
Plan.md / Progress.md / Verify.md
scripts/verify_project.py
```

Pass: all required files exist; no critical project folder is missing.

---

## 4. Level 1 — Python Syntax Check

```bash
python -m py_compile \
  main.py main_mol.py \
  src/hubbard_gme/pipeline.py \
  src/hubbard_gme/convex_roof.py \
  src/hubbard_gme/rdm.py \
  src/hubbard_gme/visualization.py \
  src/hubbard_gme/thermal_pipeline.py
```

Pass: no SyntaxError / IndentationError. (A `SyntaxWarning` for an escape sequence in
`paper_result.py` is benign.)

---

## 5. Level 2 — Dependency Check

```bash
python scripts/verify_project.py --mode deps
```

Core: numpy, scipy, matplotlib, tqdm, openfermion, pytest.
Optional: cvxpy, pyscf, openfermionpyscf (molecular path).

Pass: core dependencies import; optional ones may warn but should be documented.

---

## 6. Level 3 — Import Check

```bash
python scripts/verify_project.py --mode imports
```

Modules: `hubbard_gme.pipeline`, `.convex_roof`, `.rdm`, `.visualization`, `.gme`,
`.metrics`, `.thermal_pipeline`, `.hamiltonian`.

Pass: no import error in the `qchem` environment.

---

## 7. Level 4 — Lightweight / Fast Tests

```bash
python scripts/verify_project.py --mode pytest-light
```

or directly (fast unit tests, sub-second):

```bash
PYTHONPATH=src pytest \
  tests/test_metrics.py tests/test_hamiltonian.py tests/test_rdm.py \
  tests/test_gme.py tests/test_meyerwallach.py \
  tests/test_verification_contract.py -q
```

Pass: all fast unit tests pass (expected ~77). Documentation/contract sections present;
Python files parse.

NOTE: `test_canonical_parameters.py`, `test_convex_roof.py`, `test_mixed_states.py`,
`test_measure_param.py`, and `test_gradient.py` run real convex-roof optimisation and
take minutes — verify them at Level 13, not here.

---

## 8. Level 5 — RDM Cache Verification

First run (small solver settings just to populate caches):

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute --n-restarts 1 --k-max-extra 0 --max-iter 20
```

Expected files: `results/rdm_pkl_results/rdm_cache/*.pkl` (24 for full n=3..6 GS RDMs;
filenames like `rdm_v1_L6_Ne6_U4_t1_mode-uniform_state-gs_n3_sites-0-1-2_nso12.pkl`).

Verify:

```bash
python scripts/verify_project.py --mode cache-rdm --cache-version v1
```

Each RDM PKL should contain: `meta`, `rho`, `metrics`, `trace`, `purity`, `entropy`,
`rank_numerical`, `lambda_min`, `lambda_max`.

Scientific checks: `trace ≈ 1`; `rho = rho†` within tolerance; eigenvalues
`≥ -numerical_tolerance`; `0 ≤ purity ≤ 1`; `entropy ≥ 0`.

---

## 9. Level 6 — GME PKL Verification

After at least one convex-roof cluster finishes:

```bash
python scripts/verify_project.py --mode cache-gme --cache-version v1
```

Each GME PKL should contain: `meta`, `system_state_ref`, `rdm_ref`, `integrity`,
`cluster_audit_summary`, `result_full`, `decompositions`.

`result_full`: `C_LB`, `C_LB_supervisor`, `C_LB_best`, `C_UB`, `C_stiefel`, `C_gcg`,
`C_stiefel_gcg`, `stiefel`, `gcg`, `stiefel_gcg`, `solver_runtime`, `solver_success`,
`solver_cache_status`, `effective_settings`.

Per-solver: `rank`, `k_converged`, `k_records`, `best_record`, `n_restarts_requested`,
`n_restarts_effective`, `k_max_extra`, `tol`, `tol_k`, `max_iter_effective`,
`k_values_tried`.

Decomposition: `p`, `C_GME`, `psi`, `psi_hash_sha256`, `psi_norm`.

Pass: at least one GME PKL passes integrity checks; no required field missing; solver C
values non-negative. (`scripts/inspect_gme_pkl.py` dumps one PKL for inspection.)

---

## 10. Level 7 — CLI Smoke Tests

```bash
python scripts/verify_project.py --mode smoke --smoke-size tiny
python scripts/verify_project.py --mode smoke --smoke-size small
```

Pass: small system runs quickly; Hamiltonian builds; RDMs compute/load; GME pipeline
enters the correct stage.

---

## 11. Level 8 — Artifact Verification

```bash
python scripts/verify_project.py --mode artifacts --artifact-dir outputs
```

Checks: PDFs/PKLs exist and are non-empty with recent timestamps.

Expected single-point PDFs: `GME_3body_*`, `GME_4body_*`, `GME_5body_*`, `GME_6body_*`.

```text
n=3,4,5 : convex-roof block present for mixed RDMs
n=6 pure: pure-state formula / n=N verification, no convex-roof block
```

---

## 12. Level 9 — Per-Method Cache Test

```bash
# Recompute all
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute              # expect stiefel=NEW gcg=NEW stiefel_gcg=NEW

# Reload all
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default reload                 # expect stiefel=PKL gcg=PKL stiefel_gcg=PKL

# Recompute one method only
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default reload --gcg-cache recompute   # expect stiefel=PKL gcg=NEW stiefel_gcg=PKL
```

Pass: PDF source column and GME PKL `solver_cache_status` match the command; only the
requested solver is recomputed.

---

## 13. Level 10 — Thermal / Excited-Singlet / Scan Tasks

Run all four tasks (or `bash run_all_tasks.sh`):

```bash
# Task 1 — thermal GME
python main.py --thermal --L 6 --U 10.0 --t 1.0 --T-min 0 --T-max 5 --T-steps 40 \
  --min-n 2 --thermal-solver gcg --n-restarts 10 --max-iter 200 --k-max-extra 4 \
  --workers 6 --use-pkl-cache --output-prefix L6_thermal_t1_U10

# Task 2 — excited-singlet GME (bar + T-sweep)
python main.py --excited-singlet --T-sweep --L 6 --U 10.0 --t 1.0 \
  --T-min 0.01 --T-max 5 --T-steps 30 --min-n 2 --thermal-solver gcg --use-pkl-cache \
  --output-prefix L6_singlet_t1_U10

# Task 3 — t-scan
python main.py --t-scan --L 6 --U 10.0 --t-scan-min 0.05 --t-scan-max 2.0 \
  --t-scan-steps 20 --min-n 2 --thermal-solver gcg --use-pkl-cache \
  --output-prefix L6_tscan_U10

# Task 4 — U-scan
python main.py --u-scan --L 6 --t 1.0 --u-scan-min 1.0 --u-scan-max 20.0 \
  --u-scan-steps 20 --min-n 2 --thermal-solver gcg --use-pkl-cache \
  --output-prefix L6_uscan_t1
```

Pass criteria:

```text
Each task writes one PDF (prefixed by --output-prefix) and a result PKL.
Thermal caches populate under results/rdm_pkl_results/thermal_runs/{,job_cache/,rdm_bank_cache/}.
Task 4 (U-scan) currently has NO output on disk — confirm it now produces a PDF/PKL.
A second run with --use-pkl-cache reloads and regenerates the PDF quickly.
```

Physics sanity:

```text
C_GME(n=L, T→0) -> exact ground-state value
Thermal C_GME(n,T) decreases sensibly with T (entanglement sudden death possible)
Excited singlet correctly identified (S=0 above triplet manifold)
t-scan / U-scan: no t- or U-dependence for n=L; n<L curves vary with t/U
```

---

## 14. Level 11 — Scientific Numerical Checks

For every RDM:

```text
trace_error = |Tr(rho)-1|;  hermitian_error = max|rho-rho†|
eigvals_min ≥ -1e-8;  purity valid;  entropy ≥ -small tolerance
```

Pure `n=N`: `Tr(rho^2) ≈ 1`, pure C_GME formula used, n=N verification reported.

Mixed RDMs: `Tr(rho^2) < 1`, convex roof used, `C_LB ≤ C_solver ≤ C_UB` where available,
C values non-negative.

Solver convergence: `k_records`, `k_history`, `k_converged`, `restart_log`, `history`.

Decomposition: `sum_i p_i ≈ 1`, `p_i ≥ 0`, `‖psi_i‖ ≈ 1`, component `C_GME ≥ 0`.

---

## 15. Level 12 — Parallel Worker Verification

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute --workers 8
```

Expected: `Workers: 8`, convex roof runs per cluster, progress bar updates per completed
cluster (`0/18` for some time is normal). Pass: no multiprocessing crash; at least one
cluster completes and saves its GME PKL.

---

## 16. Level 13 — Full Production Run + Slow Tests

```bash
python main.py --L 6 --U 4.0 --min-n 3 --max-n 6 --use-pkl-cache \
  --gme-cache-default recompute --n-restarts 10 --k-max-extra 2 --workers 8 \
  --output-dir outputs/prod_L6_U4
```

Then the slow integration tests (allow several minutes each):

```bash
PYTHONPATH=src pytest tests/test_canonical_parameters.py tests/test_convex_roof.py \
  tests/test_mixed_states.py -q
```

Pass: 18 mixed clusters processed; 24 RDMs saved/loaded; `GME_3body..6body` PDFs
generated; full-run + per-cluster GME PKLs saved; verify-script cache/artifact checks
pass; slow tests pass.

---

## 17. Common Failure Diagnosis

### RDMs recompute on second run
Check: updated `pipeline.py` installed? `--use-pkl-cache` set? `--force-rdm-recompute`
absent? Filenames match cache version (`v1`) and state label?

### cache-mode verify finds nothing
`verify_project.py` defaults to `--cache-version v2`; pass `--cache-version v1`.

### Convex roof did not run for n=6
Correct for the pure ground-state `n=6` (pure formula). Convex roof runs for `n=6` only
if that density matrix is mixed.

### GME reload does not happen
Check: a previous cluster finished and saved a GME PKL? Same `cache-version`? Identical
`method/optimizer/n_restarts/k_max_extra/tol`? `--gme-cache-default reload` requested?

### Task 4 (U-scan) produces no output
Run the Level 10 Task 4 command and confirm `run_U_dependence_gme` writes its PDF/PKL.

---

## 18. Final Verification Checklist

```text
[ ] Code compiles
[ ] Static verification passes
[ ] Imports pass in qchem environment
[ ] Fast unit tests pass (~77)
[ ] RDM cache reload verified (v1)
[ ] GME cache reload/recompute verified
[ ] n=N pure handling verified
[ ] Single-point PDFs generated for n=3,4,5,6
[ ] Thermal / excited-singlet / t-scan / U-scan PDFs generated (Tasks 1–4)
[ ] PKLs contain full metadata
[ ] C values physically valid; k convergence + restart stability checked
[ ] runtime per solver recorded
[ ] Slow convex-roof integration tests pass
[ ] final command and environment recorded in Progress.md
```
