# `data_for_ML/ingested/` — GCG records mined from existing PKL caches

ML records built by `gme_qml/ml_ingest_pkls.py` from results you already
computed — no recomputation. Kept **separate** from the native `../records/` +
`../manifest.csv` so those are never touched.

## GCG ONLY
The training target `C_target` is **always the GCG convex-roof value**. Stiefel
results are **not** ingested and **no Stiefel-named field is written**, even
though the standalone GME PKLs also store Stiefel / `stiefel_gcg` / `C_LB_best`
columns — those are ignored here.

> The native dataset in `../records/` is **Stiefel-based**. Keep it separate;
> do not mix Stiefel labels with these GCG labels.

## Sources linked

| `source` | From | ρ comes from | Label (`C_target`) |
|----------|------|--------------|--------------------|
| `gme_runs` | `results/.../gme_runs/job_cache/*.pkl` | linked `rdm_cache/` RDM | `result_full.C_gcg` (records with no GCG value are skipped) |
| `surface` | `results/.../thermal_runs/tTsurface_*.pkl` | **reconstructed** from cached `rdm_bank/` + Boltzmann weights at T | `C_surface[n][t,T]` (GCG) |
| `bank` | `results/.../rdm_bank/*.pkl` | ρ from the bank (T=0 GS, or T>0 mixed) | **GCG run now** on that ρ (`--source banks`) |

`gme_runs`/`surface` *convert* already-labeled results. `bank` *generates* new
GCG labels by running the solver on each bank's RDMs — the way to grow the
dataset across all L. Options: `--bank-T 0 0.5 1 2`, `--n-restarts`, `--max-iter`,
`--k-max-extra`. Use `--min-n 2` banks (they cover n=2…L; higher-nmin banks are
redundant). `--source all` = gme + surface + banks.

Surface records are thermal (mixed) RDMs — one per (t, T, n), cluster-averaged;
`thermal=True`, `T` set. GME records are T=0 single-cluster RDMs (`thermal=False`).

## npz schema (all `solver="gcg"`)
```
rho_real, rho_imag, eigvals, config_json,
C_target (= GCG), C_gcg, C_pure, lower_bound_super, C_UB, purity, is_pure,
target_in_bracket, source, solver, T, thermal
```
No `C_stiefel*` / `C_LB_best` keys.

## Regenerate / update
```bash
cd <repo>                       # dir containing main.py
python gme_qml/ml_ingest_pkls.py --source both
#   --source gme | surface | both
#   --results-root results/rdm_pkl_results     (default)
#   --outdir gme_qml/data_for_ML               (ingested/ created under it)
```
Resumable: re-running skips records already in the manifest. Surface ingestion
needs the matching `rdm_bank/` files present (build via `--build-bank` or run the
surface first).

## Load
```python
import glob, numpy as np
files = glob.glob("gme_qml/data_for_ML/ingested/records/*.npz")
recs = [np.load(f, allow_pickle=True) for f in files]   # rho_real/imag + C_target (GCG)
```
Do **not** pool these with the native Stiefel `../records/` unless you
deliberately want mixed-solver labels.
