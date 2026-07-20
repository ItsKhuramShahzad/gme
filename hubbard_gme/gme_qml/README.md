# `gme_qml/` — Machine Learning for GME Detection & Estimation

Everything ML-related (classical ML and, later, quantum ML) for the Hubbard-GME
project lives in this folder. The physics engine stays in `../src/hubbard_gme/`;
this folder consumes it to build datasets and train surrogate models.

**Goal.** The GCG convex-roof computation of C_GME is accurate but expensive
(minutes per mixed RDM, hours per parameter sweep). We train ML models that,
given an n-body reduced density matrix Γ⁽ⁿ⁾ of the Hubbard model (or features
derived from it), perform:

| Task | Type | Output | Ground truth |
|------|------|--------|--------------|
| **A. Detection** | binary classification | GME present? (C_GME > ε) | GCG label / LB–UB certificates |
| **B. Estimation** | regression | the value of C_GME | C_GCG (verified best solver) |

The benchmark data source is the verified pipeline: L=6, Ne=6 ring, GCG
convex-roof on n-body RDMs, T=0 and thermal states, Sz=0 and full-Hamiltonian
sectors.

---

## Folder map

```
gme_qml/
├── README.md                  ← this file (pipeline + status)
├── LITERATURE.md              ← existing work: ML for entanglement (what's known)
├── ml_generate_ml_dataset.py  ← STAGE 1a: generate labeled records by direct computation
├── ml_ingest_pkls.py          ← STAGE 1b: mine existing PKL caches into records (free data)
├── ml_audit_dataset.py        ← STAGE 2 : dataset health checks (run before EVERY training)
├── train_baselines.py         ← STAGE 4 : classical baselines (detection + estimation)
├── data_for_ML/               ← the dataset
│   ├── manifest.csv           ← one row per record (all scalar features + labels)
│   ├── records/*.npz          ← full records: rho, eigvals, labels, config
│   ├── ingested/              ← records mined from PKL caches (kept separate)
│   ├── grid_summary.json      ← parameter grid provenance
│   └── run_metadata.json      ← run provenance log
└── models/                    ← trained model artifacts + metrics
```

---

## The pipeline (clear and cut)

```
 STAGE 0   Ground-truth engine                        [DONE — ../src/hubbard_gme]
           Hubbard H → eigenstates → n-body RDMs → C_GME via GCG convex roof.
           Benchmarked at L=6, Ne=6 over t, U, T; GCG verified as the best
           estimator for mixed-RDM GME.
                       │
 STAGE 1   Dataset generation                          [PARTIAL]
   1a. ml_generate_ml_dataset.py — sweeps (L, U/t or t, state, site-cluster),
       diagonalizes once per config, emits one record per RDM:
           ρ (real+imag), eigenvalues, purity, entropy, spin labels (S², Sz),
           double occupancy, energy, gap, C_pure, supervisor LB, UB,
           C_stiefel(best/mean/std over restarts), C_target.
   1b. ml_ingest_pkls.py — converts already-computed caches into records:
           gme_runs job caches (have C_gcg) + thermal surface PKLs
           (mixed Γ⁽ⁿ⁾(T) reconstructed from the RDM bank → labeled by C_GME(T)).
       Thermal records matter twice: they add MIXED states away from the
       ground state, and high-T states give C≈0 examples (class balance!).
                       │
 STAGE 2   Audit                                       [DONE — rerun before training]
           python ml_audit_dataset.py
           Checks: NaNs, bracket LB ≤ C ≤ UB, coverage by (L, n, U/t),
           label noise across restarts, pure/mixed split.
                       │
 STAGE 3   Features (two representations)
   F1 "raw"     : vectorized ρ (upper triangle, real+imag). Dimension = 4ⁿ
                  → varies with n → train ONE MODEL PER n (n=2..6).
   F2 "physics" : fixed-dimension vector usable across n:
                  eigenvalue spectrum (padded), purity, entropy, tr ρ², tr ρ³,
                  rank, n, L, U/t, S², Sz, d_occ, energy, gap.
                  → one global model; also the natural input for QML later
                  (low-dimensional, physically meaningful).
                       │
 STAGE 4   Models                                       [train_baselines.py]
   Detection : logistic regression → random forest / gradient boosting → MLP.
               Metrics: ROC-AUC, accuracy, and accuracy in the HARD BAND
               (records with C near the threshold — boundary states are the
               known weakness of all ML entanglement classifiers).
   Estimation: ridge → random forest / gradient boosting → MLP.
               Metrics: MAE, RMSE, R²; residuals vs U/t, purity, n.
               Physics baseline to beat: the supervisor analytical LB.
                       │
 STAGE 5   Honest evaluation (leakage control)
   * GROUP SPLIT by config_hash: all records from the same Hamiltonian
     (L, t, U, state) stay in the same fold. Random row-splits LEAK —
     neighbouring site-clusters of the same state are near-duplicates.
   * Extrapolation tests (the physics questions):
       train L ≤ 5      → test L = 6      (transfer across system size)
       train T = 0      → test thermal    (transfer across mixedness)
       train mid U/t    → test extremes   (transfer across coupling)
                       │
 STAGE 6   QML phase (after classical baselines exist)
   * Quantum kernels / variational classifiers on the F2 feature vector
     (few qubits: one feature per qubit via angle encoding).
   * Honest comparison vs Stage-4 classical models at equal feature budget.
   * Physically-motivated variant: features restricted to MEASURABLE
     observables (occupations, spin–spin correlators) → an
     experiment-friendly GME estimator.
                       │
 STAGE 7   Deployment back into the pipeline
   predict_gme(rho) → used as a fast pre-screen: ML predicts C_GME for every
   grid point; GCG recomputes only uncertain/boundary cases.
   → surface scans at a fraction of today's cost.
```

---

## Current status & known gaps

**Have (as of 2026-07-03):**
- 2610 records in the native manifest (L=3–6, uniform ring, vary-U + vary-t,
  ground + excited states, n=2..L), ~15 MB.
- Ingest path for GCG-labeled cache records + thermal surface states.
- Audit tooling.

**Gaps (ordered by priority):**
1. **Labels are Stiefel, not GCG.** `C_target` in the native dataset comes from
   `convex_roof_stiefel` (`target_source=stiefel`). We have since verified GCG
   is the better solver. → Fix: relabel/extend via 1b ingestion (C_gcg is in the
   caches) and/or add a `--label-solver gcg` mode to the generator.
2. **Audit flags 57/2610 (2.2%) bracket violations** (LB ≤ C ≤ UB broken) —
   consistent with Stiefel label noise; the relabel in (1) should clear most.
3. **Class imbalance for detection**: ground-state Hubbard RDMs are almost
   always entangled. Separable/near-zero examples come from high-T thermal
   states → run the surface ingestion on denser T grids.
4. **No full-Hamiltonian records yet** (all Sz=0 sector). The sector tag now
   exists in the caches, so ingestion picks it up automatically.
5. **No trained models yet** → `train_baselines.py` is the starting point.

---

## Quick start

```bash
cd /home/anjum/PhDWork/hubbard_gme

# 1. Mine everything already computed into ML records (no recomputation)
python gme_qml/ml_ingest_pkls.py --source both

# 2. Health-check the dataset
python gme_qml/ml_audit_dataset.py

# 3. Train the classical baselines (detection + estimation)
python gme_qml/train_baselines.py

# (optional) extend the native dataset, e.g. denser U/t grid at L=6
python gme_qml/ml_generate_ml_dataset.py --L 6 --scan-mode both --n-workers 8
```

See `LITERATURE.md` for what already exists in the field and where this work
is novel (short version: plenty of ML on random/GHZ/Werner states, almost
nothing on GME of *physical many-body reduced states*, and nothing we found on
Hubbard n-body RDMs — that is our niche).
