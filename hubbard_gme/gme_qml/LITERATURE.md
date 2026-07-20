# Existing work: Machine Learning for Entanglement Detection & Quantification

Working notes for the QML/ML phase. Primary source: the review you found —

> **"Entanglement Detection and Quantification Through Machine Learning: A
> Comprehensive Review"**, Brazilian Journal of Physics (2026) 56:25,
> DOI 10.1007/s13538-025-01947-y
> (local copy: `~/PhDWork/Hubbard Model and Condense Physics/papers/ML for GME paper/`)

Sections 1–3 below summarize that review. Section 4 lists adjacent work from
the broader literature (verify exact citations before quoting in a paper).
Section 5 is the gap analysis: where our project is new.

---

## 1. The problem as the field frames it

- **Detection** = the separability problem: decide entangled vs separable.
  Analytically solved only in small dimensions (PPT/Peres–Horodecki is
  necessary & sufficient only for 2×2 and 2×3); NP-hard in general, so ML is
  used as a fast approximate decision procedure.
- **Quantification** = estimating a measure (concurrence, negativity,
  entanglement entropy, GME measures). Ground truth comes from expensive
  numerics — exactly our GCG situation.
- The review warns: reported accuracies across papers are NOT comparable —
  every work uses different state ensembles, dimensions, and features.

## 2. Methodological families (review taxonomy, §4)

| Family | Idea | Representative result (as reported in the review) |
|--------|------|--------------------------------------------------|
| **CHA / BCHA** | Approximate the (convex) separable set by the convex hull of sampled separable pure states; membership via linear programming; bagging ensemble on top | 2-qubit error 2.55% at m=10⁴ hull points; BCHA reaches <3% error with ~10× fewer hull points; two-qutrit needs m=10⁵ |
| **SVM** | Hyperplanes ≈ entanglement witnesses; kernels for nonlinear boundaries; feature choice is the game | good 2-qubit accuracy; scales poorly to high dim |
| **KNN** | classify by nearest labeled states in feature space | simple baseline; used with collective-measurement features |
| **RBM (SNNS)** | "separable neural network states": restrict an RBM ansatz to separable form; failure to fit ⇒ entangled. Doubles as a quantifier | Werner-state entanglement entropy to ε<10⁻⁵; detects bound entanglement; distinguishes W vs GHZ classes; channel-capacity bounds |
| **MLP** | plain feed-forward nets on various features | >90% accuracy in early proofs-of-concept (density-matrix features, PPT labels) |
| **Deep (CNN/GAN)** | structured inputs (tomograms, correlation images) | used for larger systems / incomplete data |

**Feature lessons from the review (directly relevant to us):**
- Bell/CHSH-correlator features: few features can rival full tomography —
  random measurement directions beat fixed ones; 26 random-direction features
  ≈ 63-feature tomographic benchmark (~1% vs ~0% mismatch) for 3 qubits.
- More features → better, but with diminishing returns; an NN with 12
  witness-type features beat the best analytical criterion (92% vs 86%).
- **Boundary states (near the separable/entangled border) are the universal
  weakness** — every method degrades there. For us: states with C_GME ≈ 0
  (high-T thermal, crossover U/t) need dedicated evaluation.
- Multipartite structure can be learned: NNs classify entanglement
  *intactness/depth* of noisy GHZ blocks with ≥99.9% accuracy from just four
  fixed measurements — i.e., GME-structure classification is feasible.

## 3. Quantification (regression) in the review

- Mostly classification; quantification appears via (i) SNNS/RBM reconstruction
  distance (entropy, channel capacities), (ii) NN regression of concurrence /
  negativity from features or tomograms. Regression of a **convex-roof GME
  measure** (our C_GME) is essentially absent — labels are too expensive to
  mass-produce, which is precisely the asset our GCG pipeline + PKL caches give us.

## 4. Adjacent work outside the review (verify citations before use)

- **ML-assisted many-body entanglement measurement** — e.g. Gray, Banchi,
  Bayat, Bose, PRX 8, 011006 (2018): NN estimates entanglement (negativity-type
  measures) of many-body subsystems from a few moments of the partially
  transposed RDM — closest in spirit to our "physics features → measure" route.
- **Deep learning of entanglement from incomplete tomography** — e.g. Koutný
  et al., Sci. Adv. 9 (2023): NN regression of negativity from partial
  measurement data.
- **Entanglement classification with NNs for multiqubit families** (GHZ/W/
  cluster + noise): several works 2018–2024 (e.g. Asif et al., Sci. Rep. 2023,
  using GME-concurrence-style labels for 3-qubit families).
- **Quantum ML side**: variational entanglement detection / quantum kernels;
  measurable-on-hardware entanglement quantities (e.g. concentratable
  entanglement via SWAP tests) — candidate bridge for our Stage-6 QML phase.
- **ML on Hubbard physics** (not entanglement): phase classification from
  QMC configurations (Ch'ng et al. 2017 and successors) — establishes that
  Hubbard observables are learnable; nobody connects this to GME measures.

## 5. Gap analysis — where our project is new

1. **Physical states, not random states.** Nearly all reviewed work trains on
   synthetic ensembles (random density matrices, Werner/Horodecki/GHZ + noise).
   Our states are *n-body RDMs of an interacting fermionic model* — a
   physically structured, SU(2)-symmetric, low-dimensional manifold inside
   state space. Learning on this manifold is both easier (structure) and more
   defensible (the states one actually meets in condensed matter).
2. **GME of reduced states of a many-body system** — the object
   C_GME(Γ⁽ⁿ⁾) as a function of (n, t/U, T, sector) has, to our knowledge, no
   ML treatment at all. The closest work regresses bipartite negativity of
   subsystems (Gray et al.); genuine *multipartite* convex-roof measures for
   Hubbard RDMs are unexplored.
3. **Expensive-label regime.** Our GCG labels cost minutes–hours each; the
   caches already hold thousands. "ML surrogate for an expensive convex-roof
   optimizer" is itself a contribution (cf. how ML force fields replaced DFT
   in loops).
4. **Thermal dimension.** C_GME(T) labels from the thermal pipeline give a
   *temperature-controlled path* from entangled (T=0) to separable (high T) —
   a natural, physical way to populate the boundary region that synthetic
   datasets sample poorly.
5. **Deployment loop.** ML pre-screen + GCG verification of uncertain points
   (Stage 7) is a concrete, publishable speed-up for GME surface scans.

## 5b. Verified recent works (web search, 2026-07-13) — alignment vs our strategy

Our strategy, for reference: **physical n-body RDMs** (Hubbard + molecules) as
data; **GCG convex-roof C_GME** as label; **detection + estimation**; thermal
states for the boundary; group-split by Hamiltonian; extrapolation across
(L, T, U/t); later MLP + QML.

| Paper (verified) | What they do | Aligns with us | Differs from us |
|---|---|---|---|
| Luo, Liu & Zhang, PRA 108, 052424 (2023) — "Detecting GME via machine learning" [arXiv:2311.17548] | SVM + semi-supervised S4VM detect GME of **3-qubit random density matrices** | GME detection task; supervised framing | random synthetic states; detection only; no measure regression; 3 qubits only |
| PRA 110, 042412 (2024) — "GME verification with CNNs" [arXiv:2508.13463] | CNN (+squeeze-excitation) classifies GME for 4–6 qubit states and GHZ-diagonal up to 20 qubits, SDP-generated labels | GME, larger qubit counts, NN | special state families (GHZ-diagonal); detection only; no physical Hamiltonian |
| Gray, Banchi, Bayat & Bose, PRL 121, 150503 (2018) [arXiv:1709.04923] | NN learns map from **moments of partially-transposed RDM → logarithmic negativity** of many-body subsystems | closest in spirit: many-body subsystem RDMs, measure regression, few features | **bipartite** negativity (not GME); no convex roof; no thermal sweep |
| Entropy 27, 185 (2025) — "Quantifying unknown multiqubit entanglement using ML" | NN quantifies multipartite entanglement from **local measurement statistics**, linear number of measurements, noise-robust | multipartite quantification; measurement-friendly features | synthetic states; not a convex-roof C_GME; no many-body Hamiltonian |
| Wehrl-moment ANN [arXiv:2205.15095] | ANN regresses **geometric measure of entanglement** (SDP labels) of symmetric multiqubit states from Wehrl moments, incl. mixed | regression of a GME-type measure with expensive labels | symmetric-state families only; geometric measure (not convex-roof concurrence); no physical model |
| Sci. Rep. 14 (2024) — "Quantifying entanglement for unknown states via ANN" | measurement expectation values as features → entanglement-measure labels | quantifier framing identical to our Task B | random/unknown states; bipartite-oriented |
| ANN-enhanced 3-qubit GME detection on NMR hardware, QIP 24 (2025) [arXiv:2409.19739] | experimental 3-qubit GME detection with ANN | shows hardware relevance of GME-ML | 3 qubits, detection only |
| Neural quantum states for **entanglement-depth certification** from randomized Pauli measurements [arXiv:2512.13121] (Dec 2025) | NQS + randomized measurements certify entanglement depth | multipartite structure, scalable | certifies depth, not a value of C_GME; no convex roof |
| Huang et al., "Provably efficient ML for quantum many-body problems" [arXiv:2106.12627] | theory: classical shadows + ML predict ground-state properties incl. entanglement entropies | theoretical backing that our map ρ→property is learnable | generic framework, no GME |
| Chen, Lin & Wei, "Certifying unknown GME by neural networks" [arXiv:2210.13837] (verified 2026-07-15) | NN certifies GME of unknown states from local **measurement statistics**; reports that dropping cross-correlation terms hurts accuracy | supports our finding that **coherence/correlation features are decisive** (cf. Sz-block features) | synthetic states; certification, not convex-roof regression |
| Laskowski, Markiewicz, Paterek & Żukowski, PRA (2011) [arXiv:1110.4108] | nonlinear **correlation-tensor criteria** for genuine multiqubit entanglement | analytic precedent: structured correlations detect GME — motivates correlation-structure features (Hamming-resolved coherences, bipartition packs) | witness criteria, no ML, no measure value |
| Jungnitsch, Moroder & Gühne, PRL 106, 190502 (2011) | **PPT-mixture** SDP relaxation certifying genuine multipartite entanglement (genuine negativity) | candidate additional CHECK label / bipartition-feature motivation | certification bound, not the convex-roof value |

**Bottom line of the check (2026-07):** still **no work** found that (i) trains
on RDMs of an interacting fermionic Hamiltonian (or a real molecule), (ii)
labels with a **convex-roof GME concurrence** (GCG), (iii) does both detection
and regression, and (iv) uses a thermal path to populate the boundary. Every
GME-ML paper uses synthetic/special state families; every many-body paper uses
bipartite measures. Gaps 1–5 in Section 5 remain valid. The molecular
extension (Benzene CAS RDMs, canonical vs PM-localized) appears untouched in
the literature.

## 6. What to cite for our methods sections

- GME measure & convex roof: the project's existing references
  (Ma et al. GME-concurrence; Wootters concurrence; convex-roof numerics —
  already in `../papers/`).
- ML framing: the 2026 review above + Gray et al. 2018 + one incomplete-
  tomography regression paper.
- Baselines: standard (scikit-learn RF/GBM/MLP; logistic/ridge).
