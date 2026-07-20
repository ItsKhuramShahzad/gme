# Theory Reference

## 1. The Hubbard Model

The 1-D Hubbard Hamiltonian on a ring of L sites:

```
H = -t Σ_{⟨j,k⟩,σ} (c†_{jσ} c_{kσ} + h.c.)  +  U Σ_j n_{j↑} n_{j↓}
```

| Symbol | Meaning |
|--------|---------|
| t | Nearest-neighbour hopping amplitude |
| U | On-site Coulomb repulsion |
| c†_{jσ} / c_{jσ} | Fermionic creation / annihilation at site j, spin σ |
| n_{jσ} = c†_{jσ} c_{jσ} | Number operator |

**Spin-orbital index mapping:** p = 2j + s, where s = 0 (↑) or 1 (↓).

---

## 2. n-Body Reduced Density Matrix

For a subsystem S = {i₁, ..., iₙ} and wavefunction |ψ⟩:

```
ρ[a, b] = ⟨ψ| c†_{p₁} ··· c†_{pₙ}  c_{qₙ} ··· c_{q₁} |ψ⟩
```

where pₖ = 2·iₖ + σₐ[k] and qₖ = 2·iₖ + σᵦ[k].

**Dimension:** 2ⁿ × 2ⁿ

| n | Matrix size |
|---|-------------|
| 1 | 2 × 2 |
| 2 | 4 × 4 |
| 3 | 8 × 8 |
| 4 | 16 × 16 |

---

## 3. Quantum Metrics

### Purity
```
γ = Tr(ρ²) = Σᵢ λᵢ²
```
- γ = 1 → pure state
- γ < 1 → mixed state

### Von Neumann Entropy
```
S(ρ) = -Tr(ρ log ρ) = -Σᵢ λᵢ log λᵢ
```
- S = 0 → pure state
- S = log(d) → maximally mixed (d = dimension)

### Hilbert-Schmidt Distance
```
D_HS(ρ_A, ρ_B) = ||ρ_A - ρ_B||_F = √Tr[(ρ_A - ρ_B)²]
```

### Trace Distance
```
D_T(ρ_A, ρ_B) = ½ Σᵢ |λᵢ(ρ_A - ρ_B)|
```

---

## 4. Genuine Multipartite Entanglement (GME)

### Bipartitions
For n spins, all unique bipartitions A|B (with 0 ∈ A) are enumerated.

### Separability criterion
```
Tr(ρ_A²) = 1  ⟺  ρ = ρ_A ⊗ ρ_B   (product state on cut A|B)
```

### C_GME (Troiani et al. 2019)
For a **pure** n-spin state:
```
C_GME = √(2 · min_A (1 − Tr(ρ_A²)))
       = √(2 · (1 − max_A Tr(ρ_A²)))
```

### Decision logic
| Condition | Conclusion |
|-----------|------------|
| γ = 1, all Tr(ρ_A²) < 1 | **GME exists**, C_GME > 0 |
| γ = 1, some Tr(ρ_A²) = 1 | **No GME** (separable cut) |
| γ < 1 | **Mixed state** (convex-roof extension needed) |

### Lower bound via coherence witness
```
I_{x,y}(ρ) = |ρ_{x,y}| − Σ_A √(ρ_{x^A, x^A} · ρ_{y^A, y^A})

C_GME ≥ 2 · I_{x,y}(ρ)
```

---

## 5. Reference

F. Troiani, C. Angeli, A. Secchi, and S. Pittalis,
*"Genuine multipartite entanglement from many-electron systems,"*
Phys. Rev. A 99, 022323 (2019).
