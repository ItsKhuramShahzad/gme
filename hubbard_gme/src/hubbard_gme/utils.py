"""
utils.py
--------
General-purpose helpers: basis-state labels, spin-label generators,
and a quick eigenvector probability viewer.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# ── Basis-label generators ──────────────────────────────────────────────────

def get_basis_labels(n_orbitals: int) -> list[str]:
    """Return all 2^n_orbitals binary bitstrings, e.g. ['00','01','10','11']."""
    return ["".join(bits) for bits in product("01", repeat=n_orbitals)]


def generate_spin_labels(n_sites: int) -> list[str]:
    """Return spin-arrow labels for an n-site subsystem, e.g. |↑↓⟩."""
    labels = []
    for spins in product(["↑", "↓"], repeat=n_sites):
        labels.append("|" + "".join(spins) + "⟩")
    return labels


# ── Quick wavefunction visualizer ───────────────────────────────────────────

def visualize_fixed_electron_subspace(
    eigvec: np.ndarray,
    n_sites: int,
    U: float,
    t: float,
    n_electrons: int,
    title: str = "Eigenvector Probability Distribution",
) -> None:
    """
    Bar chart of |⟨i|ψ⟩|² restricted to states with exactly *n_electrons*
    particles. Probabilities are renormalised inside the selected subspace.

    Parameters
    ----------
    eigvec     : Full Fock-space amplitude vector (length 2^(2*n_sites)).
    n_sites    : Number of lattice sites.
    U, t       : Hubbard parameters (displayed in title only).
    n_electrons: Filter to states with this occupation number.
    title      : Plot title prefix.
    """
    n_orbitals = 2 * n_sites
    probs = np.abs(eigvec) ** 2
    basis_labels = get_basis_labels(n_orbitals)

    selected_indices = [
        i for i, lbl in enumerate(basis_labels) if lbl.count("1") == n_electrons
    ]
    selected_probs = probs[selected_indices]
    selected_labels = [basis_labels[i] for i in selected_indices]

    if np.sum(selected_probs) > 0:
        selected_probs = selected_probs / np.sum(selected_probs)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.bar(range(len(selected_probs)), selected_probs, color="steelblue")
    ax.set_xticks(range(len(selected_labels)))
    ax.set_xticklabels(selected_labels, rotation=90, fontsize=8)
    ax.set_xlabel("Computational Basis State (bitstring)")
    ax.set_ylabel(r"Probability $|\langle i|\Psi\rangle|^2$")
    ax.set_title(
        f"{title}  (Sites={n_sites}, U={U}, t={t}, Ne={n_electrons})", fontsize=13
    )
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    print(f"Displayed {len(selected_labels)} states out of {len(probs)} total.")
