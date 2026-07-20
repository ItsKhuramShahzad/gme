import numpy as np


# ============================================================
# Basic spin-1/2 matrices
# ============================================================
# We use the convention:
#
# |up>   = |0> = [1, 0]^T
# |down> = |1> = [0, 1]^T
#
# Pauli matrices:
# sigma_x, sigma_y, sigma_z
#
# Spin operators:
# Sx = sigma_x / 2
# Sy = sigma_y / 2
# Sz = sigma_z / 2
# ============================================================

sigma_x = np.array([[0, 1],
                    [1, 0]], dtype=complex)

sigma_y = np.array([[0, -1j],
                    [1j, 0]], dtype=complex)

sigma_z = np.array([[1, 0],
                    [0, -1]], dtype=complex)

I2 = np.eye(2, dtype=complex)

Sx_single = sigma_x / 2
Sy_single = sigma_y / 2
Sz_single = sigma_z / 2


# ============================================================
# Put a single-spin operator on a selected site
# ============================================================
# Example:
# For N = 3 spins,
#
# S_0^x = Sx ⊗ I  ⊗ I
# S_1^x = I  ⊗ Sx ⊗ I
# S_2^x = I  ⊗ I  ⊗ Sx
#
# Here Python indexing is used:
# site = 0, 1, ..., N-1
# ============================================================

def operator_on_site(single_site_operator, site, N):
    """
    Construct the full many-spin operator acting on one site.

    Parameters
    ----------
    single_site_operator : np.ndarray
        2 x 2 operator, for example Sx, Sy, or Sz.

    site : int
        Site where the operator acts.
        Python indexing: site = 0, 1, ..., N-1.

    N : int
        Number of spin-1/2 particles.

    Returns
    -------
    full_operator : np.ndarray
        2^N x 2^N matrix.
    """

    operators = []

    for i in range(N):
        if i == site:
            operators.append(single_site_operator)
        else:
            operators.append(I2)

    full_operator = operators[0]

    for op in operators[1:]:
        full_operator = np.kron(full_operator, op)

    return full_operator


# ============================================================
# Build the Hamiltonian from Eq. (44)
# ============================================================
#
# Eq. (44):
#
# H = -J sum_i S_i . S_{i+1}
#     + b sum_i [ S_i^x cos(alpha_i) + S_i^y sin(alpha_i) ]
#
# where
#
# S_i . S_{i+1}
# =
# S_i^x S_{i+1}^x
# + S_i^y S_{i+1}^y
# + S_i^z S_{i+1}^z
#
# and
#
# alpha_i = 2 pi i / N
#
# Because Python indexing starts at i = 0, this corresponds to the
# paper's alpha_i = 2 pi (i-1) / N if the paper uses i = 1,...,N.
#
# Periodic boundary condition:
#
# site N-1 is connected back to site 0.
# ============================================================

def build_eq44_hamiltonian(N, J=1.0, b=0.1):
    """
    Build the spin-chain Hamiltonian from Eq. (44).

    Parameters
    ----------
    N : int
        Number of spins/qubits.

    J : float
        Exchange coupling.
        For the ferromagnetic case in the paper, use J > 0
        because the Hamiltonian contains -J.

    b : float
        Strength of the local magnetic field.

    Returns
    -------
    H : np.ndarray
        Hamiltonian matrix of size 2^N x 2^N.
    """

    dim = 2 ** N

    # Initialize Hamiltonian as a zero matrix
    H = np.zeros((dim, dim), dtype=complex)

    # Build spin operators on every site
    #
    # Sx_ops[i] means S_i^x
    # Sy_ops[i] means S_i^y
    # Sz_ops[i] means S_i^z

    Sx_ops = []
    Sy_ops = []
    Sz_ops = []

    for i in range(N):
        Sx_ops.append(operator_on_site(Sx_single, i, N))
        Sy_ops.append(operator_on_site(Sy_single, i, N))
        Sz_ops.append(operator_on_site(Sz_single, i, N))

    # --------------------------------------------------------
    # First term:
    #
    # -J sum_i S_i . S_{i+1}
    #
    # The closed chain is implemented using:
    #
    # j = (i + 1) % N
    #
    # For example, if N = 3:
    # i = 0 -> j = 1
    # i = 1 -> j = 2
    # i = 2 -> j = 0
    # --------------------------------------------------------

    for i in range(N):
        j = (i + 1) % N

        spin_dot = (
            Sx_ops[i] @ Sx_ops[j]
            + Sy_ops[i] @ Sy_ops[j]
            + Sz_ops[i] @ Sz_ops[j]
        )

        H += -J * spin_dot

    # --------------------------------------------------------
    # Second term:
    #
    # b sum_i [ S_i^x cos(alpha_i) + S_i^y sin(alpha_i) ]
    #
    # This is a local magnetic field in the xy-plane.
    # The field direction changes from site to site.
    # --------------------------------------------------------

    for i in range(N):
        alpha_i = 2 * np.pi * i / N

        local_field_term = (
            np.cos(alpha_i) * Sx_ops[i]
            + np.sin(alpha_i) * Sy_ops[i]
        )

        H += b * local_field_term

    return H


# ============================================================
# Solve the Hamiltonian
# ============================================================
#
# We diagonalize:
#
# H |psi_n> = E_n |psi_n>
#
# The ground state is the eigenvector with the smallest eigenvalue.
# ============================================================

def ground_state(H):
    """
    Diagonalize H and return the ground-state energy and vector.

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix.

    Returns
    -------
    E0 : float
        Ground-state energy.

    psi0 : np.ndarray
        Ground-state vector.
    """

    energies, eigenvectors = np.linalg.eigh(H)

    E0 = energies[0]
    psi0 = eigenvectors[:, 0]

    return E0, psi0


# ============================================================
# Convert a pure state into a density matrix
# ============================================================
#
# rho = |psi><psi|
# ============================================================

def density_matrix_from_state(psi):
    """
    Construct rho = |psi><psi|.

    Parameters
    ----------
    psi : np.ndarray
        State vector.

    Returns
    -------
    rho : np.ndarray
        Density matrix.
    """

    rho = np.outer(psi, psi.conj())

    return rho


# ============================================================
# Reduced density matrix of one qubit from a pure state
# ============================================================
#
# Eq. (45) requires rho_k, the reduced density matrix of qubit k:
#
# rho_k = Tr_{all qubits except k}(|psi><psi|)
#
# This function computes rho_k directly from the state vector.
# ============================================================

def one_qubit_reduced_density_matrix_from_state(psi, k, N):
    """
    Compute the one-qubit reduced density matrix rho_k
    from a pure N-qubit state.

    Parameters
    ----------
    psi : np.ndarray
        Pure state vector of length 2^N.

    k : int
        Qubit/spin index to keep.
        Python indexing: k = 0, 1, ..., N-1.

    N : int
        Number of qubits/spins.

    Returns
    -------
    rho_k : np.ndarray
        2 x 2 reduced density matrix of qubit k.
    """

    # Reshape the state vector into an N-index tensor.
    #
    # Example for N = 3:
    # psi[a,b,c]
    #
    # where each index is 0 or 1.
    psi_tensor = psi.reshape([2] * N)

    # Move qubit k to the first position.
    #
    # Example:
    # If k = 1 and N = 3,
    # original axes: [0, 1, 2]
    # new axes:      [1, 0, 2]
    axes = [k] + [i for i in range(N) if i != k]

    psi_tensor = np.transpose(psi_tensor, axes)

    # Now reshape as a matrix:
    #
    # rows:    state of qubit k, dimension 2
    # columns: states of all other qubits, dimension 2^(N-1)
    psi_matrix = psi_tensor.reshape(2, -1)

    # Reduced density matrix:
    #
    # rho_k = psi_matrix psi_matrix^\dagger
    rho_k = psi_matrix @ psi_matrix.conj().T

    return rho_k


# ============================================================
# Eq. (45): Meyer-Wallach global entanglement
# ============================================================
#
# gamma(|psi>) =
#
# 2 [ 1 - (1/N) sum_k Tr(rho_k^2) ]
#
# This formula is directly valid for pure states.
# ============================================================

def meyer_wallach_gamma_pure_state(psi, N):
    """
    Compute the Meyer-Wallach entanglement measure for a pure state.

    Parameters
    ----------
    psi : np.ndarray
        Pure state vector of length 2^N.

    N : int
        Number of qubits/spins.

    Returns
    -------
    gamma : float
        Meyer-Wallach global entanglement.
    """

    purity_sum = 0.0

    for k in range(N):
        rho_k = one_qubit_reduced_density_matrix_from_state(psi, k, N)

        purity_k = np.trace(rho_k @ rho_k).real

        purity_sum += purity_k

    gamma = 2.0 * (1.0 - purity_sum / N)

    return gamma


# ============================================================
# Thermal density matrix
# ============================================================
#
# For finite temperature:
#
# rho(T) = exp(-beta H) / Tr[exp(-beta H)]
#
# where beta = 1 / (k_B T).
#
# In numerical simulations we usually set k_B = 1, so:
#
# beta = 1 / T
#
# Important:
# Eq. (45) is directly for pure states.
# For rho(T), which is mixed, you should pass rho(T) to your
# convex-roof optimization code.
# ============================================================

def thermal_density_matrix(H, T):
    """
    Construct the thermal density matrix rho(T).

    Parameters
    ----------
    H : np.ndarray
        Hamiltonian matrix.

    T : float
        Temperature.
        We use k_B = 1, so beta = 1/T.

    Returns
    -------
    rho_T : np.ndarray
        Thermal density matrix.
    """

    if T <= 0:
        raise ValueError("For T = 0, use the ground-state density matrix instead.")

    beta = 1.0 / T

    energies, eigenvectors = np.linalg.eigh(H)

    # Shift energies by the ground-state energy for numerical stability.
    # This avoids overflow/underflow in exp(-beta E).
    E_min = energies[0]

    weights = np.exp(-beta * (energies - E_min))

    partition_function = np.sum(weights)

    probabilities = weights / partition_function

    # rho(T) = sum_n p_n |psi_n><psi_n|
    rho_T = eigenvectors @ np.diag(probabilities) @ eigenvectors.conj().T

    return rho_T


# ============================================================
# Utility checks
# ============================================================

def check_hermitian(H, tolerance=1e-10):
    """
    Check whether H is Hermitian: H = H^\dagger.
    """

    return np.allclose(H, H.conj().T, atol=tolerance)


def check_density_matrix(rho, tolerance=1e-10):
    """
    Basic checks for a density matrix.
    """

    trace_rho = np.trace(rho)
    hermitian = np.allclose(rho, rho.conj().T, atol=tolerance)
    trace_one = np.allclose(trace_rho, 1.0, atol=tolerance)
    purity = np.trace(rho @ rho).real

    return {
        "is_hermitian": hermitian,
        "trace": trace_rho,
        "trace_is_one": trace_one,
        "purity_Tr_rho_squared": purity,
    }


# ============================================================
# Example: reproduce zero-temperature result
# ============================================================

if __name__ == "__main__":

    # Number of spins in the paper's spin chain
    N = 3

    # Coupling constants
    J = 1.0
    b = 0.1

    # --------------------------------------------------------
    # Step 1: Build Eq. (44)
    # --------------------------------------------------------

    H = build_eq44_hamiltonian(N=N, J=J, b=b)

    print("Hamiltonian shape:", H.shape)
    print("Is H Hermitian?", check_hermitian(H))

    # --------------------------------------------------------
    # Step 2: Solve H |psi> = E |psi>
    # --------------------------------------------------------

    E0, psi0 = ground_state(H)

    print("Ground-state energy:", E0)

    # --------------------------------------------------------
    # Step 3: Build ground-state density matrix
    # --------------------------------------------------------

    rho0 = density_matrix_from_state(psi0)

    print("Ground-state density matrix checks:")
    print(check_density_matrix(rho0))

    # --------------------------------------------------------
    # Step 4: Apply Eq. (45) to the ground state
    # --------------------------------------------------------

    gamma0 = meyer_wallach_gamma_pure_state(psi0, N)

    print("Meyer-Wallach gamma at T = 0:", gamma0)

    # --------------------------------------------------------
    # Step 5: Optional finite-temperature density matrix
    # --------------------------------------------------------
    # This is not directly used in Eq. (45).
    # For finite temperature, pass rho_T to your convex-roof code.

    T = 0.1

    rho_T = thermal_density_matrix(H, T)

    print("Thermal density matrix checks:")
    print(check_density_matrix(rho_T))

    # Example:
    # gamma_T = your_convex_roof_code(rho_T)