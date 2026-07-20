import numpy as np
import cudaq
import openfermion
import openfermionpyscf
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.ops import QubitOperator
from pyscf import mcscf
from molecules_data import molecules

# Setup: NH3+ at smallest active space
spec = molecules["NH3+"]
space = spec["valid_active_spaces"][0]  # (ncore=2, nele_cas=5, norb_cas=4)
ncore, nele_cas, norb_cas = space["ncore"], space["nele_cas"], space["norb_cas"]
qubit_count = 2 * norb_cas
print(f"NH3+ at ({nele_cas}e, {norb_cas}o): {qubit_count} qubits")

moldata = openfermion.MolecularData(spec["geometry"], "cc-pVDZ", 2, 1)
molecule = openfermionpyscf.run_pyscf(moldata, run_scf=True, run_fci=False)
mf = molecule._pyscf_data["scf"]
print(f"E_HF = {molecule.hf_energy:.8f}")

# CASCI reference
casci = mcscf.CASCI(mf, norb_cas, nele_cas)
casci.ncore = ncore
E_CASCI = float(casci.kernel()[0])
print(f"E_CASCI = {E_CASCI:.8f}")

# Build Hamiltonian
occ = list(range(ncore))
act = list(range(ncore, ncore + norb_cas))
mh = molecule.get_molecular_hamiltonian(occupied_indices=occ, active_indices=act)
qop = jordan_wigner(get_fermion_operator(mh))

# Strip imag + constant
qreal = QubitOperator()
for term, coeff in qop.terms.items():
    qreal += QubitOperator(term, float(np.real(coeff)))
c0 = float(np.real(qreal.terms.get((), 0.0)))
qnc = QubitOperator()
for term, coeff in qreal.terms.items():
    if term:
        qnc += QubitOperator(term, float(np.real(coeff)))
spin_nc = cudaq.SpinOperator(qnc)
print(f"c0 = {c0:.8f}")

# Try the built-in kernel
@cudaq.kernel
def kernel_nh3(thetas: list[float]):
    q = cudaq.qvector(qubit_count)
    # Fill 5 lowest qubits (HF reference, 3 alpha + 2 beta in interleaved layout)
    x(q[0]); x(q[1]); x(q[2]); x(q[3]); x(q[4])
    cudaq.kernels.uccsd(q, thetas, nele_cas, qubit_count)

n_params = cudaq.kernels.uccsd_num_parameters(nele_cas, qubit_count)
print(f"n_params = {n_params}")

# Energy at theta=0 (HF)
thetas = np.zeros(n_params).tolist()
result = cudaq.observe(kernel_nh3, spin_nc, thetas)
E_HF_circuit = c0 + float(result.expectation())
print(f"E at theta=0 (should be ≈ E_HF active): {E_HF_circuit:.8f}")

# Energy at small random theta (just to see if it runs and produces sensible numbers)
np.random.seed(0)
thetas_rand = (0.01 * np.random.randn(n_params)).tolist()
result = cudaq.observe(kernel_nh3, spin_nc, thetas_rand)
E_random = c0 + float(result.expectation())
print(f"E at small random theta: {E_random:.8f}")

print()
print(f"Summary:")
print(f"  E_CASCI         = {E_CASCI:.8f}   ← target")
print(f"  E_VQE at HF     = {E_HF_circuit:.8f}")
print(f"  E_VQE at random = {E_random:.8f}")
