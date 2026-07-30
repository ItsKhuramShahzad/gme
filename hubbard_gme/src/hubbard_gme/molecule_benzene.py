import numpy as np;
from openfermion.ops import FermionOperator
from openfermion.linalg import get_number_preserving_sparse_operator

# Step 0
electrons= {'0':0 , "a": 1, 'b': 1, "2": 2}
coeff=0
occ=[]
ele_list=[]
seen = set()
data =[]
def readfile(file_path):
    x=0.00
    l=0
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            cols= line.split()
            if len(cols)<3:
                continue
            data.append((float(cols[1]), cols[2]))
            coeff= cols[1]
            occ=cols[2]
            
            seen.add(occ)
            # electron count:
            n_e= sum(electrons[ch] for ch in cols[2])
            ele_list.append(n_e)
            
            # check Sz
            if cols[2].count('a') != cols[2].count('b'):
                print("col mismatch", cols[2])

            l= len(cols[2])
            if len(cols)>=2:
                x+=  float(cols[1]) * float(cols[1])
            # print(line)
        
    print("total sum of coffeicent:", x)
    print("length:", l)
    return x, data


# Step 1: 
def occ_to_orbitals(occ):
    orbitals= set()
    
    for k, ch in enumerate(occ):
        up = 2*k
        down= 2*k+1
        
        if ch =='a':
            orbitals.add(up)
        elif ch=='b':
            orbitals.add(down)
        elif ch=='2':
            orbitals.add(up)
            orbitals.add(down)
    # print(orbitals)
    return orbitals
    
orbitals=[]     


# Step 2  
def step_2_3():
    # 2a
    n0= get_number_preserving_sparse_operator(
        FermionOperator(((0,1), (0,0))), 12, 6, spin_preserving=True)
        
    print("n0 shape", n0.shape)
    d0= n0.diagonal()
    
    print("d0", d0)
    print(int(d0.sum()))
    
    #2b
    diags= []
    for p in range (12):
        n_p = get_number_preserving_sparse_operator(
            FermionOperator(((p,1), (p,0))), 12, 6, spin_preserving=True
                   
        )
        diags.append(n_p.diagonal())
    
    #2c
    config_to_index={}
    for i in range(400):
        occupied = frozenset(p for p in range (12) if round (diags[p][i])==1)
        config_to_index[occupied]= i
    
    print(config_to_index)
    #2d
 
    print("Map size:", len(config_to_index))                                   # expect 400
    print("all 6e:", all(sum(round(diags[p][i]) for p in range(12)) == 6
                        for i in range(400)))                              # expect True

    print("bijection:", all(frozenset(occ_to_orbitals(o)) in config_to_index
                            for o in seen))                            # expect True

    # Step 3:
    psi= np.zeros(400)
    
    for coeff, occ in data:
        key = frozenset(occ_to_orbitals(occ))
        slot = config_to_index[key]
        psi[slot]= float(coeff)
        
    psi=psi/np.linalg.norm(psi)  # make <psi|psi> excatly 1
    
    print("norm", np.vdot(psi,psi).real)
    # from openfermion.hamiltonians import s_squared_operator
    from openfermion.hamiltonians import s_squared_operator
    S2= get_number_preserving_sparse_operator(
        s_squared_operator(6), 12, 6, spin_preserving=True
    )
    print(psi)  
    print("S^2", np.vdot(psi, S2@psi).real) # expect zero
    return psi


def make_spatial_1rdm(psi):
    """Build the 6x6 spin-summed 1-RDM of the CI state in the localized basis."""
    n_spin_orbitals = 12
    n_electrons = 6
    n_spatial_orbitals = 6
    rdm1 = np.zeros(
        (n_spatial_orbitals, n_spatial_orbitals), dtype=complex
    )

    # PySCF density-matrix convention: rdm1[p, q] = <a_q^dagger a_p>.
    for p in range(n_spatial_orbitals):
        for q in range(n_spatial_orbitals):
            for spin in (0, 1):
                p_spin = 2 * p + spin
                q_spin = 2 * q + spin
                operator = FermionOperator(
                    ((q_spin, 1), (p_spin, 0))
                )
                sparse_operator = get_number_preserving_sparse_operator(
                    operator,
                    num_qubits=n_spin_orbitals,
                    num_electrons=n_electrons,
                    spin_preserving=True,
                )
                rdm1[p, q] += np.vdot(
                    psi, sparse_operator @ psi
                )

    rdm1 = 0.5 * (rdm1 + rdm1.conj().T)
    print("Spatial 1-RDM trace:", np.trace(rdm1).real)
    return np.real_if_close(rdm1)


def write_localized_mo_cubes(mol, localized_mos, output_dir="localized_mo_cubes"):
    """Write one cube per existing localized MO; no localization is performed."""
    import os
    from pyscf.tools import cubegen

    localized_mos = np.asarray(localized_mos)
    expected_shape = (mol.nao_nr(), 6)
    if localized_mos.shape != expected_shape:
        raise ValueError(
            f"localized_mos must have shape {expected_shape}, "
            f"got {localized_mos.shape}"
        )

    os.makedirs(output_dir, exist_ok=True)
    for orbital in range(6):
        cube_path = os.path.join(
            output_dir, f"benzene_localized_pz_{orbital + 1}.cube"
        )
        cubegen.orbital(
            mol,
            cube_path,
            np.real_if_close(localized_mos[:, orbital]),
        )
        print("Wrote", cube_path)


def write_ci_density_cube(
    psi,
    mol,
    localized_mos,
    cube_path="benzene_ci_density.cube",
):
    """Write the 3D correlated density represented by the many-electron CI psi."""
    from pyscf.tools import cubegen

    localized_mos = np.asarray(localized_mos)
    expected_shape = (mol.nao_nr(), 6)
    if localized_mos.shape != expected_shape:
        raise ValueError(
            f"localized_mos must have shape {expected_shape}, "
            f"got {localized_mos.shape}"
        )

    rdm1_localized = make_spatial_1rdm(psi)
    density_matrix = (
        localized_mos
        @ rdm1_localized
        @ localized_mos.conj().T
    )
    density_matrix = np.real_if_close(density_matrix)
    cubegen.density(mol, cube_path, density_matrix)
    print("Wrote", cube_path)

def main():
    print("Benzene Coefficient file")
    readfile("benzene_full_evec.dat")
    print("Electron count in each coefficient (==6)", all (e==6 for e in ele_list))
    print("Rows:", len(ele_list))
    print("Distinct:", len(seen))
    # occ_to_orbitals(seen)
    print(occ_to_orbitals("222000") )
    print(occ_to_orbitals("ababab") )
    print(occ_to_orbitals("000222") )
    print("all len==6:", all(len(occ_to_orbitals(o)) == 6 for o in seen))
# expect: True
    for ooc_string in seen:
        orbitals.append(occ_to_orbitals(ooc_string))
    # print(orbitals)
    # print(len(orbitals))
    psi= step_2_3()

    import sys, os
    _SRC= os.path.dirname(os.path.dirname(os.path.abspath(__file__ )))
    if _SRC not in sys.path:
        sys.path.insert(0,_SRC)

    from hubbard_gme.molecular_pipeline import _precompute_mol_ops, _mol_rdm_from_ops
    sites =(0,1,2)
    ops= _precompute_mol_ops(sites,12, 6)
    rho= _mol_rdm_from_ops(ops, psi, len(sites))
    rho = rho/np.trace(rho).real
    print("Shape: ", rho.shape)
    print("Trace: ", np.trace(rho))
    print("Hermition: ", np.allclose(rho, rho.conj().T))
    evals= np.linalg.eigvalsh(rho)
    print("Eigen vals0: ", np.round(evals, 6))
    print("Sum of Eigenvals: ", evals.sum())
    print("Purity: ", np.trace(rho@rho).real)
    
    from hubbard_gme.molecular_pipeline import run_convex_roof_on_rdm
    result = run_convex_roof_on_rdm(rho, sites, method="gcg", n_restarts=8, 
                                    max_iter=200, k_max_extra=2, verbose=True)
    
    print("C_gcg (Benzene: 2 site GME)", result["C_gcg"])
    print("status", result["status"])
    
if __name__ == "__main__":
    main()
