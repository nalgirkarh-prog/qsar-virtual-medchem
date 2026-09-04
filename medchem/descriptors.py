"""
Molecular descriptor calculator for medchem package.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, GraphDescriptors, MolSurf, QED, AllChem, MACCSkeys, rdMolDescriptors
from rdkit import DataStructs

def compute_all_descriptors(smiles: str) -> Optional[Dict[str, Any]]:
    """Compute a comprehensive set of molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Calculate basic descriptors
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    mr = Crippen.MolMR(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)
    ring_count = Lipinski.RingCount(mol)
    aromatic_rings = Lipinski.NumAromaticRings(mol)
    num_heteroatoms = Lipinski.NumHeteroatoms(mol)
    fraction_csp3 = Lipinski.FractionCSP3(mol)
    heavy_atom_count = Lipinski.HeavyAtomCount(mol)
    num_stereocenters = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    num_amide_bonds = rdMolDescriptors.CalcNumAmideBonds(mol)
    
    # Topological / Graph descriptors
    balaban_j = GraphDescriptors.BalabanJ(mol)
    bertz_ct = GraphDescriptors.BertzCT(mol)
    chi0 = GraphDescriptors.Chi0(mol)
    chi1 = GraphDescriptors.Chi1(mol)
    kappa1 = GraphDescriptors.Kappa1(mol)
    kappa2 = GraphDescriptors.Kappa2(mol)
    kappa3 = GraphDescriptors.Kappa3(mol)
    hall_kier_alpha = GraphDescriptors.HallKierAlpha(mol)
    
    # Other specific descriptor classes
    qed_val = QED.qed(mol)
    num_aliphatic_rings = Lipinski.NumAliphaticRings(mol)
    num_aromatic_heterocycles = Lipinski.NumAromaticHeterocycles(mol)
    num_saturated_rings = Lipinski.NumSaturatedRings(mol)
    labute_asa = MolSurf.LabuteASA(mol)
    
    # PEOE_VSA
    peoe_vsa = MolSurf.PEOE_VSA_(mol)
    peoe_dict = {f"PEOE_VSA{i+1}": val for i, val in enumerate(peoe_vsa[:6])}
    
    # Lipinski Violations
    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1
    
    # ESOL LogS
    # LogS = 0.16 - 0.63*LogP - 0.0062*MW + 0.066*RotBonds - 0.74*AromaticProportion
    aromatic_proportion = 0
    if heavy_atom_count > 0:
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        aromatic_proportion = aromatic_atoms / heavy_atom_count
    logs_esol = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rot_bonds - 0.74 * aromatic_proportion

    desc = {
        'MW': mw,
        'LogP': logp,
        'MR': mr,
        'TPSA': tpsa,
        'HBD': hbd,
        'HBA': hba,
        'RotatableBonds': rot_bonds,
        'RingCount': ring_count,
        'AromaticRings': aromatic_rings,
        'NumHeteroatoms': num_heteroatoms,
        'FractionCSP3': fraction_csp3,
        'HeavyAtomCount': heavy_atom_count,
        'NumStereocenters': num_stereocenters,
        'NumAmideBonds': num_amide_bonds,
        'BalabanJ': balaban_j,
        'BertzCT': bertz_ct,
        'Chi0': chi0,
        'Chi1': chi1,
        'Kappa1': kappa1,
        'Kappa2': kappa2,
        'Kappa3': kappa3,
        'HallKierAlpha': hall_kier_alpha,
        'LogS_ESOL': logs_esol,
        'QED': qed_val,
        'NumAliphaticRings': num_aliphatic_rings,
        'NumAromaticHeterocycles': num_aromatic_heterocycles,
        'NumSaturatedRings': num_saturated_rings,
        'LabuteASA': labute_asa,
        'Lipinski_Violations': lipinski_violations
    }
    desc.update(peoe_dict)
    
    return desc

def compute_descriptors_batch(smiles_list: List[str]) -> pd.DataFrame:
    """Compute descriptors for a list of SMILES strings and return a DataFrame."""
    records = []
    for sm in smiles_list:
        desc = compute_all_descriptors(sm)
        if desc is not None:
            desc['SMILES'] = sm
            records.append(desc)
    
    if records:
        df = pd.DataFrame(records)
        cols = ['SMILES'] + [c for c in df.columns if c != 'SMILES']
        return df[cols]
    return pd.DataFrame()

def compute_fingerprint(smiles: str, fp_type: str = 'morgan', radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Compute molecular fingerprint as a numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    fp_type = fp_type.lower()
    
    if fp_type == 'morgan':
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = gen.GetFingerprint(mol)
        except Exception:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    elif fp_type == 'maccs':
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == 'rdkit':
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
    else:
        print(f"Error: Unknown fingerprint type '{fp_type}'")
        return None
        
    arr = np.zeros(fp.GetNumBits(), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def tanimoto_similarity(smiles1: str, smiles2: str, fp_type: str = 'morgan') -> Optional[float]:
    """Compute Tanimoto similarity between two SMILES strings."""
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return None
        
    fp_type = fp_type.lower()
    
    if fp_type == 'morgan':
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = gen.GetFingerprint(mol1)
            fp2 = gen.GetFingerprint(mol2)
        except Exception:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2)
    elif fp_type == 'maccs':
        fp1 = MACCSkeys.GenMACCSKeys(mol1)
        fp2 = MACCSkeys.GenMACCSKeys(mol2)
    elif fp_type == 'rdkit':
        fp1 = Chem.RDKFingerprint(mol1)
        fp2 = Chem.RDKFingerprint(mol2)
    else:
        print(f"Error: Unknown fingerprint type '{fp_type}'")
        return None
        
    return DataStructs.TanimotoSimilarity(fp1, fp2)
