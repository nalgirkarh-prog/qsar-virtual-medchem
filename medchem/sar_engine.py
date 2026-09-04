import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Optional, List, Dict, Tuple
from medchem.descriptors import compute_all_descriptors, tanimoto_similarity
from medchem.substituents import SUBSTITUENT_LIBRARY, get_all_substituents

from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFMCS


def _attach_substituent(core_mol: Chem.Mol, sub_data: dict) -> Optional[str]:
    """
    Connects a substituent fragment to core_mol at the dummy atom position.
    Returns canonical SMILES of product or None if failed.
    """
    sub_smi = sub_data.get("smiles", "")
    sub_name = sub_data.get("name", "")

    # Dummy atom in core
    dummy_indices = [a.GetIdx() for a in core_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_indices:
        return None
    d_idx = dummy_indices[0]

    # Special case: -H replacement
    if sub_smi in ("[H]", "H") or sub_name in ("-H", "H"):
        rw = Chem.RWMol(core_mol)
        rw.RemoveAtom(d_idx)
        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)

    sub_mol = Chem.MolFromSmiles(sub_smi)
    if not sub_mol:
        return None

    core_dummy = core_mol.GetAtomWithIdx(d_idx)
    neighbors = core_dummy.GetNeighbors()
    if not neighbors:
        return None
    attach_core_idx = neighbors[0].GetIdx()
    bond_type = core_mol.GetBondBetweenAtoms(d_idx, attach_core_idx).GetBondType()

    combo = Chem.RWMol(Chem.CombineMols(core_mol, sub_mol))
    sub_offset = core_mol.GetNumAtoms()

    # Find dummy in sub_mol if present
    sub_dummies = [a.GetIdx() for a in sub_mol.GetAtoms() if a.GetAtomicNum() == 0]
    if sub_dummies:
        sub_d = sub_mol.GetAtomWithIdx(sub_dummies[0])
        sub_neighbors = sub_d.GetNeighbors()
        if not sub_neighbors:
            return None
        sub_attach_idx = sub_offset + sub_neighbors[0].GetIdx()
        combo.AddBond(attach_core_idx, sub_attach_idx, bond_type)
        for idx in sorted([d_idx, sub_offset + sub_dummies[0]], reverse=True):
            combo.RemoveAtom(idx)
    else:
        # Default: attach at atom 0 of sub
        sub_attach_idx = sub_offset
        combo.AddBond(attach_core_idx, sub_attach_idx, bond_type)
        combo.RemoveAtom(d_idx)

    mol = combo.GetMol()
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def enumerate_rgroup_substitutions(
    core_smiles: str,
    position_smarts: str = "[#0]",
    substituent_names: Optional[List[str]] = None
) -> List[Dict]:
    """
    Given a core scaffold SMILES with a dummy atom [*] marking the substitution point,
    generates all substituted molecules using the substituent library.
    """
    # Standardize dummy marker in SMILES if user used other markers
    norm_smiles = core_smiles.replace("[R]", "[*]")
    core_mol = Chem.MolFromSmiles(norm_smiles)
    if core_mol is None:
        return []

    if substituent_names is None:
        substituent_names = list(SUBSTITUENT_LIBRARY.keys())

    results = []
    for name in substituent_names:
        cleaned_name = name if name.startswith("-") else "-" + name
        sub_data = SUBSTITUENT_LIBRARY.get(cleaned_name)
        if not sub_data:
            continue

        try:
            prod_smiles = _attach_substituent(core_mol, sub_data)
            if prod_smiles:
                descriptors = compute_all_descriptors(prod_smiles)
                if descriptors:
                    results.append({
                        "substituent_name": sub_data["name"],
                        "product_smiles": prod_smiles,
                        "descriptors": descriptors
                    })
        except Exception:
            continue

    return results


def extract_scaffold(smiles: str) -> Dict:
    """
    Extracts the Murcko scaffold and generic scaffold for a molecule.
    Returns: {"murcko_scaffold": SMILES, "generic_scaffold": SMILES, "side_chains": list[str]}
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return {}

    try:
        murcko = MurckoScaffold.GetScaffoldForMol(mol)
        murcko_smiles = Chem.MolToSmiles(murcko)

        fw = MurckoScaffold.MakeScaffoldGeneric(murcko)
        generic_smiles = Chem.MolToSmiles(fw)

        side_chains_mol = Chem.ReplaceCore(mol, murcko)
        side_chains = []
        if side_chains_mol:
            raw_chains = Chem.MolToSmiles(side_chains_mol).split(".")
            side_chains = [c for c in raw_chains if c and c != "*"]

        return {
            "murcko_scaffold": murcko_smiles,
            "generic_scaffold": generic_smiles,
            "side_chains": side_chains
        }
    except Exception:
        return {}


def find_mcs(smiles_list: List[str]) -> Dict:
    """
    Find Maximum Common Substructure among a list of molecules.
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if s]
    mols = [m for m in mols if m is not None]

    if len(mols) < 2:
        return {}

    res = rdFMCS.FindMCS(mols)
    return {
        "mcs_smarts": res.smartsString,
        "num_atoms": res.numAtoms,
        "num_bonds": res.numBonds
    }


def matched_molecular_pairs(smiles_activity_pairs: List[Tuple[str, float]]) -> List[Dict]:
    """
    Given list of (SMILES, activity) tuples, identify structural transformations and activity changes.
    """
    results = []
    n = len(smiles_activity_pairs)
    for i in range(n):
        for j in range(i + 1, n):
            smi1, act1 = smiles_activity_pairs[i]
            smi2, act2 = smiles_activity_pairs[j]
            mol1 = Chem.MolFromSmiles(smi1)
            mol2 = Chem.MolFromSmiles(smi2)

            if not mol1 or not mol2:
                continue

            mcs = rdFMCS.FindMCS([mol1, mol2])
            if mcs.numAtoms > 0:
                core = Chem.MolFromSmarts(mcs.smartsString)
                if core:
                    frag1 = Chem.ReplaceCore(mol1, core)
                    frag2 = Chem.ReplaceCore(mol2, core)
                    f1_smi = Chem.MolToSmiles(frag1) if frag1 else "[H]"
                    f2_smi = Chem.MolToSmiles(frag2) if frag2 else "[H]"
                    transformation = f"{f1_smi} >> {f2_smi}"

                    results.append({
                        "mol1_smiles": smi1,
                        "mol2_smiles": smi2,
                        "transformation": transformation,
                        "activity_change": round(act2 - act1, 4),
                        "mol1_activity": act1,
                        "mol2_activity": act2
                    })
    return results


def sar_summary(compounds: List[Dict]) -> Dict:
    """
    Takes list of dicts with "smiles", "activity", and optionally "name" keys.
    Returns summary: activity range, most/least active, scaffold analysis.
    """
    if not compounds:
        return {}

    activities = [c.get("activity", 0.0) for c in compounds]
    smiles_list = [c.get("smiles", "") for c in compounds]

    max_act = max(activities) if activities else 0.0
    min_act = min(activities) if activities else 0.0

    most_active = compounds[activities.index(max_act)] if activities else {}
    least_active = compounds[activities.index(min_act)] if activities else {}

    scaffolds = {}
    for smi in smiles_list:
        if not smi:
            continue
        scaf_dict = extract_scaffold(smi)
        scaf = scaf_dict.get("murcko_scaffold")
        if scaf:
            scaffolds[scaf] = scaffolds.get(scaf, 0) + 1

    return {
        "n_compounds": len(compounds),
        "activity_min": min_act,
        "activity_max": max_act,
        "activity_range": (min_act, max_act),
        "most_active": most_active,
        "least_active": least_active,
        "dominant_scaffolds": scaffolds
    }
