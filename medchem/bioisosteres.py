import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List, Dict
from rdkit import Chem
from rdkit.Chem import rdChemReactions

BIOISOSTERE_LIBRARY: Dict[str, List[Dict]] = {
    "COOH": [
        {"name": "tetrazole", "reaction_smarts": "[*:1][CX3](=O)[OX2H1]>>[*:1]-c1nn[nH]n1", "type": "classical"},
        {"name": "acyl_sulfonamide", "reaction_smarts": "[*:1][CX3](=O)[OX2H1]>>[*:1]C(=O)NS(=O)(=O)C", "type": "non-classical"},
        {"name": "hydroxamic_acid", "reaction_smarts": "[*:1][CX3](=O)[OX2H1]>>[*:1]C(=O)NO", "type": "non-classical"},
        {"name": "phosphonic_acid", "reaction_smarts": "[*:1][CX3](=O)[OX2H1]>>[*:1]P(=O)(O)O", "type": "non-classical"},
    ],
    "amide": [
        {"name": "sulfonamide", "reaction_smarts": "[*:1][CX3](=O)[NH][#6:2]>>[*:1]S(=O)(=O)N[#6:2]", "type": "non-classical"},
        {"name": "urea", "reaction_smarts": "[*:1][CX3](=O)[NH][#6:2]>>[*:1]NC(=O)N[#6:2]", "type": "non-classical"},
        {"name": "reverse_amide", "reaction_smarts": "[#6:1][CX3](=O)[NH][#6:2]>>[#6:1]NC(=O)[#6:2]", "type": "classical"},
    ],
    "ester": [
        {"name": "amide", "reaction_smarts": "[*:1][CX3](=O)O[#6:2]>>[*:1]C(=O)N[#6:2]", "type": "classical"},
        {"name": "reverse_ester", "reaction_smarts": "[#6:1][CX3](=O)O[#6:2]>>[#6:1]OC(=O)[#6:2]", "type": "classical"},
    ],
    "phenyl": [
        {"name": "thiophene", "reaction_smarts": "[*:1]-c1ccccc1>>[*:1]-c1ccsc1", "type": "classical"},
        {"name": "pyridine_2yl", "reaction_smarts": "[*:1]-c1ccccc1>>[*:1]-c1ccccn1", "type": "classical"},
        {"name": "pyridine_3yl", "reaction_smarts": "[*:1]-c1ccccc1>>[*:1]-c1cccnc1", "type": "classical"},
        {"name": "pyridine_4yl", "reaction_smarts": "[*:1]-c1ccccc1>>[*:1]-c1ccncc1", "type": "classical"},
        {"name": "cyclopentyl", "reaction_smarts": "[*:1]-c1ccccc1>>[*:1]-C1CCCC1", "type": "non-classical"},
    ],
    "hydroxyl": [
        {"name": "phenol_to_amino", "reaction_smarts": "[c:1][OX2H]>>[c:1]N", "type": "classical"},
        {"name": "phenol_to_thiol", "reaction_smarts": "[c:1][OX2H]>>[c:1]S", "type": "classical"},
        {"name": "phenol_to_fluoro", "reaction_smarts": "[c:1][OX2H]>>[c:1]F", "type": "classical"},
        {"name": "alcohol_to_amino", "reaction_smarts": "[CX4:1][OX2H]>>[CX4:1]N", "type": "classical"},
        {"name": "alcohol_to_fluoro", "reaction_smarts": "[CX4:1][OX2H]>>[CX4:1]F", "type": "classical"},
    ],
    "halogen": [
        {"name": "Cl_to_F", "reaction_smarts": "[#6:1]Cl>>[#6:1]F", "type": "classical"},
        {"name": "Cl_to_Br", "reaction_smarts": "[#6:1]Cl>>[#6:1]Br", "type": "classical"},
        {"name": "Cl_to_CF3", "reaction_smarts": "[#6:1]Cl>>[#6:1]C(F)(F)F", "type": "non-classical"},
        {"name": "Br_to_Cl", "reaction_smarts": "[#6:1]Br>>[#6:1]Cl", "type": "classical"},
        {"name": "F_to_H", "reaction_smarts": "[#6:1]F>>[#6:1][H]", "type": "classical"},
    ],
    "ether_amine": [
        {"name": "O_to_NH", "reaction_smarts": "[#6:1][OX2][#6:2]>>[#6:1][NH][#6:2]", "type": "classical"},
        {"name": "O_to_S", "reaction_smarts": "[#6:1][OX2][#6:2]>>[#6:1][SX2][#6:2]", "type": "classical"},
        {"name": "NH_to_O", "reaction_smarts": "[#6:1][NH][#6:2]>>[#6:1][OX2][#6:2]", "type": "classical"},
        {"name": "NH_to_CH2", "reaction_smarts": "[#6:1][NH][#6:2]>>[#6:1][CH2][#6:2]", "type": "classical"},
    ]
}

def get_bioisosteres(group_name: str) -> List[Dict]:
    """Returns the list of bioisosteric replacements for a given functional group name."""
    return BIOISOSTERE_LIBRARY.get(group_name, [])

def list_replaceable_groups(smiles: str) -> List[str]:
    """Identifies which bioisosteric-replaceable groups are present in the molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []
    
    found = []
    for grp_name, replacements in BIOISOSTERE_LIBRARY.items():
        for repl in replacements:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(repl["reaction_smarts"])
                prods = rxn.RunReactants((mol,))
                if prods:
                    found.append(grp_name)
                    break
            except Exception:
                continue
    return found

def generate_bioisosteric_analogs(smiles: str) -> List[Dict]:
    """
    Scan the molecule for functional groups with known bioisosteric replacements
    and generate all valid substituted analogs.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return []

    results = []
    seen_smiles = {Chem.MolToSmiles(mol)}

    for grp_name, replacements in BIOISOSTERE_LIBRARY.items():
        for repl in replacements:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(repl["reaction_smarts"])
                prods = rxn.RunReactants((mol,))
                for prod_tuple in prods:
                    prod_mol = prod_tuple[0]
                    try:
                        Chem.SanitizeMol(prod_mol)
                        prod_smi = Chem.MolToSmiles(prod_mol)
                        if prod_smi and prod_smi not in seen_smiles and "." not in prod_smi:
                            seen_smiles.add(prod_smi)
                            results.append({
                                "original_group": grp_name,
                                "replacement_name": repl["name"],
                                "product_smiles": prod_smi,
                                "replacement_type": repl["type"]
                            })
                    except Exception:
                        continue
            except Exception:
                continue

    return results
