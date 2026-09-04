import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Drug-likeness filters and scoring for medchem package.
"""

from typing import Optional, Dict, Any
from rdkit import Chem
from rdkit.Chem import FilterCatalog
from medchem.descriptors import compute_all_descriptors

def lipinski_rule_of_5(smiles: str) -> Optional[Dict[str, Any]]:
    """Evaluate Lipinski Rule of 5."""
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return None
        
    violations = 0
    details = []
    
    mw = desc.get('MW', 0)
    logp = desc.get('LogP', 0)
    hbd = desc.get('HBD', 0)
    hba = desc.get('HBA', 0)
    
    if mw > 500:
        violations += 1
        details.append(f"MW ({mw:.2f}) > 500")
    if logp > 5:
        violations += 1
        details.append(f"LogP ({logp:.2f}) > 5")
    if hbd > 5:
        violations += 1
        details.append(f"HBD ({hbd}) > 5")
    if hba > 10:
        violations += 1
        details.append(f"HBA ({hba}) > 10")
        
    return {
        'MW': mw,
        'LogP': logp,
        'HBD': hbd,
        'HBA': hba,
        'pass': violations <= 1,
        'violations': violations,
        'details': details
    }

def veber_rules(smiles: str) -> Optional[Dict[str, Any]]:
    """Evaluate Veber rules."""
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return None
        
    violations = 0
    details = []
    
    rot_bonds = desc.get('RotatableBonds', 0)
    tpsa = desc.get('TPSA', 0)
    
    if rot_bonds > 10:
        violations += 1
        details.append(f"RotatableBonds ({rot_bonds}) > 10")
    if tpsa > 140:
        violations += 1
        details.append(f"TPSA ({tpsa:.2f}) > 140")
        
    return {
        'RotatableBonds': rot_bonds,
        'TPSA': tpsa,
        'pass': violations == 0,
        'violations': violations,
        'details': details
    }

def ghose_filter(smiles: str) -> Optional[Dict[str, Any]]:
    """Evaluate Ghose filter."""
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return None
        
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetNumAtoms() if mol else 0
    
    violations = 0
    details = []
    
    mw = desc.get('MW', 0)
    logp = desc.get('LogP', 0)
    mr = desc.get('MR', 0)
    
    if not (160 <= mw <= 480):
        violations += 1
        details.append(f"MW ({mw:.2f}) not in [160, 480]")
    if not (-0.4 <= logp <= 5.6):
        violations += 1
        details.append(f"LogP ({logp:.2f}) not in [-0.4, 5.6]")
    if not (40 <= mr <= 130):
        violations += 1
        details.append(f"MR ({mr:.2f}) not in [40, 130]")
    if not (20 <= atoms <= 70):
        violations += 1
        details.append(f"Total atoms ({atoms}) not in [20, 70]")
        
    return {
        'MW': mw,
        'LogP': logp,
        'MR': mr,
        'Atoms': atoms,
        'pass': violations == 0,
        'violations': violations,
        'details': details
    }

def muegge_filter(smiles: str) -> Optional[Dict[str, Any]]:
    """Evaluate Muegge filter."""
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return None
        
    violations = 0
    details = []
    
    mw = desc.get('MW', 0)
    logp = desc.get('LogP', 0)
    tpsa = desc.get('TPSA', 0)
    ring_count = desc.get('RingCount', 0)
    hba = desc.get('HBA', 0)
    hbd = desc.get('HBD', 0)
    rot_bonds = desc.get('RotatableBonds', 0)
    num_het = desc.get('NumHeteroatoms', 0)
    
    if not (200 <= mw <= 600): violations += 1; details.append(f"MW ({mw:.2f}) not in [200, 600]")
    if not (-2 <= logp <= 5): violations += 1; details.append(f"LogP ({logp:.2f}) not in [-2, 5]")
    if tpsa > 150: violations += 1; details.append(f"TPSA ({tpsa:.2f}) > 150")
    if ring_count > 7: violations += 1; details.append(f"RingCount ({ring_count}) > 7")
    if hba > 10: violations += 1; details.append(f"HBA ({hba}) > 10")
    if hbd > 5: violations += 1; details.append(f"HBD ({hbd}) > 5")
    if rot_bonds > 15: violations += 1; details.append(f"RotatableBonds ({rot_bonds}) > 15")
    if num_het < 1: violations += 1; details.append(f"NumHeteroatoms ({num_het}) < 1")
    
    return {
        'pass': violations == 0,
        'violations': violations,
        'details': details
    }

def egan_filter(smiles: str) -> Optional[Dict[str, Any]]:
    """Evaluate Egan egg (absorption) filter."""
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return None
        
    violations = 0
    details = []
    
    logp = desc.get('LogP', 0)
    tpsa = desc.get('TPSA', 0)
    
    if logp > 5.88:
        violations += 1
        details.append(f"LogP ({logp:.2f}) > 5.88")
    if tpsa > 131.6:
        violations += 1
        details.append(f"TPSA ({tpsa:.2f}) > 131.6")
        
    return {
        'LogP': logp,
        'TPSA': tpsa,
        'pass': violations == 0,
        'violations': violations,
        'details': details
    }

def _run_filter_catalog(smiles: str, catalogs) -> Optional[Dict[str, Any]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    params = FilterCatalog.FilterCatalogParams()
    for cat in catalogs:
        params.AddCatalog(cat)
        
    catalog = FilterCatalog.FilterCatalog(params)
    
    alerts = []
    if catalog.HasMatch(mol):
        matches = catalog.GetMatches(mol)
        for match in matches:
            alerts.append(match.GetDescription())
            
    return {
        'pass': len(alerts) == 0,
        'alerts': alerts,
        'num_alerts': len(alerts)
    }

def pains_filter(smiles: str) -> Optional[Dict[str, Any]]:
    """Evaluate PAINS filter."""
    catalogs = [
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B,
        FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C
    ]
    return _run_filter_catalog(smiles, catalogs)

def brenk_filter(smiles: str) -> Optional[Dict[str, Any]]:
    """Evaluate BRENK filter."""
    catalogs = [FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK]
    return _run_filter_catalog(smiles, catalogs)

def full_druglikeness_profile(smiles: str) -> Optional[Dict[str, Any]]:
    """Run all drug-likeness filters and return a combined report."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
        
    lipinski = lipinski_rule_of_5(smiles)
    veber = veber_rules(smiles)
    ghose = ghose_filter(smiles)
    muegge = muegge_filter(smiles)
    egan = egan_filter(smiles)
    pains = pains_filter(smiles)
    brenk = brenk_filter(smiles)
    
    overall_pass = True
    for f in [lipinski, veber, ghose, muegge, egan, pains, brenk]:
        if f is not None and not f.get('pass', False):
            overall_pass = False
            
    return {
        'lipinski': lipinski,
        'veber': veber,
        'ghose': ghose,
        'muegge': muegge,
        'egan': egan,
        'pains': pains,
        'brenk': brenk,
        'overall_pass': overall_pass
    }
