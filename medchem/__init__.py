"""
MedChem — Virtual Medicinal Chemistry Tool
==========================================

A comprehensive QSAR and SAR platform for pharmaceutical predictions.

Modules:
    descriptors     — Full physicochemical descriptor calculator
    druglikeness    — Lipinski, Veber, QED, PAINS, and other drug-likeness filters
    substituents    — Hammett σ, Taft Es, Hansch π constants for common substituents
    sar_engine      — R-group enumeration, MMP analysis, scaffold decomposition
    bioisosteres    — Bioisosteric replacement engine
    qsar_model      — ML model training pipeline (RF, Linear, Ridge)
    predict         — Prediction + full medchem reporting
    virtual_screen  — Virtual substitution and screening workflow
    medchem_cli     — Unified CLI entry point
"""

__version__ = "1.0.0"
__author__ = "MedChem Virtual Tool"

# Lazy imports — only load what's needed
def profile(smiles: str) -> dict:
    """Quick access: compute full medchem profile for a SMILES string."""
    from medchem.descriptors import compute_all_descriptors
    from medchem.druglikeness import full_druglikeness_profile
    
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return None
    drug = full_druglikeness_profile(smiles)
    return {**desc, 'druglikeness': drug}
