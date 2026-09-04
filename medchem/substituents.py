import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from typing import Optional, List, Dict, Any

# Substituent library containing common pharma substituents and their constants.
# References: Hansch, Leo, and Hoekman (1995)
# Fragment SMILES include "*" marking the exact point of attachment.
SUBSTITUENT_LIBRARY: Dict[str, Dict[str, Any]] = {
    "-H": {"name": "-H", "smiles": "[H]", "sigma_para": 0.00, "sigma_meta": 0.00, "pi": 0.00, "es": 0.00, "mr": 1.03, "type": "neutral", "category": "other"},
    "-CH3": {"name": "-CH3", "smiles": "*C", "sigma_para": -0.17, "sigma_meta": -0.07, "pi": 0.56, "es": -1.24, "mr": 5.65, "type": "EDG", "category": "alkyl"},
    "-C2H5": {"name": "-C2H5", "smiles": "*CC", "sigma_para": -0.15, "sigma_meta": -0.07, "pi": 1.02, "es": -1.31, "mr": 10.30, "type": "EDG", "category": "alkyl"},
    "-iPr": {"name": "-iPr", "smiles": "*C(C)C", "sigma_para": -0.15, "sigma_meta": -0.07, "pi": 1.53, "es": -1.71, "mr": 14.96, "type": "EDG", "category": "alkyl"},
    "-tBu": {"name": "-tBu", "smiles": "*C(C)(C)C", "sigma_para": -0.20, "sigma_meta": -0.10, "pi": 1.98, "es": -2.78, "mr": 19.62, "type": "EDG", "category": "alkyl"},
    "-F": {"name": "-F", "smiles": "*F", "sigma_para": 0.06, "sigma_meta": 0.34, "pi": 0.14, "es": -0.46, "mr": 0.92, "type": "EWG", "category": "halogen"},
    "-Cl": {"name": "-Cl", "smiles": "*Cl", "sigma_para": 0.23, "sigma_meta": 0.37, "pi": 0.71, "es": -0.97, "mr": 6.03, "type": "EWG", "category": "halogen"},
    "-Br": {"name": "-Br", "smiles": "*Br", "sigma_para": 0.23, "sigma_meta": 0.39, "pi": 0.86, "es": -1.16, "mr": 8.88, "type": "EWG", "category": "halogen"},
    "-I": {"name": "-I", "smiles": "*I", "sigma_para": 0.18, "sigma_meta": 0.35, "pi": 1.12, "es": -1.40, "mr": 13.94, "type": "EWG", "category": "halogen"},
    "-CF3": {"name": "-CF3", "smiles": "*C(F)(F)F", "sigma_para": 0.54, "sigma_meta": 0.43, "pi": 0.88, "es": -2.40, "mr": 5.02, "type": "EWG", "category": "alkyl"},
    "-OH": {"name": "-OH", "smiles": "*O", "sigma_para": -0.37, "sigma_meta": 0.12, "pi": -0.67, "es": -0.55, "mr": 2.85, "type": "EDG", "category": "hydroxyl"},
    "-OCH3": {"name": "-OCH3", "smiles": "*OC", "sigma_para": -0.27, "sigma_meta": 0.12, "pi": -0.02, "es": -0.55, "mr": 7.87, "type": "EDG", "category": "ether"},
    "-OC2H5": {"name": "-OC2H5", "smiles": "*OCC", "sigma_para": -0.24, "sigma_meta": 0.10, "pi": 0.38, "es": -0.55, "mr": 12.47, "type": "EDG", "category": "ether"},
    "-OCF3": {"name": "-OCF3", "smiles": "*OC(F)(F)F", "sigma_para": 0.35, "sigma_meta": 0.38, "pi": 1.04, "es": None, "mr": 7.87, "type": "EWG", "category": "ether"},
    "-NH2": {"name": "-NH2", "smiles": "*N", "sigma_para": -0.66, "sigma_meta": -0.16, "pi": -1.23, "es": -0.61, "mr": 5.42, "type": "EDG", "category": "amino"},
    "-NHCH3": {"name": "-NHCH3", "smiles": "*NC", "sigma_para": -0.84, "sigma_meta": -0.30, "pi": -0.47, "es": None, "mr": 10.33, "type": "EDG", "category": "amino"},
    "-N(CH3)2": {"name": "-N(CH3)2", "smiles": "*N(C)C", "sigma_para": -0.83, "sigma_meta": -0.15, "pi": 0.18, "es": None, "mr": 15.55, "type": "EDG", "category": "amino"},
    "-NO2": {"name": "-NO2", "smiles": "*[N+](=O)[O-]", "sigma_para": 0.78, "sigma_meta": 0.71, "pi": -0.28, "es": -2.52, "mr": 7.36, "type": "EWG", "category": "nitro"},
    "-CN": {"name": "-CN", "smiles": "*C#N", "sigma_para": 0.66, "sigma_meta": 0.56, "pi": -0.57, "es": -0.51, "mr": 6.33, "type": "EWG", "category": "cyano"},
    "-COOH": {"name": "-COOH", "smiles": "*C(=O)O", "sigma_para": 0.45, "sigma_meta": 0.36, "pi": -0.32, "es": -2.53, "mr": 6.93, "type": "EWG", "category": "carbonyl"},
    "-COCH3": {"name": "-COCH3", "smiles": "*C(=O)C", "sigma_para": 0.50, "sigma_meta": 0.38, "pi": -0.55, "es": None, "mr": 11.18, "type": "EWG", "category": "carbonyl"},
    "-CHO": {"name": "-CHO", "smiles": "*C=O", "sigma_para": 0.42, "sigma_meta": 0.35, "pi": -0.65, "es": None, "mr": 6.88, "type": "EWG", "category": "carbonyl"},
    "-CONH2": {"name": "-CONH2", "smiles": "*C(=O)N", "sigma_para": 0.36, "sigma_meta": 0.28, "pi": -1.49, "es": None, "mr": 9.81, "type": "EWG", "category": "carbonyl"},
    "-CONHCH3": {"name": "-CONHCH3", "smiles": "*C(=O)NC", "sigma_para": 0.36, "sigma_meta": 0.35, "pi": -1.27, "es": None, "mr": 14.43, "type": "EWG", "category": "carbonyl"},
    "-SO2NH2": {"name": "-SO2NH2", "smiles": "*S(=O)(=O)N", "sigma_para": 0.57, "sigma_meta": 0.46, "pi": -1.82, "es": None, "mr": 12.28, "type": "EWG", "category": "sulfonyl"},
    "-Ph": {"name": "-Ph", "smiles": "*c1ccccc1", "sigma_para": 0.01, "sigma_meta": 0.06, "pi": 1.96, "es": -3.82, "mr": 25.36, "type": "neutral", "category": "aryl"},
    "-NHAc": {"name": "-NHAc", "smiles": "*NC(=O)C", "sigma_para": 0.00, "sigma_meta": 0.21, "pi": -0.97, "es": None, "mr": 14.93, "type": "neutral", "category": "amino"},
    "-SCH3": {"name": "-SCH3", "smiles": "*SC", "sigma_para": 0.00, "sigma_meta": 0.15, "pi": 0.61, "es": -1.07, "mr": 13.82, "type": "neutral", "category": "thiol"},
    "-SH": {"name": "-SH", "smiles": "*S", "sigma_para": 0.15, "sigma_meta": 0.25, "pi": 0.39, "es": -1.07, "mr": 9.22, "type": "EWG", "category": "thiol"},
    "-NHSO2CH3": {"name": "-NHSO2CH3", "smiles": "*NS(=O)(=O)C", "sigma_para": 0.03, "sigma_meta": 0.20, "pi": -1.18, "es": None, "mr": 17.75, "type": "EWG", "category": "amino"}
}

def get_substituent_constants(name: str) -> Optional[Dict[str, Any]]:
    """Lookup substituent constants by name (case-insensitive, optional - prefix)."""
    cleaned = name.strip()
    if not cleaned.startswith("-"):
        cleaned = "-" + cleaned
    key_map = {k.upper(): k for k in SUBSTITUENT_LIBRARY.keys()}
    if cleaned.upper() in key_map:
        return SUBSTITUENT_LIBRARY[key_map[cleaned.upper()]]
    return None

def classify_substituent(name: str) -> str:
    """Classify a substituent as EDG, EWG, or neutral based on sigma_para."""
    data = get_substituent_constants(name)
    if data is None:
        return "unknown"
    sigma_para = data.get("sigma_para", 0.0)
    if sigma_para > 0.05:
        return "EWG"
    elif sigma_para < -0.05:
        return "EDG"
    else:
        return "neutral"

def get_all_substituents() -> pd.DataFrame:
    """Get the full substituent library as a pandas DataFrame."""
    return pd.DataFrame.from_dict(SUBSTITUENT_LIBRARY, orient="index")

def get_substituents_by_type(sub_type: str) -> List[Dict[str, Any]]:
    """Filter substituents by type (EDG, EWG, neutral)."""
    return [v for v in SUBSTITUENT_LIBRARY.values() if v.get("type", "").upper() == sub_type.upper()]

def get_substituents_by_category(category: str) -> List[Dict[str, Any]]:
    """Filter substituents by category (alkyl, halogen, amino, etc.)."""
    return [v for v in SUBSTITUENT_LIBRARY.values() if v.get("category", "").upper() == category.upper()]

def craig_plot_data() -> pd.DataFrame:
    """Returns DataFrame with columns: name, sigma_para, pi, ready for Craig plot."""
    df = get_all_substituents()
    return df[["name", "sigma_para", "pi"]]
