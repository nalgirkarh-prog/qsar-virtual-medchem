"""
predict.py — Enhanced prediction with full medchem profile reporting.

Loads a trained QSAR model and generates comprehensive medchem reports
for new compounds including activity predictions, drug-likeness, and
substituent analysis.
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Optional

# Add parent to path so medchem package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medchem.descriptors import compute_all_descriptors, compute_descriptors_batch
from medchem.druglikeness import (
    lipinski_rule_of_5, veber_rules, ghose_filter,
    muegge_filter, egan_filter, pains_filter, full_druglikeness_profile
)
from medchem.qsar_model import load_model, predict_activity


def generate_full_report(smiles: str, model_dict: dict = None) -> dict:
    """
    Generate a comprehensive medchem report for a single compound.

    Returns a dict with:
        - descriptors: full physicochemical profile
        - predicted_activity: from QSAR model (if model available)
        - druglikeness: all filter results
        - alerts: any structural alerts or warnings
    """
    report = {'smiles': smiles}

    # 1. Compute all descriptors
    desc = compute_all_descriptors(smiles)
    if desc is None:
        return {'smiles': smiles, 'error': f'Invalid SMILES: {smiles}'}
    report['descriptors'] = desc

    # 2. Predict activity if model is available
    if model_dict is not None:
        try:
            pred_df = predict_activity([smiles], model_dict=model_dict)
            if not pred_df.empty:
                report['predicted_activity'] = pred_df['Predicted_Activity'].iloc[0]
            else:
                report['predicted_activity'] = None
        except Exception as e:
            report['predicted_activity'] = None
            report['prediction_error'] = str(e)
    else:
        report['predicted_activity'] = None

    # 3. Drug-likeness profile
    report['druglikeness'] = full_druglikeness_profile(smiles)

    # 4. Key alerts
    alerts = []
    if desc.get('MW', 0) > 500:
        alerts.append('High molecular weight (>500)')
    if desc.get('LogP', 0) > 5:
        alerts.append('High lipophilicity (LogP >5)')
    if desc.get('TPSA', 0) > 140:
        alerts.append('High TPSA (>140) — poor oral absorption expected')
    if desc.get('RotatableBonds', 0) > 10:
        alerts.append('High flexibility (>10 rotatable bonds)')
    if desc.get('FractionCSP3', 0) < 0.25:
        alerts.append('Low Fsp3 (<0.25) — flat molecule, poor solubility risk')
    if desc.get('QED', 0) < 0.3:
        alerts.append('Low QED (<0.3) — poor drug-likeness')
    if desc.get('LogS_ESOL', 0) < -6:
        alerts.append('Very low predicted solubility (LogS < -6)')

    # PAINS
    pains = pains_filter(smiles)
    if pains and not pains.get('pass', True):
        alerts.append(f"PAINS alert(s): {', '.join(pains.get('alerts', []))}")

    report['alerts'] = alerts

    return report


def format_report_text(report: dict) -> str:
    """Format a report dict as a human-readable text block."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"  MEDCHEM PROFILE: {report['smiles']}")
    lines.append("=" * 70)

    if 'error' in report:
        lines.append(f"\n  ERROR: {report['error']}\n")
        return '\n'.join(lines)

    # Predicted Activity
    if report.get('predicted_activity') is not None:
        lines.append(f"\n  Predicted Activity: {report['predicted_activity']:.4f}")
    else:
        lines.append("\n  Predicted Activity: N/A (no model loaded)")

    # Key Descriptors
    desc = report.get('descriptors', {})
    lines.append("\n  --- Physicochemical Properties ---")
    props = [
        ('Molecular Weight', 'MW', '.2f'),
        ('LogP (Crippen)', 'LogP', '.3f'),
        ('Molar Refractivity', 'MR', '.2f'),
        ('TPSA', 'TPSA', '.2f'),
        ('H-Bond Donors', 'HBD', 'd'),
        ('H-Bond Acceptors', 'HBA', 'd'),
        ('Rotatable Bonds', 'RotatableBonds', 'd'),
        ('Ring Count', 'RingCount', 'd'),
        ('Aromatic Rings', 'AromaticRings', 'd'),
        ('Fraction Csp3', 'FractionCSP3', '.3f'),
        ('Heavy Atom Count', 'HeavyAtomCount', 'd'),
        ('Stereocenters', 'NumStereocenters', 'd'),
        ('LogS (ESOL)', 'LogS_ESOL', '.3f'),
        ('QED', 'QED', '.3f'),
    ]
    for label, key, fmt in props:
        val = desc.get(key)
        if val is not None:
            lines.append(f"  {label:.<30} {val:{fmt}}")

    # Drug-likeness summary
    dl = report.get('druglikeness', {})
    lines.append("\n  --- Drug-Likeness Filters ---")
    for filter_name in ['lipinski', 'veber', 'ghose', 'muegge', 'egan']:
        fdata = dl.get(filter_name, {})
        status = "PASS ✓" if fdata.get('pass', False) else "FAIL ✗"
        violations = fdata.get('violations', fdata.get('num_alerts', '?'))
        lines.append(f"  {filter_name.capitalize():.<20} {status}  (violations: {violations})")

    pains_data = dl.get('pains', {})
    pains_status = "PASS ✓" if pains_data.get('pass', True) else "FAIL ✗"
    lines.append(f"  {'PAINS':.<20} {pains_status}  (alerts: {pains_data.get('num_alerts', 0)})")

    # Alerts
    alerts = report.get('alerts', [])
    if alerts:
        lines.append("\n  --- Alerts & Warnings ---")
        for alert in alerts:
            lines.append(f"  ⚠ {alert}")
    else:
        lines.append("\n  No alerts — compound looks drug-like.")

    lines.append("\n" + "=" * 70)
    return '\n'.join(lines)


def predict_and_report(input_csv: str, model_path: str = None,
                       output_csv: str = None) -> pd.DataFrame:
    """
    Predict activity and generate full profiles for compounds in a CSV.

    Args:
        input_csv: Path to CSV with a 'SMILES' column.
        model_path: Path to saved model (optional).
        output_csv: Path to save results (optional).

    Returns:
        DataFrame with all predictions and descriptors.
    """
    df = pd.read_csv(input_csv)
    if 'SMILES' not in df.columns:
        raise ValueError("Input CSV must have a 'SMILES' column")

    # Try to load model
    model_dict = None
    try:
        model_dict = load_model(model_path)
        print(f"  Model loaded successfully ({model_dict.get('model_type', 'unknown')} "
              f"trained on {model_dict.get('n_compounds', '?')} compounds)")
    except FileNotFoundError:
        print("  No trained model found — skipping activity prediction.")
        print("  Run 'medchem train <data.csv>' to train a model first.")

    # Generate reports
    reports = []
    for smiles in df['SMILES']:
        report = generate_full_report(smiles, model_dict=model_dict)
        print(format_report_text(report))
        reports.append(report)

    # Build output DataFrame
    rows = []
    for r in reports:
        row = {'SMILES': r['smiles']}
        if 'error' not in r:
            row.update(r.get('descriptors', {}))
            row['Activity'] = r.get('predicted_activity')
            row['Predicted_Activity'] = r.get('predicted_activity')
            dl = r.get('druglikeness', {})
            row['Lipinski_Pass'] = dl.get('lipinski', {}).get('pass')
            row['Veber_Pass'] = dl.get('veber', {}).get('pass')
            row['QED'] = r.get('descriptors', {}).get('QED')
            row['Num_Alerts'] = len(r.get('alerts', []))
        rows.append(row)

    result_df = pd.DataFrame(rows)

    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"\n  Results saved to '{output_csv}'")

    return result_df


if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'new_compounds.csv'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'predictions_output.csv'

    # Resolve paths relative to qsar-setup directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(input_file):
        input_file = os.path.join(project_dir, input_file)
    if not os.path.isabs(output_file):
        output_file = os.path.join(project_dir, output_file)

    predict_and_report(input_file, output_csv=output_file)
