"""
virtual_screen.py — Virtual substitution and screening workflow.

Core workflow:
1. User provides a parent compound SMILES with substitution positions marked
2. Tool generates all substituted analogs from the built-in library
3. Computes all descriptors for each analog
4. Predicts activity using the trained QSAR model
5. Ranks analogs by predicted activity
6. Outputs a full comparison table with SAR annotations
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from medchem.descriptors import compute_all_descriptors, tanimoto_similarity
from medchem.druglikeness import full_druglikeness_profile, lipinski_rule_of_5
from medchem.substituents import SUBSTITUENT_LIBRARY, get_all_substituents, craig_plot_data
from medchem.sar_engine import enumerate_rgroup_substitutions, extract_scaffold
from medchem.bioisosteres import generate_bioisosteric_analogs, list_replaceable_groups
from medchem.qsar_model import load_model, predict_activity


def virtual_substitution_screen(
    core_smiles: str,
    model_path: str = None,
    substituent_names: list[str] = None,
    include_bioisosteres: bool = True,
    top_n: int = 20,
    output_csv: str = None
) -> pd.DataFrame:
    """
    Full virtual substitution screening workflow.

    Given a core scaffold with [*] marking substitution position(s),
    generates all analogs, predicts activity, and ranks them.

    Args:
        core_smiles: SMILES with [*] at substitution position(s).
        model_path: Path to trained QSAR model.
        substituent_names: Specific substituents to use (None = all).
        include_bioisosteres: Also generate bioisosteric replacements.
        top_n: Number of top analogs to display in detail.
        output_csv: Path to save full results.

    Returns:
        DataFrame with all analogs ranked by predicted activity.
    """
    print("=" * 70)
    print("  VIRTUAL SUBSTITUTION SCREEN")
    print("=" * 70)
    print(f"\n  Core scaffold: {core_smiles}")

    # Load model if available
    model_dict = None
    try:
        model_dict = load_model(model_path)
        print(f"  Model: {model_dict.get('model_type', 'unknown')} "
              f"(trained on {model_dict.get('n_compounds', '?')} compounds, "
              f"CV R² = {model_dict.get('cv_r2', 0):.3f})")
    except FileNotFoundError:
        print("  ⚠ No trained model found — will compute descriptors only.")

    # Step 1: R-group enumeration
    print(f"\n  Step 1: Enumerating R-group substitutions...")
    analogs = enumerate_rgroup_substitutions(
        core_smiles,
        substituent_names=substituent_names
    )
    print(f"  Generated {len(analogs)} R-group analogs")

    # Step 2: Bioisosteric replacements (on the core itself, without [*])
    bioisostere_analogs = []
    if include_bioisosteres:
        # Try bioisosteric replacements on a representative analog (H-substituted)
        # or on the core if it's a complete molecule
        from rdkit import Chem
        test_mol = Chem.MolFromSmiles(core_smiles.replace('[*]', '[H]'))
        if test_mol is not None:
            clean_smiles = Chem.MolToSmiles(test_mol)
            replaceable = list_replaceable_groups(clean_smiles)
            if replaceable:
                print(f"\n  Step 2: Generating bioisosteric replacements...")
                print(f"  Replaceable groups found: {', '.join(replaceable)}")
                bio_results = generate_bioisosteric_analogs(clean_smiles)
                for br in bio_results:
                    br['source'] = 'bioisostere'
                    # Compute descriptors for bioisosteric analogs
                    desc = compute_all_descriptors(br['product_smiles'])
                    if desc is not None:
                        br['descriptors'] = desc
                        bioisostere_analogs.append(br)
                print(f"  Generated {len(bioisostere_analogs)} bioisosteric analogs")
            else:
                print("\n  Step 2: No replaceable groups found for bioisosteric substitution.")
        else:
            print("\n  Step 2: Could not parse core for bioisosteric analysis.")

    # Step 3: Compute descriptors and predict
    print(f"\n  Step 3: Computing descriptors and predicting activity...")
    rows = []

    for analog in analogs:
        row = {
            'SMILES': analog['product_smiles'],
            'Substituent': analog['substituent_name'],
            'Source': 'R-group',
        }
        desc = analog.get('descriptors', {})
        row.update(desc)

        # Drug-likeness quick check
        lip = lipinski_rule_of_5(analog['product_smiles'])
        if lip:
            row['Lipinski_Pass'] = lip.get('pass', False)
            row['Lipinski_Violations'] = lip.get('violations', 0)

        rows.append(row)

    for bio_analog in bioisostere_analogs:
        row = {
            'SMILES': bio_analog['product_smiles'],
            'Substituent': f"{bio_analog.get('original_group', '?')}→{bio_analog.get('replacement_name', '?')}",
            'Source': 'Bioisostere',
        }
        desc = bio_analog.get('descriptors', {})
        row.update(desc)

        lip = lipinski_rule_of_5(bio_analog['product_smiles'])
        if lip:
            row['Lipinski_Pass'] = lip.get('pass', False)
            row['Lipinski_Violations'] = lip.get('violations', 0)

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # Predict activity if model is available
    if model_dict is not None and not result_df.empty:
        try:
            smiles_list = result_df['SMILES'].tolist()
            pred_df = predict_activity(smiles_list, model_dict=model_dict)
            if not pred_df.empty:
                act_col = 'Activity' if 'Activity' in pred_df.columns else 'Predicted_Activity'
                if act_col in pred_df.columns:
                    result_df['Activity'] = pred_df[act_col].values
                    result_df['Predicted_Activity'] = pred_df[act_col].values
        except Exception as e:
            print(f"  ⚠ Prediction failed: {e}")
            result_df['Activity'] = np.nan
            result_df['Predicted_Activity'] = np.nan

    # Step 4: Rank
    if 'Predicted_Activity' in result_df.columns:
        result_df = result_df.sort_values('Predicted_Activity', ascending=False)
        print(f"\n  Step 4: Ranking {len(result_df)} analogs by predicted activity...")
    else:
        # Rank by QED if no model
        if 'QED' in result_df.columns:
            result_df = result_df.sort_values('QED', ascending=False)
            print(f"\n  Step 4: Ranking {len(result_df)} analogs by QED (no model available)...")

    # Step 5: Display top results
    print(f"\n  --- Top {min(top_n, len(result_df))} Analogs ---\n")
    display_cols = ['SMILES', 'Substituent', 'Source']
    if 'Predicted_Activity' in result_df.columns:
        display_cols.append('Predicted_Activity')
    for col in ['MW', 'LogP', 'QED', 'TPSA', 'Lipinski_Pass']:
        if col in result_df.columns:
            display_cols.append(col)

    available_cols = [c for c in display_cols if c in result_df.columns]
    top_df = result_df[available_cols].head(top_n)
    print(top_df.to_string(index=False))

    # SAR insights
    if 'Predicted_Activity' in result_df.columns and len(result_df) > 1:
        print("\n  --- SAR Insights ---")
        best = result_df.iloc[0]
        worst = result_df.iloc[-1]
        print(f"  Most active:  {best['Substituent']} → Activity = {best['Predicted_Activity']:.4f}")
        print(f"  Least active: {worst['Substituent']} → Activity = {worst['Predicted_Activity']:.4f}")

        # Property-activity correlations
        for prop in ['LogP', 'MW', 'TPSA', 'QED']:
            if prop in result_df.columns:
                corr = result_df['Predicted_Activity'].corr(result_df[prop])
                if not np.isnan(corr):
                    direction = "↑" if corr > 0 else "↓"
                    strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
                    print(f"  {prop} vs Activity: r = {corr:.3f} ({strength} {direction})")

    # Save
    if output_csv:
        result_df.to_csv(output_csv, index=False)
        print(f"\n  Full results saved to '{output_csv}'")

    print("\n" + "=" * 70)
    return result_df


def compare_compounds(smiles1: str, smiles2: str, model_path: str = None) -> dict:
    """
    Head-to-head comparison of two compounds.

    Returns a dict with side-by-side descriptors, drug-likeness, and similarity.
    """
    print("=" * 70)
    print("  COMPOUND COMPARISON")
    print("=" * 70)

    desc1 = compute_all_descriptors(smiles1)
    desc2 = compute_all_descriptors(smiles2)

    if desc1 is None:
        print(f"  ERROR: Invalid SMILES: {smiles1}")
        return {}
    if desc2 is None:
        print(f"  ERROR: Invalid SMILES: {smiles2}")
        return {}

    # Similarity
    sim = tanimoto_similarity(smiles1, smiles2)
    print(f"\n  Compound 1: {smiles1}")
    print(f"  Compound 2: {smiles2}")
    print(f"  Tanimoto Similarity: {sim:.3f}")

    # Side-by-side comparison
    print(f"\n  {'Property':<25} {'Compound 1':>15} {'Compound 2':>15} {'Δ (2-1)':>10}")
    print("  " + "-" * 67)

    comparison = {}
    props = ['MW', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotatableBonds', 'RingCount',
             'AromaticRings', 'FractionCSP3', 'QED', 'LogS_ESOL', 'MR']

    for prop in props:
        v1 = desc1.get(prop)
        v2 = desc2.get(prop)
        if v1 is not None and v2 is not None:
            delta = v2 - v1
            comparison[prop] = {'compound1': v1, 'compound2': v2, 'delta': delta}
            fmt = '.3f' if isinstance(v1, float) else 'd'
            print(f"  {prop:<25} {v1:>15{fmt}} {v2:>15{fmt}} {delta:>+10{fmt}}")

    # Drug-likeness
    dl1 = full_druglikeness_profile(smiles1)
    dl2 = full_druglikeness_profile(smiles2)

    print(f"\n  {'Filter':<25} {'Compound 1':>15} {'Compound 2':>15}")
    print("  " + "-" * 57)
    for filt in ['lipinski', 'veber', 'ghose', 'muegge', 'egan']:
        p1 = "PASS" if dl1.get(filt, {}).get('pass', False) else "FAIL"
        p2 = "PASS" if dl2.get(filt, {}).get('pass', False) else "FAIL"
        print(f"  {filt.capitalize():<25} {p1:>15} {p2:>15}")

    # Predict
    model_dict = None
    try:
        model_dict = load_model(model_path)
    except FileNotFoundError:
        pass

    if model_dict is not None:
        pred_df = predict_activity([smiles1, smiles2], model_dict=model_dict)
        if len(pred_df) == 2:
            a1 = pred_df['Predicted_Activity'].iloc[0]
            a2 = pred_df['Predicted_Activity'].iloc[1]
            print(f"\n  Predicted Activity (1): {a1:.4f}")
            print(f"  Predicted Activity (2): {a2:.4f}")
            print(f"  Activity Difference:    {a2 - a1:+.4f}")

    # Scaffold comparison
    sc1 = extract_scaffold(smiles1)
    sc2 = extract_scaffold(smiles2)
    if sc1 and sc2:
        print(f"\n  Scaffold (1): {sc1.get('murcko_scaffold', 'N/A')}")
        print(f"  Scaffold (2): {sc2.get('murcko_scaffold', 'N/A')}")
        same = sc1.get('murcko_scaffold') == sc2.get('murcko_scaffold')
        print(f"  Same scaffold: {'Yes' if same else 'No'}")

    print("\n" + "=" * 70)

    return {
        'smiles1': smiles1,
        'smiles2': smiles2,
        'similarity': sim,
        'comparison': comparison,
        'druglikeness1': dl1,
        'druglikeness2': dl2,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python virtual_screen.py <core_SMILES_with_[*]> [output.csv]")
        print("  python virtual_screen.py compare <SMILES1> <SMILES2>")
        print()
        print("Examples:")
        print("  python virtual_screen.py 'c1ccc([*])cc1'")
        print("  python virtual_screen.py compare 'CCO' 'CC(=O)O'")
        sys.exit(1)

    if sys.argv[1] == 'compare' and len(sys.argv) >= 4:
        compare_compounds(sys.argv[2], sys.argv[3])
    else:
        core = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else 'virtual_screen_results.csv'
        virtual_substitution_screen(core, output_csv=out)
