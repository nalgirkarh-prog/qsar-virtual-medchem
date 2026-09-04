#!/usr/bin/env python3
"""
medchem_cli.py — Unified CLI for the Virtual MedChem Tool.

Commands:
    profile <SMILES>              Full physicochemical + drug-likeness profile
    train <data.csv>              Train QSAR model
    predict <compounds.csv>       Predict activity + full profile
    substitute <SMILES>           Generate R-group substituted analogs
    bioisostere <SMILES>          Generate bioisosteric replacements
    sar <compounds.csv>           SAR analysis on a compound set
    compare <SMILES1> <SMILES2>   Head-to-head comparison
    similarity <SMILES1> <SMILES2> Compute Tanimoto similarity
    scaffold <SMILES>             Extract Murcko scaffold
    constants <substituent>       Look up Hammett/Taft/Hansch constants
    craig-plot                    Display Craig plot data for all substituents
    drugcheck <SMILES>            Run all drug-likeness filters
"""

import sys
import os

# Ensure medchem package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_profile(args):
    """Full medchem profile for a single SMILES."""
    if not args:
        print("Usage: medchem profile <SMILES>")
        return
    from medchem.predict import generate_full_report, format_report_text
    from medchem.qsar_model import load_model

    smiles = args[0]
    model_dict = None
    try:
        model_dict = load_model()
    except FileNotFoundError:
        pass

    report = generate_full_report(smiles, model_dict=model_dict)
    print(format_report_text(report))


def cmd_train(args):
    """Train a QSAR model from a CSV file."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="medchem train",
        description="Train a QSAR model from a CSV file containing SMILES and Activity columns."
    )
    parser.add_argument("csv_path", help="Path to training CSV file")
    parser.add_argument(
        "--model", "--model-type", "-m",
        dest="model_type",
        choices=["random_forest", "random-forest", "rf", "linear", "ridge"],
        default="random_forest",
        help="Algorithm to train: random_forest (default), linear, or ridge"
    )

    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return

    try:
        parsed_args = parser.parse_args(args)
    except SystemExit:
        return

    from medchem.qsar_model import train_model

    csv_path = parsed_args.csv_path
    model_type = parsed_args.model_type
    if model_type in ("random-forest", "rf"):
        model_type = "random_forest"

    print(f"\n  Training {model_type} model from '{csv_path}'...")
    try:
        result = train_model(csv_path, model_type=model_type)
    except Exception as e:
        print(f"  ERROR: Failed to train model: {e}")
        return

    print(f"\n  === Training Results ===")
    print(f"  Model type:    {result['model_type']}")
    print(f"  Compounds:     {result['n_compounds']}")
    print(f"  Features:      {len(result['features'])}")
    print(f"  CV R²:         {result['cv_r2']:.4f}")
    print(f"  CV RMSE:       {result['cv_rmse']:.4f}")
    print(f"  CV MAE:        {result['cv_mae']:.4f}")

    if result.get('feature_importance'):
        print(f"\n  Feature Importance (top 10):")
        fi = sorted(result['feature_importance'].items(), key=lambda x: -x[1])
        max_val = max(abs(v) for _, v in fi) if fi else 1.0
        scale = 35.0 / max_val if max_val > 0 else 1.0
        for name, imp in fi[:10]:
            bar = "█" * max(1, int(abs(imp) * scale)) if abs(imp) > 0 else ""
            print(f"    {name:<25} {imp:.4f} {bar}")

    if result.get('warnings'):
        print(f"\n  ⚠ Model Validation Diagnostics & Alerts:")
        for w in result['warnings']:
            print(f"    • {w}")
    elif result.get('cv_r2', 0) >= 0.5 and result.get('n_compounds', 0) >= 20:
        print(f"\n  ✓ Model validation passed (q² >= 0.50, N >= 20). Satisfies standard QSAR criteria.")

    print(f"\n  Model saved successfully!")


def cmd_predict(args):
    """Predict activity for compounds in a CSV or a single SMILES."""
    if not args:
        print("Usage:")
        print("  medchem predict <SMILES>                  Predict activity for a single compound")
        print("  medchem predict <compounds.csv> [out.csv] Predict activity for a batch of compounds")
        return
    from medchem.predict import predict_and_report, generate_full_report, format_report_text
    from medchem.qsar_model import load_model

    target = args[0]
    if os.path.isfile(target) or target.endswith('.csv'):
        output_csv = args[1] if len(args) > 1 else 'predictions_output.csv'
        predict_and_report(target, output_csv=output_csv)
    else:
        model_dict = None
        try:
            model_dict = load_model()
        except FileNotFoundError:
            pass
        report = generate_full_report(target, model_dict=model_dict)
        print(format_report_text(report))



def cmd_substitute(args):
    """Generate R-group substituted analogs."""
    if not args:
        print("Usage: medchem substitute <SMILES_with_[*]> [output.csv]")
        print("Example: medchem substitute 'c1ccc([*])cc1'")
        return
    from medchem.virtual_screen import virtual_substitution_screen

    core_smiles = args[0]
    output_csv = args[1] if len(args) > 1 else 'substitution_results.csv'
    virtual_substitution_screen(core_smiles, output_csv=output_csv)


def cmd_bioisostere(args):
    """Generate bioisosteric replacements."""
    if not args:
        print("Usage: medchem bioisostere <SMILES>")
        return
    from medchem.bioisosteres import generate_bioisosteric_analogs, list_replaceable_groups
    from medchem.descriptors import compute_all_descriptors
    from medchem.druglikeness import lipinski_rule_of_5

    smiles = args[0]
    print("=" * 70)
    print(f"  BIOISOSTERIC REPLACEMENT ANALYSIS")
    print("=" * 70)
    print(f"\n  Input: {smiles}")

    # List replaceable groups
    groups = list_replaceable_groups(smiles)
    if not groups:
        print("  No replaceable functional groups found.")
        return

    print(f"  Replaceable groups: {', '.join(groups)}")

    # Generate analogs
    analogs = generate_bioisosteric_analogs(smiles)
    print(f"  Generated {len(analogs)} bioisosteric analogs\n")

    print(f"  {'Original Group':<18} {'Replacement':<18} {'Product SMILES':<35} {'Type'}")
    print("  " + "-" * 85)

    for a in analogs:
        orig = a.get('original_group', '?')
        repl = a.get('replacement_name', '?')
        prod = a.get('product_smiles', '?')
        rtype = a.get('replacement_type', '?')
        print(f"  {orig:<18} {repl:<18} {prod:<35} {rtype}")

    # Quick property comparison
    print(f"\n  --- Property Comparison ---\n")
    print(f"  {'Analog':<35} {'MW':>8} {'LogP':>8} {'TPSA':>8} {'QED':>8} {'Lipinski'}")
    print("  " + "-" * 80)

    orig_desc = compute_all_descriptors(smiles)
    if orig_desc:
        lip = lipinski_rule_of_5(smiles)
        lip_pass = "PASS" if lip and lip.get('pass') else "FAIL"
        print(f"  {'[PARENT] ' + smiles[:26]:<35} {orig_desc.get('MW',0):>8.1f} "
              f"{orig_desc.get('LogP',0):>8.2f} {orig_desc.get('TPSA',0):>8.1f} "
              f"{orig_desc.get('QED',0):>8.3f} {lip_pass}")

    for a in analogs:
        desc = compute_all_descriptors(a['product_smiles'])
        if desc:
            lip = lipinski_rule_of_5(a['product_smiles'])
            lip_pass = "PASS" if lip and lip.get('pass') else "FAIL"
            label = f"{a.get('replacement_name', '?')}"
            print(f"  {label:<35} {desc.get('MW',0):>8.1f} "
                  f"{desc.get('LogP',0):>8.2f} {desc.get('TPSA',0):>8.1f} "
                  f"{desc.get('QED',0):>8.3f} {lip_pass}")

    print("\n" + "=" * 70)


def cmd_sar(args):
    """SAR analysis on a compound set."""
    if not args:
        print("Usage: medchem sar <compounds.csv>")
        print("  CSV must have 'SMILES' and 'Activity' columns")
        return
    import pandas as pd
    from medchem.sar_engine import sar_summary, matched_molecular_pairs, find_mcs

    df = pd.read_csv(args[0])
    if 'SMILES' not in df.columns or 'Activity' not in df.columns:
        print("  ERROR: CSV must have 'SMILES' and 'Activity' columns")
        return

    compounds = [{'smiles': row['SMILES'], 'activity': row['Activity'],
                   'name': row.get('Name', row['SMILES'])}
                  for _, row in df.iterrows()]

    print("=" * 70)
    print("  STRUCTURE-ACTIVITY RELATIONSHIP ANALYSIS")
    print("=" * 70)

    # SAR summary
    summary = sar_summary(compounds)
    print(f"\n  Compounds analyzed: {summary.get('n_compounds', len(compounds))}")
    print(f"  Activity range: {summary.get('activity_min', '?'):.4f} — {summary.get('activity_max', '?'):.4f}")
    if summary.get('most_active'):
        print(f"  Most active:  {summary['most_active'].get('smiles', '?')} "
              f"(activity = {summary['most_active'].get('activity', '?'):.4f})")
    if summary.get('least_active'):
        print(f"  Least active: {summary['least_active'].get('smiles', '?')} "
              f"(activity = {summary['least_active'].get('activity', '?'):.4f})")

    # MCS
    smiles_list = df['SMILES'].tolist()
    if len(smiles_list) >= 2:
        mcs = find_mcs(smiles_list)
        if mcs:
            print(f"\n  Maximum Common Substructure: {mcs.get('mcs_smarts', 'N/A')}")
            print(f"  MCS atoms: {mcs.get('num_atoms', '?')}, bonds: {mcs.get('num_bonds', '?')}")

    # MMP
    pairs = [(row['SMILES'], row['Activity']) for _, row in df.iterrows()]
    mmps = matched_molecular_pairs(pairs)
    if mmps:
        print(f"\n  Matched Molecular Pairs: {len(mmps)}")
        for mmp in mmps[:5]:
            print(f"    {mmp.get('mol1_smiles', '?')} → {mmp.get('mol2_smiles', '?')}: "
                  f"ΔActivity = {mmp.get('activity_change', 0):+.4f}")

    print("\n" + "=" * 70)


def cmd_compare(args):
    """Head-to-head comparison of two compounds."""
    if len(args) < 2:
        print("Usage: medchem compare <SMILES1> <SMILES2>")
        return
    from medchem.virtual_screen import compare_compounds
    compare_compounds(args[0], args[1])


def cmd_similarity(args):
    """Compute Tanimoto similarity."""
    if len(args) < 2:
        print("Usage: medchem similarity <SMILES1> <SMILES2> [morgan|maccs|rdkit]")
        return
    from medchem.descriptors import tanimoto_similarity

    fp_type = args[2] if len(args) > 2 else 'morgan'
    sim = tanimoto_similarity(args[0], args[1], fp_type=fp_type)
    if sim is not None:
        print(f"  Tanimoto Similarity ({fp_type}): {sim:.4f}")
    else:
        print("  ERROR: Could not compute similarity (invalid SMILES?)")


def cmd_scaffold(args):
    """Extract Murcko scaffold."""
    if not args:
        print("Usage: medchem scaffold <SMILES>")
        return
    from medchem.sar_engine import extract_scaffold

    result = extract_scaffold(args[0])
    if result:
        print(f"  Input:            {args[0]}")
        print(f"  Murcko scaffold:  {result.get('murcko_scaffold', 'N/A')}")
        print(f"  Generic scaffold: {result.get('generic_scaffold', 'N/A')}")
        if result.get('side_chains'):
            print(f"  Side chains:      {', '.join(result['side_chains'])}")
    else:
        print(f"  ERROR: Could not extract scaffold from '{args[0]}'")


def cmd_constants(args):
    """Look up substituent constants."""
    if not args:
        print("Usage: medchem constants <substituent_name>")
        print("Example: medchem constants CH3")
        print("         medchem constants -NO2")
        return
    from medchem.substituents import get_substituent_constants, classify_substituent

    name = args[0]
    constants = get_substituent_constants(name)
    if constants is None:
        print(f"  Substituent '{name}' not found in library.")
        print(f"  Use 'medchem craig-plot' to see all available substituents.")
        return

    cls = classify_substituent(name)
    print(f"\n  Substituent: {constants.get('name', name)}")
    print(f"  Type: {cls}")
    print(f"  SMILES: {constants.get('smiles', 'N/A')}")
    print(f"\n  Hammett σ_para:  {constants.get('sigma_para', 'N/A')}")
    print(f"  Hammett σ_meta:  {constants.get('sigma_meta', 'N/A')}")
    print(f"  Hansch π:        {constants.get('pi', 'N/A')}")
    print(f"  Taft Es:         {constants.get('es', 'N/A')}")
    print(f"  Molar Refract.:  {constants.get('mr', 'N/A')}")
    print(f"  Category:        {constants.get('category', 'N/A')}")


def cmd_craig_plot(args):
    """Display Craig plot data."""
    from medchem.substituents import craig_plot_data

    df = craig_plot_data()
    print("\n  CRAIG PLOT DATA (σ_para vs π)")
    print("  " + "-" * 50)
    print(f"  {'Substituent':<15} {'σ_para':>10} {'π (Hansch)':>12} {'Type':<8}")
    print("  " + "-" * 50)

    for _, row in df.iterrows():
        sigma = row.get('sigma_para', 0)
        pi = row.get('pi', 0)
        if sigma < -0.1:
            stype = "EDG"
        elif sigma > 0.1:
            stype = "EWG"
        else:
            stype = "neutral"
        print(f"  {str(row.get('name', '?')):<15} {sigma:>10.2f} {pi:>12.2f} {stype:<8}")

    print("\n  Quadrant Guide:")
    print("  Q1 (+σ, +π): EWG + lipophilic (e.g., -CF3, -Cl)")
    print("  Q2 (-σ, +π): EDG + lipophilic (e.g., -CH3, -C2H5)")
    print("  Q3 (-σ, -π): EDG + hydrophilic (e.g., -NH2, -OH)")
    print("  Q4 (+σ, -π): EWG + hydrophilic (e.g., -NO2, -CN)")


def cmd_drugcheck(args):
    """Run all drug-likeness filters."""
    if not args:
        print("Usage: medchem drugcheck <SMILES>")
        return
    from medchem.druglikeness import full_druglikeness_profile

    profile = full_druglikeness_profile(args[0])
    if profile is None:
        print(f"  ERROR: Invalid SMILES '{args[0]}'")
        return

    print(f"\n  Drug-Likeness Check: {args[0]}")
    print("  " + "=" * 55)

    for filter_name in ['lipinski', 'veber', 'ghose', 'muegge', 'egan', 'pains', 'brenk']:
        fdata = profile.get(filter_name, {})
        if not fdata:
            continue
        status = "PASS ✓" if fdata.get('pass', False) else "FAIL ✗"
        print(f"\n  {filter_name.upper()}: {status}")
        details = fdata.get('details', [])
        for d in details:
            print(f"    • {d}")
        alerts = fdata.get('alerts', [])
        for a in alerts:
            print(f"    ⚠ {a}")


def print_usage():
    """Print main usage."""
    print("""
  ╔══════════════════════════════════════════════════════════════╗
  ║          Virtual MedChem Tool — v1.0.0                      ║
  ║    Comprehensive QSAR & SAR Platform                        ║
  ╚══════════════════════════════════════════════════════════════╝

  COMMANDS:

    profile <SMILES>               Full physicochemical + drug-likeness profile
    train <data.csv> [--model M]   Train QSAR model (random_forest, linear, ridge)
    predict <compounds.csv>        Predict activity + full profile
    substitute <SMILES_with_[*]>   Generate R-group substituted analogs
    bioisostere <SMILES>           Generate bioisosteric replacements
    sar <compounds.csv>            SAR analysis on a compound set
    compare <SMILES1> <SMILES2>    Head-to-head compound comparison
    similarity <SMILES1> <SMILES2> Compute Tanimoto similarity
    scaffold <SMILES>              Extract Murcko scaffold
    constants <substituent>        Look up Hammett/Taft/Hansch constants
    craig-plot                     Display Craig plot data
    drugcheck <SMILES>             Run all drug-likeness filters

  EXAMPLES:

    medchem profile 'c1ccccc1O'
    medchem train data.csv --model random_forest
    medchem train data.csv --model linear
    medchem train data.csv --model ridge
    medchem predict new_compounds.csv
    medchem substitute 'c1ccc([*])cc1'
    medchem bioisostere 'CC(=O)O'
    medchem compare 'CCO' 'CC(=O)O'
    medchem constants NO2
    medchem drugcheck 'CC(=O)Oc1ccccc1C(=O)O'
""")


COMMANDS = {
    'profile': cmd_profile,
    'train': cmd_train,
    'predict': cmd_predict,
    'substitute': cmd_substitute,
    'bioisostere': cmd_bioisostere,
    'sar': cmd_sar,
    'compare': cmd_compare,
    'similarity': cmd_similarity,
    'scaffold': cmd_scaffold,
    'constants': cmd_constants,
    'craig-plot': cmd_craig_plot,
    'drugcheck': cmd_drugcheck,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help', 'help'):
        print_usage()
        return

    command = sys.argv[1].lower()
    if command not in COMMANDS:
        print(f"  Unknown command: '{command}'")
        print_usage()
        return

    COMMANDS[command](sys.argv[2:])


if __name__ == '__main__':
    main()
