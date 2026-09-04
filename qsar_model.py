#!/usr/bin/env python3
"""
qsar_model.py — Backward-compatible wrapper.

This script calls into the medchem package to train the model.
The original bash alias `qsar_model` continues to work.
"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medchem.qsar_model import train_model

parser = argparse.ArgumentParser(
    description="Train QSAR model (backward-compatible wrapper)"
)
parser.add_argument(
    "csv_path",
    nargs="?",
    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv"),
    help="Path to training CSV file (default: data.csv)"
)
parser.add_argument(
    "--model", "--model-type", "-m",
    dest="model_type",
    choices=["random_forest", "random-forest", "rf", "linear", "ridge"],
    default="random_forest",
    help="Algorithm to train: random_forest (default), linear, or ridge"
)

args = parser.parse_args()
model_type = args.model_type
if model_type in ("random-forest", "rf"):
    model_type = "random_forest"

print(f"Training QSAR model ({model_type}, enhanced medchem pipeline)...")
result = train_model(args.csv_path, model_type=model_type)

print(f"\nModel type:    {result['model_type']}")
print(f"Compounds:     {result['n_compounds']}")
print(f"Features used: {len(result['features'])}")
print(f"CV R²:         {result['cv_r2']:.4f}")
print(f"CV RMSE:       {result['cv_rmse']:.4f}")
print(f"CV MAE:        {result['cv_mae']:.4f}")

if result.get('feature_importance'):
    print("\nTop feature importances:")
    fi = sorted(result['feature_importance'].items(), key=lambda x: -x[1])
    for name, imp in fi[:5]:
        print(f"  {name}: {imp:.4f}")

if result.get('warnings'):
    print("\n⚠ Model Validation Diagnostics & Alerts:")
    for w in result['warnings']:
        print(f"  • {w}")
elif result.get('cv_r2', 0) >= 0.5 and result.get('n_compounds', 0) >= 20:
    print("\n✓ Model validation passed (q² >= 0.50, N >= 20). Satisfies standard QSAR criteria.")

print("\nModel trained and saved successfully!")
