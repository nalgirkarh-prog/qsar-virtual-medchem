#!/usr/bin/env python3
"""
qsar_model.py — Backward-compatible wrapper.

This script calls into the medchem package to train the model.
The original bash alias `qsar_model` continues to work.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medchem.qsar_model import train_model

# Use the original data.csv by default
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data.csv"
)

print("Training QSAR model (enhanced medchem pipeline)...")
result = train_model(csv_path)

print(f"\nModel type:    {result['model_type']}")
print(f"Compounds:     {result['n_compounds']}")
print(f"Features used: {len(result['features'])}")
print(f"CV R²:         {result['cv_r2']:.4f}")
print(f"CV RMSE:       {result['cv_rmse']:.4f}")

if result.get('feature_importance'):
    print("\nTop feature importances:")
    fi = sorted(result['feature_importance'].items(), key=lambda x: -x[1])
    for name, imp in fi[:5]:
        print(f"  {name}: {imp:.4f}")

print("\nModel trained and saved successfully!")
