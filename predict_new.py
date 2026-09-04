#!/usr/bin/env python3
"""
predict_new.py — Backward-compatible wrapper.

This script calls into the medchem package for enhanced prediction.
The original bash alias `predict_qsar` continues to work.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from medchem.predict import predict_and_report, generate_full_report, format_report_text
from medchem.qsar_model import load_model

if len(sys.argv) > 1:
    target = sys.argv[1]
    if os.path.isfile(target) or target.endswith('.csv'):
        output_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "predictions_output.csv"
        )
        predict_and_report(target, output_csv=output_csv)
    else:
        # Treat as SMILES string
        model_dict = None
        try:
            model_dict = load_model()
        except FileNotFoundError:
            pass
        report = generate_full_report(target, model_dict=model_dict)
        print(format_report_text(report))
else:
    # Default to new_compounds.csv
    input_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_compounds.csv")
    output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_output.csv")
    predict_and_report(input_csv, output_csv=output_csv)

