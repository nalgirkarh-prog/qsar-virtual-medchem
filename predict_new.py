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

from medchem.predict import predict_and_report

# Get the target CSV from command-line argument
input_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "new_compounds.csv"
)
output_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "predictions_output.csv"
)

predict_and_report(input_csv, output_csv=output_csv)
