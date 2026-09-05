# Virtual MedChem Tool — QSAR & SAR Platform

A comprehensive virtual medicinal chemistry platform built on RDKit and scikit-learn. Transforms basic molecular structures into full pharmaceutical predictions: multi-parameter physicochemical profiling, automated substituent SAR enumeration, bioisosteric replacement, drug-likeness evaluation, and machine-learning QSAR modeling.

---

## Installation

### Option A — pip (any Python ≥ 3.9 with RDKit)

```bash
git clone https://github.com/nalgirkarh-prog/qsar-virtual-medchem.git
cd qsar-virtual-medchem
pip install -e .
```

> **Note:** RDKit must be installed first. The easiest way is via conda (see Option B).

### Option B — conda (recommended)

```bash
git clone https://github.com/nalgirkarh-prog/qsar-virtual-medchem.git
cd qsar-virtual-medchem

conda env create -f environment.yml
conda activate medchem-env
pip install -e .
```

### Option C — manual (no pip install)

```bash
conda install -c conda-forge rdkit
pip install scikit-learn pandas numpy scipy matplotlib joblib

# Add the CLI alias to ~/.bashrc or ~/.zshrc:
export QSAR_DIR="/path/to/qsar-virtual-medchem"
alias medchem="python $QSAR_DIR/medchem/medchem_cli.py"
source ~/.bashrc
```

### Verify installation

```bash
medchem profile "CC(=O)Oc1ccccc1C(=O)O"
```

---

## Key Capabilities

### 1. Molecular Property & Descriptor Calculator (`descriptors.py`)
Computes **30+** pharmaceutically relevant descriptors from SMILES:
- **Physicochemical:** MW, MolLogP (Crippen), Molar Refractivity (MR), TPSA, Labute Approximate Surface Area
- **Lipinski Parameters:** Hydrogen Bond Donors (HBD), Acceptors (HBA), Rotatable Bonds, Ro5 Violation Count
- **Structural / Shape:** Fraction Csp3, Heavy Atom Count, Stereocenters, Amide Bonds, Aromatic / Aliphatic / Saturated Rings
- **Topological & Graph:** Balaban J index, Bertz complexity (BertzCT), Kier-Hall Chi ($\\chi_0, \\chi_1$) and Kappa ($\\kappa_1, \\kappa_2, \\kappa_3$) indices, Hall-Kier Alpha
- **Aqueous Solubility:** LogS via the Delaney ESOL model
- **Drug-likeness Scores:** Quantitative Estimate of Drug-likeness (QED)
- **Electrotopological State:** PEOE_VSA descriptors (PEOE_VSA1 through PEOE_VSA6)
- **Fingerprints & Similarity:** Modern Morgan (ECFP4/6), MACCS keys, RDKit topological, and Tanimoto similarity

### 2. Substituent Library & Constants (`substituents.py`)
Built-in library of **30 common pharmaceutical substituents** with published empirical constants (Hansch, Leo, Hoekman 1995):
- **Hammett constants:** $\\sigma_{para}$ and $\\sigma_{meta}$ (electronic / resonance / inductive effects)
- **Hansch hydrophobic constant:** $\\pi$ (lipophilicity contribution)
- **Taft steric parameter:** $E_s$ (steric bulk)
- **Molar refractivity:** $MR$ (volume / polarizability contribution)
- **Classification:** Electron-Donating (EDG), Electron-Withdrawing (EWG), Neutral
- **Craig Plot generator:** Automatic $\\sigma_{para}$ vs $\\pi$ quadrant classification

### 3. Automated Substituent SAR Engine (`sar_engine.py`)
- **R-Group Enumeration:** Replaces dummy atoms (`[*]`) on a core scaffold with all 30 substituents using molecular graph transformations (preserving exact attachment points and chemistry)
- **Scaffold Analysis:** Extracts Bemis-Murcko scaffolds, generic carbon frameworks, and side chains
- **Maximum Common Substructure (MCS):** Identifies the conserved core across compound series
- **Matched Molecular Pairs (MMP):** Computes pairwise transformations and activity changes ($\\Delta\\text{Activity}$)
- **SAR Insights:** Calculates Pearson correlations between descriptors (LogP, MW, TPSA, QED) and activity

### 4. Bioisosteric Replacement Engine (`bioisosteres.py`)
Automated detection and replacement of vulnerable or sub-optimal functional groups using reaction SMARTS:
- **Carboxylic acid ($\\text{-COOH}$):** Tetrazole, acyl sulfonamide, hydroxamic acid, phosphonic acid
- **Amide ($\\text{-CONH-}$):** Sulfonamide, urea, reverse amide
- **Ester ($\\text{-COO-}$):** Amide, reverse ester
- **Phenyl ring:** Thiophene, 2-pyridyl, 3-pyridyl, 4-pyridyl, cyclopentyl
- **Phenol / Alcohol ($\\text{-OH}$):** Amino, thiol, fluoro, methoxy
- **Halogens:** $\\text{-Cl} \\leftrightarrow \\text{-F}$, $\\text{-Cl} \\leftrightarrow \\text{-Br}$, $\\text{-Cl} \\rightarrow \\text{-CF}_3$, $\\text{-F} \\rightarrow \\text{-H}$
- **Linkers:** Ether ($\\text{-O-}$) $\\leftrightarrow$ Amine ($\\text{-NH-}$) $\\leftrightarrow$ Thioether ($\\text{-S-}$) $\\leftrightarrow$ Methylene ($\\text{-CH}_2\\text{-}$)

### 5. Multi-Filter Drug-Likeness Evaluation (`druglikeness.py`)
Evaluates compounds against major medicinal chemistry filters:
- **Lipinski Rule of 5:** MW $\\le 500$, LogP $\\le 5$, HBD $\\le 5$, HBA $\\le 10$
- **Veber Rules:** Rotatable bonds $\\le 10$, TPSA $\\le 140 \\,\\text{\\AA}^2$ (oral bioavailability)
- **Ghose Filter:** $160 \\le \\text{MW} \\le 480$, $-0.4 \\le \\text{LogP} \\le 5.6$, $40 \\le \\text{MR} \\le 130$, $20 \\le \\text{atoms} \\le 70$
- **Muegge Filter:** Pharmacophore point filter for drug-like chemical space
- **Egan Egg:** Human intestinal absorption prediction (LogP vs TPSA)
- **PAINS:** Pan-Assay Interference Compounds (PAINS A, B, C filters)
- **Brenk Filter:** Reactive or toxic structural alerts

### 6. Machine Learning QSAR Pipeline (`qsar_model.py`)
- **Models:** Random Forest Regressor (with feature importance), Linear Regression, Ridge
- **Adaptive Validation:** Leave-One-Out CV for small series ($n < 10$), 5-Fold CV for larger sets
- **Metrics:** Cross-validated $R^2$ ($q^2$), RMSE, MAE
- **Statistical Validity Safeguards:** Automatic checks for sample size ($N \ge 20$), Topliss descriptor-to-sample ratio ($N/P \ge 5:1$), and Golbraikh-Tropsha predictive thresholds ($q^2 \ge 0.50$)
- **Feature Selection:** Variance thresholding and descriptor importance ranking
- **Backward Compatible:** Seamlessly loads legacy 2-feature models and saves compatible checkpoints

---

## Command-Line Usage

The unified CLI can be invoked via the `medchem` command or directly via Python:

```bash
# Using the medchem CLI (after pip install -e .)
medchem <command> [arguments]

# Or invoke directly without installing
python medchem/medchem_cli.py <command>
```


### Commands

| Command | Usage | Description |
|---|---|---|
| `profile` | `medchem profile <SMILES>` | Full physicochemical + drug-likeness profile |
| `train` | `medchem train <data.csv> [--model M]` | Train QSAR model (`random_forest`, `linear`, `ridge`) |
| `predict` | `medchem predict <SMILES>` or `<compounds.csv>` | Predict activity + generate drug-likeness profiles |
| `substitute` | `medchem substitute <core_[*]>` | Enumerate 30 R-group analogs, predict & rank |
| `bioisostere` | `medchem bioisostere <SMILES>` | Generate bioisosteres & compare properties |
| `sar` | `medchem sar <compounds.csv>` | Full SAR analysis, MCS, and Matched Molecular Pairs |
| `compare` | `medchem compare <SMI1> <SMI2>` | Head-to-head property comparison table |
| `drugcheck` | `medchem drugcheck <SMILES>` | Run Lipinski, Veber, Ghose, Muegge, Egan, PAINS, Brenk |
| `constants` | `medchem constants <substituent>` | Look up Hammett $\sigma$, Hansch $\pi$, Taft $E_s$ |
| `craig-plot` | `medchem craig-plot` | Display Craig plot ($\sigma$ vs $\pi$) data & quadrants |
| `scaffold` | `medchem scaffold <SMILES>` | Extract Murcko scaffold, generic framework & side chains |
| `similarity` | `medchem similarity <SMI1> <SMI2>` | Calculate Morgan/MACCS/RDKit Tanimoto similarity |

---

## Examples

### 1. Checking QSAR Activity for a New Compound

#### Single Molecule (CLI):
Pass any SMILES string directly to `medchem predict` or `medchem profile`:
```bash
# Direct activity prediction + full medchem profile:
medchem predict "CC(=O)Oc1ccccc1C(=O)O"

# Or using the profile command:
medchem profile "CC(=O)Oc1ccccc1C(=O)O"

# Or using the wrapper script / alias:
python predict_new.py "CC(=O)Oc1ccccc1C(=O)O"
```
**Output includes:**
- **Predicted Activity** (from the trained QSAR model)
- **30+ Physicochemical & Topological Descriptors** (MW, LogP, TPSA, HBD, HBA, RotBonds, Fsp3, Delaney LogS, QED)
- **Drug-Likeness Filters** (Lipinski Ro5, Veber, Ghose, Muegge, Egan egg, PAINS)
- **Structural Alerts & Warnings** (solubility risks, reactive groups, high flexibility)

#### Batch Prediction from CSV (CLI):
To predict activity for multiple compounds at once, prepare a CSV file with a `SMILES` column (e.g. `new_compounds.csv`):
```bash
# Predict for all compounds in CSV (saves to predictions_output.csv):
medchem predict new_compounds.csv

# Or specify a custom output filename:
medchem predict new_compounds.csv custom_results.csv
```

> **Note:** Batch prediction CSV outputs include both `Activity` and `Predicted_Activity` columns, allowing prediction output files like `predictions_output.csv` to be passed directly into `medchem sar` for immediate SAR analysis.

#### In Python:
```python
from medchem.qsar_model import predict_activity

# Predict activity for a list of SMILES:
df = predict_activity(["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1O"])
print(df[["SMILES", "Activity", "Predicted_Activity", "MW", "LogP", "TPSA"]])
```

### 2. Training QSAR Models & Statistical Validation

Train machine learning models (`random_forest`, `ridge`, `linear`) from CSV files containing `SMILES` and `Activity` columns:

```bash
# Train Random Forest (default, supports --model random-forest or random_forest)
medchem train data.csv --model random-forest

# Train Ridge Regression (L2 regularized)
medchem train data.csv --model ridge

# Train Ordinary Least Squares Linear Regression
medchem train data.csv --model linear
```

#### Dataset Sizing & Statistical Validity Guidelines
In accordance with OECD Principles for QSAR Validation and Golbraikh–Tropsha criteria:
- **Sample Size ($N \ge 20$):** Machine learning models require sufficient data to learn generalizable structure-activity relationships. Small series ($N < 10$) trigger automatic warnings.
- **Topliss Descriptor Ratio ($N/P \ge 5:1$):** At least 5 compounds per descriptor feature are recommended to avoid chance correlation.
- **Acceptance Criteria ($q^2 \ge 0.50$):** Cross-validated $R^2$ ($q^2$) must be positive and $\ge 0.50$ for acceptable predictive reliability.

#### Benchmark Performance on 1,500-Compound Dataset (`data.csv`):

| Algorithm | Flag | Compounds ($N$) | Features ($P$) | Topliss Ratio | CV $R^2$ ($q^2$) | CV RMSE | CV MAE | Status |
|---|---|---|---|---|---|---|---|---|
| **Random Forest** | `--model random-forest` | 1,500 | 16 | 93.8 : 1 | **0.9038** | **0.3861** | **0.2769** | ✓ Passed ($q^2 \ge 0.50$) |
| **Ridge** | `--model ridge` | 1,500 | 16 | 93.8 : 1 | **0.7723** | **0.5942** | **0.4177** | ✓ Passed ($q^2 \ge 0.50$) |
| **Linear** | `--model linear` | 1,500 | 16 | 93.8 : 1 | **0.7720** | **0.5946** | **0.4190** | ✓ Passed ($q^2 \ge 0.50$) |

### 3. Virtual R-Group Screening on a Core Scaffold
Mark the desired substitution site with `[*]`:
```bash
medchem substitute "c1ccc([*])cc1"
```
Output:
- Generates 30 analogs across alkyl, halogen, amino, ether, nitro, cyano, carbonyl, sulfonyl classes
- Computes descriptors and predicts activity
- Ranks analogs by predicted activity
- Calculates property-activity correlations (e.g. $\\text{LogP} \\rightarrow \\text{Activity}$)
- Saves full table to `substitution_results.csv`

### 4. Bioisosteric Replacement
```bash
medchem bioisostere "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
```

### 5. Head-to-Head Comparison
```bash
medchem compare "CCO" "CC(=O)O"
```

### 6. Substituent Constant Lookup
```bash
medchem constants CF3
medchem constants -NO2
```

---

## Backward Compatibility

The root-level wrapper scripts continue to work for pre-existing workflows:

```bash
python qsar_model.py              # Train (calls medchem.qsar_model)
python predict_new.py <SMILES>    # Predict (calls medchem.predict)
```

---

## Python API

```python
from medchem import profile

result = profile("CC(=O)Oc1ccccc1C(=O)O")
print(result["descriptors"])
print(result["druglikeness"])

# Descriptor computation
from medchem.descriptors import compute_all_descriptors
desc = compute_all_descriptors("c1ccccc1")

# Drug-likeness
from medchem.druglikeness import full_druglikeness_profile
dl = full_druglikeness_profile("CC(=O)Oc1ccccc1C(=O)O")

# R-group enumeration
from medchem.sar_engine import enumerate_rgroup_substitutions
analogs = enumerate_rgroup_substitutions("c1ccc([*])cc1")

# Bioisosteres
from medchem.bioisosteres import generate_bioisosteric_analogs
bioisosteres = generate_bioisosteric_analogs("CC(C)Cc1ccc(C(C)C(=O)O)cc1")
```

---

## Project Structure

```
qsar-virtual-medchem/
├── medchem/
│   ├── __init__.py          # Package init & convenience API
│   ├── descriptors.py       # 30+ molecular descriptors & fingerprints
│   ├── druglikeness.py      # Lipinski, Veber, Ghose, Muegge, Egan, PAINS, Brenk
│   ├── substituents.py      # Hammett/Hansch/Taft constants library (30 substituents)
│   ├── sar_engine.py        # R-group enumeration, scaffold, MCS, MMP
│   ├── bioisosteres.py      # Bioisosteric replacement engine (reaction SMARTS)
│   ├── qsar_model.py        # ML QSAR pipeline (RF / Linear / Ridge)
│   ├── predict.py           # Report generator
│   ├── virtual_screen.py    # Virtual screening & comparison
│   └── medchem_cli.py       # Unified 12-command CLI
├── data/
│   ├── benchmark_qsar.csv   # Curated 80-compound QSAR benchmark (kinase inhibitors)
│   ├── training_data.csv    # Example training data
│   └── substituent_library.csv
├── models/                  # Saved QSAR model checkpoints
├── qsar_model.py            # Backward-compatible wrapper (trains model)
├── predict_new.py           # Backward-compatible wrapper (predicts)
├── setup.py                 # Pip-installable package setup
├── requirements.txt         # Pip dependencies
├── environment.yml          # Conda environment spec
└── LICENSE                  # MIT License
```

---

## Dependencies

| Package | Minimum Version | Purpose |
|---|---|---|
| `rdkit` | ≥ 2023.03 | Cheminformatics core |
| `scikit-learn` | ≥ 1.3 | QSAR ML models |
| `pandas` | ≥ 2.0 | Data handling |
| `numpy` | ≥ 1.24 | Array computation |
| `scipy` | ≥ 1.10 | Statistics |
| `matplotlib` | ≥ 3.7 | Visualization |
| `joblib` | ≥ 1.3 | Model serialization |

---

## License

MIT — see [LICENSE](LICENSE).
