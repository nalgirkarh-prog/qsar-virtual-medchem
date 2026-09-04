# Virtual MedChem Tool — QSAR & SAR Platform

A comprehensive virtual medicinal chemistry platform built on RDKit and scikit-learn. Transforms basic molecular structures into full pharmaceutical predictions: multi-parameter physicochemical profiling, automated substituent SAR enumeration, bioisosteric replacement, drug-likeness evaluation, and machine-learning QSAR modeling.

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
- **Metrics:** Cross-validated $R^2$, RMSE, MAE
- **Feature Selection:** Variance thresholding and descriptor importance ranking
- **Backward Compatible:** Seamlessly loads legacy 2-feature models and saves compatible checkpoints

---

## Command-Line Usage

The unified CLI can be invoked via the `medchem` command or directly via Python:

```bash
# Using the medchem CLI
medchem <command> [arguments]

# Or using the conda environment Python directly
/home/harshnalgirkar/miniconda3/envs/qsardb-env/bin/python /home/harshnalgirkar/qsar-setup/medchem/medchem_cli.py <command>
```

### Commands

| Command | Usage | Description |
|---|---|---|
| `profile` | `medchem profile <SMILES>` | Full physicochemical + drug-likeness profile |
| `train` | `medchem train <data.csv>` | Train QSAR model with CV and feature importance |
| `predict` | `medchem predict <compounds.csv>` | Predict activity + generate drug-likeness profiles |
| `substitute` | `medchem substitute <core_[*]>` | Enumerate 30 R-group analogs, predict & rank |
| `bioisostere` | `medchem bioisostere <SMILES>` | Generate bioisosteres & compare properties |
| `sar` | `medchem sar <compounds.csv>` | Full SAR analysis, MCS, and Matched Molecular Pairs |
| `compare` | `medchem compare <SMI1> <SMI2>` | Head-to-head property comparison table |
| `drugcheck` | `medchem drugcheck <SMILES>` | Run Lipinski, Veber, Ghose, Muegge, Egan, PAINS, Brenk |
| `constants` | `medchem constants <substituent>` | Look up Hammett $\\sigma$, Hansch $\\pi$, Taft $E_s$ |
| `craig-plot` | `medchem craig-plot` | Display Craig plot ($\\sigma$ vs $\\pi$) data & quadrants |
| `scaffold` | `medchem scaffold <SMILES>` | Extract Murcko scaffold, generic framework & side chains |
| `similarity` | `medchem similarity <SMI1> <SMI2>` | Calculate Morgan/MACCS/RDKit Tanimoto similarity |

---

## Examples

### 1. Profiling a Drug Molecule
```bash
medchem profile "CC(=O)Oc1ccccc1C(=O)O"
```

### 2. Virtual R-Group Screening on a Core Scaffold
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

### 3. Bioisosteric Replacement
```bash
medchem bioisostere "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
```

### 4. Head-to-Head Comparison
```bash
medchem compare "CCO" "CC(=O)O"
```

### 5. Substituent Constant Lookup
```bash
medchem constants CF3
medchem constants -NO2
```

---

## Backward Compatibility
Existing bash aliases continue to work without modification:
- `qsar_model` runs `/home/harshnalgirkar/qsar-setup/qsar_model.py` (enhanced pipeline)
- `predict_qsar` runs `/home/harshnalgirkar/qsar-setup/predict_new.py` (enhanced prediction)
