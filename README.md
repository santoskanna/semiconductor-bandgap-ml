# Predicting Semiconductor Band Gaps: DFT + Machine Learning

A project combining real quantum-mechanical simulation (DFT) with a machine learning model to predict semiconductor band gaps from structural descriptors.

## Overview

This project has two connected parts:

1. **DFT Validation (Quantum ESPRESSO):** Ran a real Density Functional Theory calculation on Silicon to compute its band gap from first principles, validating the computational pipeline against known database values.
2. **ML Prediction Pipeline:** Built a dataset of 150 semiconductors/insulators from the Materials Project database, engineered structural/chemical descriptors, and trained a Random Forest model to predict band gap.

## Results

### DFT Validation (Silicon)
- Computed band gap: **0.578 eV**
- Materials Project's own DFT value: **0.610 eV** (close match, validates pipeline)
- Experimental value: **1.12 eV**
- The gap between computed and experimental values reflects the well-known DFT-GGA band gap underestimation problem, not a calculation error.

### ML Model (150 materials)
- 5-Fold CV MAE: 0.742 +/- 0.079 eV
- Held-out Test MAE: 0.791 eV
- R2: 0.264
- Top predictive feature: electronegativity difference (matches known solid-state chemistry: more ionic bonding correlates with wider band gaps)

## Repository Structure
scripts/ - Data fetching, feature extraction, EDA, and model training scripts
data/ - Raw and processed datasets (pickled structures, feature CSVs)
results/ - Generated plots (EDA, correlation heatmap, predicted vs actual)
dft/ - Quantum ESPRESSO input/output files for Silicon DFT calculation

## Methodology

**Data source:** Materials Project API (materialsproject.org) — pulled 150 stable, non-metallic compounds with band gaps between 0.01-6.0 eV.

**Features used:** Lattice constants, cell volume, mean atomic number, electronegativity difference, mean electronegativity, mean atomic radius, number of elements.

**Model:** Random Forest Regressor (scikit-learn), evaluated with 5-fold cross-validation and a held-out test set.

**DFT setup:** Quantum ESPRESSO, PBE functional, SSSP pseudopotentials, plane-wave cutoff 40 Ry, 8x8x8 (SCF) / 16x16x16 (NSCF) k-point mesh.

## Known Limitations

- Dataset size (150 materials) is small for ML by field standards; R2 would likely improve substantially with a larger dataset and additional composition-based descriptors (e.g., Magpie features).
- DFT-GGA is known to systematically underestimate band gaps; both the training data and model predictions inherit this limitation.

## Tools Used

Python, pymatgen, mp-api, scikit-learn, pandas, numpy, matplotlib, seaborn, Quantum ESPRESSO, ASE
