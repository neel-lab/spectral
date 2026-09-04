# Spectral Flow Cytometry Data Analysis

This repository contains code for downstream processing and analysis of spectral flow cytometry data, with a focus on lectin-binding and glycogene-expression datasets.

## Scripts

### `KNN_commented.py`

`KNN_commented.py` interpolates missing lectin-panel values using a conserved antibody backbone.

The script takes CSV files generated from spectral flow cytometry experiments as input. Input files must be formatted such that each row represents a single cell, and each column represents a measured fluorophore corresponding to either a lectin or an antibody.

The program can be run in either **3-panel** or **5-panel** mode, depending on the experimental design. A K-nearest neighbors model is trained using antibody-binding features that are common across all samples. This model is then used to interpolate missing values in the remaining lectin columns.

The output is a single consolidated CSV file in which missing lectin values have been filled in.

---

### `Differential_glycogenes.py`

`Differential_glycogenes.py` processes differential glycogene-expression data between two cell types.

The input data are expected to be generated using the CellxGene differential-expression tool. The script generates bar plots summarizing differential glycogene expression based on the input dataset.

---

### `BM_vs_PBMC_commented.py`

`BM_vs_PBMC_commented.py` provides a pipeline for comparing spectral flow cytometry data between two cell populations: **bone marrow** and **peripheral blood mononuclear cells (PBMCs)**.

The script maps to two input folders, each containing Parquet files for one cell type. Each folder may contain multiple files, which are treated as biological or technical replicates.

Input files must be formatted such that each row represents a single cell, and columns include antibody-binding data, lectin-binding data, and cell-type annotations.

Fluorescence-intensity data are normalized using an arcsinh transformation.

The script generates the following outputs:

- PCA plots showing the distribution of samples from each tissue type based on lectin-binding data
- Bubble plots showing differential lectin binding between tissue types, including statistical significance
- Box plots showing lectin-binding distributions for each cell type  
