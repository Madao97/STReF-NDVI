# STReF-NDVI architecture
# STReF-NDVI

**Spatio-Temporal SAR–Optical Fusion Framework for Reconstructing NDVI Time Series under Persistent Cloud Cover**

[![Paper](https://img.shields.io/badge/Paper-Information%20Fusion-blue)](https://doi.org/10.1016/j.inffus.2026.104760)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.inffus.2026.104760-blue)](https://doi.org/10.1016/j.inffus.2026.104760)

STReF-NDVI is a spatio-temporal SAR–optical fusion framework designed to reconstruct
10 m NDVI time series under persistent cloud cover. It exploits Sentinel-1 SAR
observations together with sparse and irregular Sentinel-2 optical supervision,
combining spatial texture encoding, temporal graph modeling, Bi-LSTM, and
Transformer-based temporal attention to capture vegetation dynamics across prolonged
optical observation gaps.

## Publication

**STReF-NDVI: A spatio-temporal SAR-optical fusion framework for reconstructing NDVI time series under persistent cloud cover**

*Information Fusion*, 2026, Article 104760.

**DOI:** https://doi.org/10.1016/j.inffus.2026.104760

<img width="6470" height="5628" alt="Figure 4" src="https://github.com/user-attachments/assets/fd3901a2-e119-44be-900a-5ce4d64e346f" />


# STReF-NDVI: Spatio-Temporal SAR–Optical Fusion for NDVI Reconstruction

STReF-NDVI is a spatio-temporal SAR–optical fusion framework for reconstructing
10 m NDVI time series under persistent cloud cover using Sentinel-1 SAR observations
and quality-controlled Sentinel-2 NDVI supervision.

This repository provides the core implementation of STReF-NDVI to:
1) construct model-ready datasets from preprocessed Sentinel-1 features and Sentinel-2 NDVI observations,
2) train the STReF-NDVI model,
3) reconstruct continuous NDVI time series, and
4) quantitatively evaluate the reconstructed NDVI.

The repository focuses on the core reconstruction framework and starts from
preprocessed remote sensing data. Sensor-specific preprocessing and quality-control
procedures follow the protocol described in the published paper.

All scripts are CLI-friendly with English comments and generalized paths for GitHub use.
---
## Contents

- **01_Dataset_Construction.py** — Constructs STReF-NDVI datasets from
  preprocessed Sentinel-1 features and quality-controlled Sentinel-2 NDVI
  observations, including temporal refinement, validity masks, and dataset
  organization.

- **02_STReF-NDVI_Training.py** — Implements and trains the STReF-NDVI model,
  integrating Tex-CNN, Time-GCN, BiLSTM, and Transformer-based temporal
  modeling.

- **03_NDVI_Reconstruction.py** — Performs NDVI prediction and continuous
  time-series reconstruction using the trained STReF-NDVI model.

- **04_Evaluation_Dataset_Construction.py** — Constructs independent evaluation
  samples from quality-controlled optical NDVI observations.

- **05_Evaluation.py** — Evaluates reconstructed NDVI using R², RMSE, and MAE
  and generates quantitative evaluation plots.

---

## Data Preparation

This repository focuses on the STReF-NDVI reconstruction framework rather than
sensor-specific preprocessing utilities. Therefore, the released workflow starts
from preprocessed Sentinel-1 features and quality-controlled Sentinel-2 NDVI
observations.

The input data should be prepared following the protocol described in the paper.
In brief:

1. Sentinel-2 L2A surface reflectance is used to derive NDVI.
2. Cloud-, shadow-, and other contaminated optical observations are removed
   through quality control.
3. Rainfall-affected SAR observations are screened using precipitation
   information.
4. Sentinel-1 observations are spatially aligned with the Sentinel-2 10 m
   reference grid.
5. Sentinel-1 and valid Sentinel-2 observations are temporally matched according
   to the matching strategy described in the paper.
6. Quality-controlled optical NDVI observations are temporally refined before
   being used as supervision.

Please refer to the published paper for the complete preprocessing criteria,
parameter settings, and experimental protocol.
---

## Environment & Installation

> **Recommended:** Use Conda, install GDAL/GEOS/PROJ via `conda-forge`, then pip-install the rest.  
> **PyTorch & PyG:** Install following official wheels that match your CUDA.

```bash
# 1) Create env (Python >= 3.10 recommended)
conda create -n rendvival python=3.10 -y
conda activate rendvival

# 2) Geospatial stack via conda-forge (GDAL/GEOS/PROJ/Fiona)
conda install -c conda-forge gdal rasterio geopandas fiona pyproj shapely -y

# 3) PyTorch (choose CUDA/CPU per your machine)
# See https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# 4) PyTorch Geometric (match your torch/CUDA)
# See https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# Example:
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric

# 5) The rest
pip install -r requirements.txt

# Optional: CuPy (if you plan to use 01_S2_DataPreparation_2022-.py GPU path)
# Choose the package that matches your CUDA:
# pip install cupy-cuda12x
