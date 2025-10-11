# ST-ClearNDVI architecture
ReNDVIval-RaST (Radar-driven NDVI Revival via Spatio-Temporal modeling) is a deep learning framework for reconstructing 10 m NDVI time series in cloud-prone regions. It fuses Sentinel-1 SAR with Sentinel-2 supervision, integrating CNN, temporal graph, Bi-LSTM, and Transformer attention for accurate and transferable vegetation monitoring.
<img width="1204" height="1714" alt="Figure 4" src="https://github.com/user-attachments/assets/574530fb-c1d8-4769-96ea-ee164c70ffd6" />



# ST-ClearNDVI: End-to-end S1→NDVI Reconstruction Pipeline

This repository contains a reproducible pipeline to:
1) preprocess Sentinel-2 & Sentinel-1 data,  
2) construct training/validation datasets,  
3) train the ReNDVIval-RaST model, and  
4) predict & evaluate reconstructed NDVI time series.

All scripts are CLI-friendly with English comments and generalized paths for GitHub use.
---

## Contents

- **01\_S2\_DataPreparation\_2022+.py** — S2 L2A batch preprocessing (ESA 2022 Radiometric Offset aware, SCL + Otsu shadow enhancement).  
- **01\_S2\_DataPreparation\_2022-.py** — S2 L2A preprocessing for pre-2022 products (with optional CuPy acceleration).  
- **01\_generate\_water\_mask\_from\_scl.py** — Convert raw SCL-based water mask to binary 0/1.  
- **02\_Climate.py** — Convert daily rainfall CSV to a binary “usable/disturbed” label.  
- **03\_S1\_Resampling.py** — Resample S1 rasters to match a S2 reference grid.  
- **04\_Clip.py** — Clip rasters by AOI vector boundary.  
- **05\_S2\_Data\_Availability\_Judgment.py** — Judge each S2 date’s usability by cloud ratio + rainfall (and optional WaterMask presence).  
- **06\_S1&S2\_Data\_Matching.py** — Match S1 acquisition dates to “usable” S2 dates within ±N days.  
- **07\_Dataset\_Construction.py** — Build training tensors (S1 features + smoothed NDVI + masks), with QA and temporal smoothing.  
- **08\_ST-ClearNDVI\_training.py** — Train the model (Tex-CNN + GCN + BiLSTM + Transformer).  
- **09\_Predict&Reconstruct\_complete\_NDVI.py** — Full NDVI reconstruction using trained weights, optional AOI masking.  
- **10\_Evaluation\_Dataset\_Construction.py** — Build validation coordinates from smoothed & clear NDVI.  
- **11\_Evaluation.py** — Quantitative evaluation & density scatter plots (R²/RMSE/MAE) per date and yearly.

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
