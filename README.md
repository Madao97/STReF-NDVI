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
1) construct model-ready datasets from preprocessed Sentinel-1 features and Sentinel-2 NDVI observations.
2) train the STReF-NDVI model.
3) reconstruct continuous NDVI time series.
4) quantitatively evaluate the reconstructed NDVI.

The repository focuses on the core reconstruction framework and starts from
preprocessed remote sensing data. Sensor-specific preprocessing and quality-control
procedures follow the protocol described in the published paper.

All scripts are CLI-friendly with English comments and generalized paths for GitHub use.
---
## Contents

- **01_Dataset_Construction.py** — Builds model-ready datasets from aligned Sentinel-1 VV and VH imagery, Sentinel-2 NDVI and cloud masks, water masks, and a supplied temporal matching table. Computes five SAR features, applies percentile-based normalization, and generates supervision masks.
- **02_STReF-NDVI_Training.py** — Implements STReF-NDVI and trains selected SAR feature combinations using a pixel-wise training and validation split.
- **03_NDVI_Reconstruction.py** — Reconstructs NDVI sequences at eligible pixels using a trained model and exports prediction arrays, optional GeoTIFFs, and diagnostic comparisons.
- **04_Evaluation_Dataset_Construction.py** — Samples evaluation coordinates from valid optical labels within eligible pixels using NDVI-bin reservoir sampling.
- **05_Evaluation.py** — Calculates R², RMSE, and MAE at supplied evaluation coordinates and generates per-date and pooled density plots.

---
## Data Preparation

The released workflow starts from preprocessed and spatially aligned Sentinel-1 VV and VH imagery, Sentinel-2 NDVI and cloud masks, water masks, and an S1-to-S2 matching table.

Sensor-specific preprocessing should follow the protocol described in the paper. The following steps are performed before running the dataset construction script:

1. Derive NDVI from Sentinel-2 L2A surface reflectance and prepare cloud and shadow masks.
2. Prepare water masks for the optical acquisition dates.
3. Apply the precipitation-based SAR screening described in the paper.
4. Align all input rasters to the same 10 m reference grid, including the coordinate reference system, affine transform, and raster dimensions.
5. Prepare the Sentinel-1-to-Sentinel-2 matching table using the temporal matching rules described in the paper.

The dataset construction script uses the supplied matching table. It does not perform satellite data acquisition, sensor-specific preprocessing, or temporal refinement of optical observations. Sentinel-1 dates listed in the matching table are retained even when no Sentinel-2 supervision is available.

Please refer to the published paper for the complete preprocessing criteria, parameter settings, and experimental protocol.
---
## Example Results

### Spatial reconstruction

The animation shows a fixed land-cover map, matched Sentinel-2 true-color imagery, cloud-masked observed NDVI, and reconstructed NDVI over a demonstration area. The observed and reconstructed NDVI panels use the same color scale. Actual acquisition dates are displayed for the matched optical imagery.





### Vegetation-related NDVI time series

The animation shows mean reconstructed NDVI over eligible pixels in vegetation-related land-cover classes, including other vegetation, cropland, and herbaceous wetland in this demonstration area. Sentinel-2 points represent the mean of valid observed pixels at each accepted date, whose spatial coverage may vary over time.

<img width="1440" height="720" alt="vegetation_mean_ndvi" src="https://github.com/user-attachments/assets/867e9adc-0912-44ef-adb0-301e9664f879" />


These animations illustrate reconstruction behavior in a demonstration area. They do not replace the quantitative evaluation reported in the paper.
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
