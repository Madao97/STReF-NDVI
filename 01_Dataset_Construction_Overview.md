# Dataset Construction Overview

This document outlines the dataset preparation workflow for STReF-NDVI.
The full dataset construction implementation is not included in this release.

## Overview

The dataset follows the Sentinel-1 observation dates recorded in the
matching table. Available Sentinel-2 NDVI observations provide reference
labels. Sentinel-1 observations without valid optical labels remain
in the temporal sequence.

Input rasters must already share the same spatial grid.

## Conceptual Workflow

```text
Inputs:
    Sentinel-1 observations
    Sentinel-2 NDVI observations and quality masks
    Water masks
    SAR–optical matching records

1. Read the matching records and organize Sentinel-1 observation dates.
2. Load Sentinel-1 observations and construct the required SAR features.
3. Load corresponding optical reference data when available.
4. Identify valid reference labels using observation-quality criteria.
5. Retain Sentinel-1 dates without optical supervision,
   marking their reference labels as unavailable.
6. Assemble aligned feature sequences, reference arrays and masks.
7. Normalize SAR features and record the transformation parameters.
8. Identify pixels eligible for supervised sample construction.
9. Export dataset arrays, dates and metadata.

Outputs:
    SAR feature sequences
    Optical reference observations
    Validity and supervision masks
    Temporal metadata and normalization parameters
```

## Output Files

| File | Shape | Description |
| --- | --- | --- |
| `S1.npy` | `(T, 5, H, W)` | Normalized SAR feature sequences |
| `S2.npy` | `(T, H, W, 3)` | NDVI, cloud and water information |
| `target_mask.npy` | `(T, H, W)` | Valid optical supervision locations |
| `S1_valid.npy` | `(T, H, W)` | SAR validity flags from preprocessing |
| `good_pixel.npy` | `(H, W)` | Pixels eligible for supervised sampling |
| `dates.csv` | — | SAR dates and associated optical label dates |
| `metadata.json` | — | Dimensions, channel order and processing metadata |
| `normalization.pkl` | — | Saved SAR normalization parameters |

Here, T denotes the number of temporal observations, and H and W
denote the spatial dimensions.

## Usage Notes

- Read the channel order and normalization information from the metadata.
- Missing optical labels must not be interpreted as zero NDVI.
- Use the supervision mask when evaluating against optical reference data.
- Do not normalize the supplied normalized arrays a second time.
