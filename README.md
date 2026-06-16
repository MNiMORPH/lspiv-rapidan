# lspiv-rapidan *(working name)*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20706919.svg)](https://doi.org/10.5281/zenodo.20706919)

A Snakemake pipeline for measuring surface water velocity from near-nadir drone
video. Three stages run automatically:

1. **Stabilize** — removes residual camera motion (stabilo library)
2. **Georeference** — ties the video frame to real-world coordinates (SIFT
   feature matching against an orthophoto, or known physical dimensions for lab
   use)
3. **PIV** — computes surface velocity via cross-correlation (OpenPIV)

Output: georeferenced velocity fields as GeoTIFF, NetCDF, GeoPackage, and a set
of map figures — all in UTM coordinates.

The repository is currently named after its first application (Rapidan Dam, MN)
but the pipeline is site-agnostic. See `examples/rapidan/` for a worked example.

---

## Installation

The geo-stack (rasterio, geopandas, pyproj) and Snakemake are best installed
via conda-forge to get pre-built binaries; everything else comes from pip.

```bash
conda create -n lspiv-env -c conda-forge -c bioconda python=3.12 \
    numpy scipy matplotlib geopandas rasterio pyproj shapely \
    xarray h5py snakemake jupyter pytest
conda activate lspiv-env
pip install openpiv stabilo matplotlib-scalebar opencv-python h5netcdf
pip install -e .
```

Dependencies are declared in `pyproject.toml`. NumPy 2.x is fully supported.

---

## Quick start

1. Place `.MP4` files in `data/raw/`.
2. Copy an example config and edit it for your site:
   ```bash
   cp examples/rapidan/config.yaml config/config.yaml
   ```
3. Run:
   ```bash
   snakemake --cores 4
   ```

The pipeline discovers all clips automatically via `glob_wildcards`. To process
a single clip:

```bash
snakemake --cores 4 results/MY_CLIP/velocity_utm.png
```

---

## Georeferencing methods

Set `georeference_method` in `config/config.yaml`:

| Method | When to use | What you need |
|---|---|---|
| `sift` | Field deployment with a reference orthophoto | A GeoTIFF orthophoto of the site at comparable flow conditions |
| `lab` | Lab flume or site with known dimensions | Physical width and height of the field of view (m) |

**SIFT note:** Matching quality depends heavily on shared visual features between
the video frame and the orthophoto. If they were acquired at very different flow
conditions, inlier counts may be too low for a valid homography. A minimum of
~20 RANSAC inliers is recommended; always inspect
`results/<clip>/georeference_debug.png`.

---

## Output files

Each processed clip produces a subdirectory under `results/`:

| File | Description |
|---|---|
| `velocity.tif` | 10-band GeoTIFF: speed, bearing, v\_x, v\_y, corr, s2n, speed\_std, v\_x\_std, v\_y\_std, speed\_cv\_pct |
| `velocity.nc` | Full velocity field as NetCDF |
| `velocity.gpkg` | Quality-filtered vectors (GeoPackage) |
| `frame_utm.tif` | Mid-clip video frame georeferenced to UTM |
| `velocity_utm.png` / `*_all.png` | Colored quiver on georeferenced frame (masked / unmasked) |
| `velocity_raster_utm.png` / `*_all.png` | Speed raster on frame |
| `velocity_raster_arrows_utm.png` / `*_all.png` | Speed raster + direction arrows |
| `velocity_std_utm.png` / `*_all.png` | Speed standard deviation |
| `velocity_cv_utm.png` / `*_all.png` | Coefficient of variation |
| `PIVquiverFrame.png` | Speed-colored quiver on projected UTM frame (all cells) |
| `PIVquiverFiltered.png` | Speed-colored quiver on projected UTM frame (quality-filtered cells only) |
| `georeference_debug.png` | SIFT feature match overlay |

Paired `*_utm.png` / `*_utm_all.png` files use identical pixel layout so they
can be toggled between in an image viewer for direct comparison.

---

## Configuration reference

See `config/config.yaml` for a fully annotated template.

Key parameters:

| Parameter | Default | Notes |
|---|---|---|
| `georeference_method` | `sift` | `sift` or `lab` |
| `orthophoto` | `data/orthophoto.tif` | Required for `sift` method |
| `piv.engine` | `numba` | Ignored — retained for config compatibility; OpenPIV is always used |
| `piv.window_size` | `10` | Interrogation window (px); 1 window ≈ 1 output cell |
| `piv.h_a` | `0.0` | Water surface elevation (m); update per clip if known |
| `piv.min_s2n` | `1.0` | Signal-to-noise threshold (OpenPIV peak2mean has low dynamic range; CV mask is the primary quality filter) |
| `piv.min_corr` | `0.5` | Cross-correlation threshold |
| `piv.min_speed` | `0.02` | Minimum speed (m/s) |
| `piv.cv_threshold` | `100.0` | Max CV (%) to classify a cell as water |

---

## Example: Rapidan Dam

`examples/rapidan/` contains the configuration and processing notes for Rapidan
Dam on the Blue Earth River, Martin County, MN. The dam failed on June 23, 2024;
footage spans peak flood through post-failure channel adjustment.

Processed results are in the companion repository:
[MNiMORPH/rapidan-lspiv](https://github.com/MNiMORPH/rapidan-lspiv)

---

## Dependencies

- [stabilo](https://github.com/mmaelicke/stabilo) — video stabilization
- [OpenPIV](https://openpiv.readthedocs.io) — cross-correlation PIV engine
- [Snakemake](https://snakemake.readthedocs.io) — workflow orchestration
- opencv-python, rasterio, pyproj, geopandas, xarray, netcdf4, scipy, matplotlib, matplotlib-scalebar
