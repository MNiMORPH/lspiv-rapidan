# lspiv-rapidan pipeline — notes for Claude

## What this repo is

A Snakemake pipeline: drone video → stabilize → georeference → PIV → georeferenced
velocity maps. The pipeline is site-agnostic; `examples/rapidan/` is the worked
example. The Python package is installed as `lspiv` (entry points: `lspiv-stabilize`,
`lspiv-georeference`, `lspiv-lab-config`, `lspiv-piv`).

---

## How to run

The Snakefile discovers every video matching `video_extension` in `raw_dir`
automatically — no per-clip configuration changes needed.

```bash
snakemake --cores 4                                    # process everything
snakemake --cores 4 results/MY_CLIP/velocity_utm.png  # single clip
```

Always run inside the conda environment:

```bash
conda activate lspiv-env
```

To reproduce the Rapidan example, copy its config first:

```bash
cp examples/rapidan/config.yaml config/config.yaml
```

---

## Repository layout

```
src/lspiv/          Python package (pip install -e .)
  __init__.py       PROJ_DATA fix — must run before any rasterio import
  piv.py            core pipeline: _piv_chunked(), run_piv(), figures
  stabilize.py      lspiv-stabilize entry point
  georeference.py   lspiv-georeference (SIFT) and lspiv-lab-config
workflow/
  Snakefile         three rules: stabilize → georeference → piv
config/
  config.yaml       annotated generic template — copy and edit for new sites
examples/
  rapidan/
    config.yaml     Rapidan Dam site config
    NOTES.md        Rapidan clip inventory, SIFT failure history, sharing workflow
data/               gitignored: raw/, stabilized/, orthophoto.tif, camera_configs/
results/            gitignored: all pipeline outputs
```

---

## Key technical decisions (non-obvious)

**PROJ_DATA fix (`__init__.py`):** The base conda environment leaks a stale
`PROJ_DATA` path. The package `__init__.py` unconditionally overrides it to the
active environment's `share/proj` before any rasterio or pyproj import. This must
stay unconditional (not `setdefault`) — the base-env path causes schema errors.

**50-frame chunked PIV (`_piv_chunked`):** Grayscale frames load as float32
(~8 MB/frame at 3836×2102). Full clips (~600 frames) would need ~5 GB just for
raw frames, more after projection. The helper processes 50 frames at a time;
peak RAM stays ~1.6 GB.

**RGB frame optimization:** Only two projected RGB frames are needed per clip
(frame 0 for diagnostics, mid-frame for UTM background). Each is loaded with
`cv2.VideoCapture` and warped with `cv2.warpPerspective` individually.

**CV noisiness mask:** Primary land/water discriminator. CV = speed_std / speed × 100%.
Stationary land → very high CV; flowing water → bounded CV. Default 100%.
DSM elevation is a secondary/optional filter.

**Fixed figure layout:** All UTM figures use `subplots_adjust` + manual colorbar
axes (no `bbox_inches="tight"`) so the same geographic region falls at the same
pixel in every output — enabling direct toggle comparison.

**NumPy 2.x:** No pin. pyORC (which required NumPy < 2) was removed in v0.2.
OpenPIV is pure Python and supports NumPy 2.x natively. h5netcdf is used for
NetCDF output instead of netcdf4 to avoid a compiled-extension ABI warning.

---

## Georeferencing

SIFT (`lspiv-georeference`): matches a single extracted video frame against a
reference orthophoto. Quality depends on shared visual features — particularly
exposed rock, sediment, and fixed structures. Inspect
`results/<clip>/georeference_debug.png` and treat results with <~20 RANSAC
inliers as suspect.

Lab (`lspiv-lab-config`): uses known physical dimensions. No orthophoto needed.
Good for flumes or small sites with measurable extents.

For Rapidan-specific georeferencing issues, see `examples/rapidan/NOTES.md`.

---

## Adding a new site

1. Create `config/config.yaml` from `examples/rapidan/config.yaml` (or from
   `config/config.yaml` template).
2. Place videos in `data/raw/`.
3. For `sift` method: place orthophoto in `data/orthophoto.tif`.
4. Run `snakemake --cores N`.
5. Check `results/<clip>/georeference_debug.png` before trusting PIV output.
6. Document site-specific decisions in an `examples/<site>/NOTES.md`.

---

## README notes policy

Always add notes about processing decisions, georeferencing outcomes, parameter
choices, and failures to the *results* repository's README — not here. This
repo's README describes the tool; site-specific observations belong with the data.
For Rapidan: `~/dataanalysis/rapidan-lspiv/README.md`.
