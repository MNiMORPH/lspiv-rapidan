# lspiv-rapidan
Perform large-scale particle-imaging velocimetry on data from the Blue Earth River near Rapidan Dam

## Pipeline outputs

Each video clip produces the following files under `results/{sample}/`:

| File | Description |
|------|-------------|
| `Frame.png` | Single projected video frame (visual reference) |
| `PIVquiverFrame.png` | Velocity quiver plot overlaid on the projected frame |
| `velocity.nc` | Full velocity dataset (NetCDF); archive/analysis copy |
| `velocity.tif` | Velocity GeoTIFF; 5-band float32 in the orthophoto CRS |

### GeoTIFF band layout (`velocity.tif`)

| Band | Name | Units | Description |
|------|------|-------|-------------|
| 1 | `speed_m_s` | m/s | Velocity magnitude √(v_x² + v_y²) |
| 2 | `v_x_m_s` | m/s | Eastward velocity component |
| 3 | `v_y_m_s` | m/s | Northward velocity component |
| 4 | `bearing_deg_cwN` | degrees | Flow bearing, clockwise from north (0° = north, 90° = east) |
| 5 | `corr` | — | PIV cross-correlation coefficient |
| 6 | `s2n` | — | Signal-to-noise ratio |

Band names are also stored as metadata tags within the GeoTIFF (`name=` tag on each band), readable by GDAL/rasterio.
