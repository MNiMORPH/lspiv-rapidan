import argparse
import json
import os
import subprocess
import sys

# conda's PROJ C library (9.7+) ships a newer proj.db (layout v1.6) than pyproj
# bundles (v1.4), so proj_context_set_database_path fails when using `conda run`
# (which skips activation scripts that set PROJ_DATA). Point pyproj at the conda
# env's database before importing pyorc, which triggers the first CRS lookup.
_conda_proj_data = os.path.join(sys.prefix, "share", "proj")
if os.path.isdir(_conda_proj_data):
    import pyproj.datadir
    pyproj.datadir.set_data_dir(_conda_proj_data)

import cv2
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; figures are saved to disk only
import matplotlib.pyplot as plt
import numpy as np
import pyorc
import pyproj
import rasterio
import rasterio.transform
from scipy.interpolate import griddata
from shapely.geometry import Point


def _placeholder_camera_config(width, height, crs=32615):
    """Build a camera config from pixel-scaled GCPs.

    This is a stand-in until a real georeferenced config is provided.
    Real GCPs (surveyed or from GPS-tagged imagery) should be supplied
    via --camera-config instead.
    """
    gcps = {
        "src": [[1, height], [1, 1], [width, 1], [width, height]],
        "dst": [
            [0, 0],
            [0, height / 100.0],
            [width / 100.0, height / 100.0],
            [width / 100.0, 0],
        ],
        "h_ref": 0.0,
        "z_0": 0.0,
    }
    camera_config = pyorc.CameraConfig(height=height, width=width, gcps=gcps, crs=crs)
    camera_config.set_bbox_from_corners(
        [[0, height], [width, height], [width, 0], [0, 0]]
    )
    return camera_config


def _crs_from_ds(ds_mean):
    cc = json.loads(ds_mean.attrs["camera_config"])
    return pyproj.CRS(cc["crs"])


def _utm_coords(ds_mean):
    """Return absolute UTM coordinate arrays (2-D, shape ny×nx)."""
    return ds_mean.xs.values, ds_mean.ys.values


def _velocity_arrays(ds_mean):
    v_x = ds_mean["v_x"].values
    v_y = ds_mean["v_y"].values
    speed   = np.sqrt(v_x**2 + v_y**2)
    bearing = (90.0 - np.degrees(np.arctan2(v_y, v_x))) % 360.0
    corr    = ds_mean["corr"].values
    s2n     = ds_mean["s2n"].values
    return v_x, v_y, speed, bearing, corr, s2n


def _dsm_water_mask(ds_mean, dsm_path, water_elev_m=None, elev_tolerance=0.5):
    """Boolean (ny, nx) mask — True where the DSM elevation indicates water.

    When water_elev_m is None (auto mode), the lower bound is the 5th-percentile
    elevation and the upper bound is the 90th-percentile elevation of the domain.
    This captures the full waterfall relief while excluding the highest rock walls.
    When water_elev_m is supplied, cells at or below that value plus elev_tolerance
    are kept (suitable for a flat-water reach with a known gauge stage).
    """
    xs, ys = _utm_coords(ds_mean)
    coords = list(zip(xs.flatten(), ys.flatten()))

    with rasterio.open(dsm_path) as src:
        nodata = src.nodata
        elev = np.array([v[0] for v in src.sample(coords)]).reshape(xs.shape)

    if nodata is not None:
        elev[elev == nodata] = np.nan

    if water_elev_m is None:
        lower = float(np.nanpercentile(elev, 5))
        upper = float(np.nanpercentile(elev, 90))
        print(f"DSM: auto-detected elevation range {lower:.2f}–{upper:.2f} m "
              f"(5th–90th percentile of domain)")
        mask = elev <= upper
    else:
        upper = water_elev_m + elev_tolerance
        mask = elev <= upper

    n = int(np.sum(mask))
    print(f"DSM: {n}/{mask.size} cells at or below {upper:.2f} m classified as water")
    return mask


def _noisiness_mask(ds_mean, cv_threshold=100.0):
    """Boolean (ny, nx) mask — True where motion is coherent (likely water).

    Stationary land pixels produce PIV noise in all frames: mean speed ≈ noise
    floor and temporal std ≈ same noise → coefficient of variation (CV) → very
    large.  Coherent water flow has a meaningful mean speed and bounded CV.

    cv_threshold: maximum allowed CV (%) to be classified as water.  Default
    100 % works well for fast-flowing sites; raise for slow, turbulent reaches.

    Falls back to an all-True mask when speed_std is not present in ds_mean.
    """
    if "speed_std" not in ds_mean:
        print("Noisiness mask: speed_std not found; skipping (all cells kept)")
        return np.ones(ds_mean["v_x"].shape, dtype=bool)

    speed     = np.sqrt(ds_mean["v_x"].values**2 + ds_mean["v_y"].values**2)
    speed_std = ds_mean["speed_std"].values
    eps       = 1e-6
    cv        = speed_std / np.maximum(speed, eps) * 100.0
    mask      = cv < cv_threshold

    n = int(np.sum(mask))
    print(f"Noisiness mask (CV < {cv_threshold:.0f}%): "
          f"{n}/{mask.size} cells kept as coherent motion")
    return mask


def _save_netcdf(ds_mean, output_dir):
    path = os.path.join(output_dir, "velocity.nc")
    ds_mean.to_netcdf(path)
    print(f"Velocity NetCDF saved to {path}")


def _save_geotiff(ds_mean, output_dir):
    """Interpolate the rotated pyORC grid onto a regular UTM grid and write GeoTIFF."""
    xs, ys = _utm_coords(ds_mean)
    v_x, v_y, speed, bearing, corr, s2n = _velocity_arrays(ds_mean)
    crs = _crs_from_ds(ds_mean)

    # Resolution: use local grid spacing (same in x and y)
    res = float(ds_mean.x.values[1] - ds_mean.x.values[0])

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    xi = np.arange(x_min, x_max + res, res)
    yi = np.arange(y_min, y_max + res, res)
    xi_grid, yi_grid = np.meshgrid(xi, yi)   # (ny_out, nx_out)

    src_pts = np.column_stack([xs.flatten(), ys.flatten()])

    has_std = "v_x_std" in ds_mean
    bands_data = [
        ("speed_m_s",       speed),
        ("bearing_deg_cwN", bearing),
        ("v_x_m_s",         v_x),
        ("v_y_m_s",         v_y),
        ("corr",            corr),
        ("s2n",             s2n),
    ]
    if has_std:
        bands_data += [
            ("speed_std_m_s", ds_mean["speed_std"].values),
            ("v_x_std_m_s",   ds_mean["v_x_std"].values),
            ("v_y_std_m_s",   ds_mean["v_y_std"].values),
        ]

    bands = [
        (name, griddata(src_pts, arr.flatten(), (xi_grid, yi_grid), method="linear").astype("float32"))
        for name, arr in bands_data
    ]

    transform = rasterio.transform.from_origin(
        west=x_min - res / 2,
        north=y_max + res / 2,
        xsize=res,
        ysize=res,
    )
    ny_out, nx_out = xi_grid.shape

    path = os.path.join(output_dir, "velocity.tif")
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=ny_out,
        width=nx_out,
        count=len(bands),
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        for i, (name, arr) in enumerate(bands, start=1):
            dst.write(np.flipud(arr), i)   # flip: row 0 = northernmost
            dst.update_tags(i, name=name)

    print(f"Velocity GeoTIFF saved to {path}  ({len(bands)} bands: {[b[0] for b in bands]})")


def _save_gpkg(ds_mean, output_dir, min_s2n=6.0, min_corr=0.5, min_speed=0.02, land_mask=None):
    xs, ys = _utm_coords(ds_mean)
    v_x, v_y, speed, bearing, corr, s2n = _velocity_arrays(ds_mean)
    crs = _crs_from_ds(ds_mean)

    mask = (s2n >= min_s2n) & (corr >= min_corr) & (speed >= min_speed)
    if land_mask is not None:
        mask = mask & land_mask

    n_total, n_kept = mask.size, int(mask.sum())

    attrs = {
        "v_x_m_s":         v_x[mask].astype(float),
        "v_y_m_s":         v_y[mask].astype(float),
        "speed_m_s":       speed[mask].astype(float),
        "bearing_deg_cwN": bearing[mask].astype(float),
        "corr":            corr[mask].astype(float),
        "s2n":             s2n[mask].astype(float),
    }
    for std_var, col in [("v_x_std", "v_x_std_m_s"),
                         ("v_y_std", "v_y_std_m_s"),
                         ("speed_std", "speed_std_m_s")]:
        if std_var in ds_mean:
            attrs[col] = ds_mean[std_var].values[mask].astype(float)

    gdf = gpd.GeoDataFrame(
        attrs,
        geometry=[Point(xi, yi) for xi, yi in zip(xs[mask], ys[mask])],
        crs=crs,
    )

    path = os.path.join(output_dir, "velocity.gpkg")
    gdf.to_file(path, driver="GPKG")
    print(f"Velocity GeoPackage saved to {path}  ({n_kept}/{n_total} points after filters)")


def _make_frame_utm(frame_da, ds_mean, output_dir):
    """Warp the projected frame to a north-up UTM GeoTIFF using GCPs + gdalwarp.

    Fits an affine local→UTM transform from all PIV cell positions, extrapolates
    it to the full projected-frame corners, embeds those as GCPs, and calls
    gdalwarp to produce a regular north-up raster.

    Returns the path to frame_utm.tif in output_dir.
    """
    from rasterio.control import GroundControlPoint

    xs, ys = _utm_coords(ds_mean)
    crs = _crs_from_ds(ds_mean)

    # Normalise to (n_bands, ny, nx) uint8
    arr = frame_da.values
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):   # channels-last → channels-first
        arr = np.moveaxis(arr, -1, 0)
    if arr.dtype != np.uint8:
        lo, hi = float(arr.min()), float(arr.max())
        scale = 255.0 / (hi - lo) if hi > lo else 1.0
        arr = ((arr - lo) * scale).clip(0, 255).astype(np.uint8)
    n_bands, ny_img, nx_img = arr.shape

    # Fit affine local→UTM from all PIV cells (least-squares over full grid)
    x_piv = ds_mean.x.values          # shape (nx_piv,)
    y_piv = ds_mean.y.values          # shape (ny_piv,)
    xx, yy = np.meshgrid(x_piv, y_piv)
    A = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    cx, _, _, _ = np.linalg.lstsq(A, xs.ravel(), rcond=None)
    cy, _, _, _ = np.linalg.lstsq(A, ys.ravel(), rcond=None)

    def _local_to_utm(x_l, y_l):
        P = np.column_stack([np.ravel(x_l), np.ravel(y_l), np.ones(np.size(x_l))])
        return (P @ cx).ravel(), (P @ cy).ravel()

    # 4 corner GCPs: map frame pixel corners to UTM via the fitted transform
    x_fr = frame_da.x.values          # 1-D local x, length nx_img
    y_fr = frame_da.y.values          # 1-D local y, length ny_img
    c_xl = [x_fr[0],  x_fr[-1], x_fr[0],  x_fr[-1]]
    c_yl = [y_fr[0],  y_fr[0],  y_fr[-1], y_fr[-1]]
    ux, uy = _local_to_utm(c_xl, c_yl)
    rows   = [0,       0,        ny_img-1, ny_img-1]
    cols   = [0,       nx_img-1, 0,        nx_img-1]
    gcps   = [GroundControlPoint(row=r, col=c, x=float(ex), y=float(ey))
              for r, c, ex, ey in zip(rows, cols, ux, uy)]

    gcp_path    = os.path.join(output_dir, "_frame_gcp.tif")
    warped_path = os.path.join(output_dir, "frame_utm.tif")
    with rasterio.open(gcp_path, "w", driver="GTiff",
                       height=ny_img, width=nx_img,
                       count=n_bands, dtype="uint8") as dst:
        dst.write(arr)
        dst.gcps = (gcps, crs)    # rasterio 1.3+ property setter (replaces update_gcps)

    subprocess.run(
        ["gdalwarp", "-r", "bilinear", "-overwrite", "-dstalpha", gcp_path, warped_path],
        check=True, capture_output=True,
    )
    os.remove(gcp_path)

    # Zero out alpha at the stabilization border (stabilo fills uncovered corners
    # with 0) and also erode the valid region by a few pixels to remove the
    # bilinear-interpolation blend zone between the fill zeros and real pixels.
    from scipy.ndimage import binary_erosion
    with rasterio.open(warped_path, "r+") as src:
        data  = src.read()              # (bands, ny, nx); last band = alpha from -dstalpha
        rgb   = data[:-1]
        alpha = data[-1]
        stab_border = rgb.max(axis=0) < 10     # pure-black fill pixels
        alpha[stab_border] = 0
        valid = alpha > 0
        valid = binary_erosion(valid, iterations=3)  # trim blended fringe
        alpha[~valid] = 0
        data[-1] = alpha
        src.write(data)

    print(f"Warped background frame saved to {warped_path}")
    return warped_path


def _nice_upper(v):
    """Round v up to the nearest 0.5-unit step (0.5, 1.0, 1.5, 2.0, …)."""
    return float(np.ceil(v * 2.0)) / 2.0


def _save_plots_utm(ds_mean, frame_utm_path, output_dir, land_mask=None, filter_mask=None):
    """Generate UTM geographic figures.

    land_mask:   boolean (ny, nx) — cells classified as water by the noisiness
                 criterion; applied to the full-field figures so end users can
                 screen quality themselves using the companion CV/std figures.
    filter_mask: boolean (ny, nx) — stricter quality filter (s2n + corr + speed
                 + land mask); applied to the filtered raster+arrows figure.

    Colorbar upper bound and arrow scale are derived from the land-masked
    distribution and shared across all figures for consistent comparison.

    Outputs:
      velocity_utm.png                       — background + colored quiver (land-masked)
      velocity_raster_utm.png                — background + speed raster (land-masked)
      velocity_raster_arrows_utm.png         — background + raster + white arrows (land-masked)
      velocity_raster_arrows_filtered_utm.png— background + raster + white arrows (quality-filtered)
      velocity_std_utm.png                   — temporal std dev of speed
      velocity_cv_utm.png                    — coefficient of variation of speed (%)
    """
    import matplotlib.colors as mcolors
    from rasterio.plot import reshape_as_image

    xs, ys = _utm_coords(ds_mean)
    v_x, v_y, speed, _bearing, corr, s2n = _velocity_arrays(ds_mean)

    # Land mask only — quality screening is left to the end user
    mask = land_mask if land_mask is not None else np.ones(speed.shape, dtype=bool)

    speed_vals = speed[mask]

    # Shared scale: colorbar 0 → nice ceiling of 99th-percentile speed
    vmax = _nice_upper(float(np.nanpercentile(speed_vals, 99))) if speed_vals.size else 5.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    # Arrow scale: 95th-percentile arrow fits in 80% of one PIV cell
    cell_spacing = float(abs(ds_mean.x.values[1] - ds_mean.x.values[0]))
    speed_p95 = float(np.nanpercentile(speed_vals, 95)) if speed_vals.size else 1.0
    arrow_scale = speed_p95 / (0.8 * cell_spacing)

    print(f"UTM plots: vmax={vmax:.1f} m/s  arrow_scale={arrow_scale:.2f}  "
          f"cell_spacing={cell_spacing:.2f} m")

    with rasterio.open(frame_utm_path) as src:
        img = reshape_as_image(src.read())    # (ny, nx, bands)
        ext = [src.bounds.left, src.bounds.right,
               src.bounds.bottom, src.bounds.top]
        n_bands = src.count

    speed_raster = np.ma.array(speed, mask=~mask)

    def _bg(ax):
        # Use RGBA (4 bands) when available so nodata corners are transparent
        if n_bands >= 3:
            ax.imshow(img[..., :n_bands], extent=ext, origin="upper", aspect="equal")
        else:
            ax.imshow(img[..., 0], extent=ext, origin="upper", aspect="equal", cmap="gray")

    def _raster(ax, alpha=1.0):
        return ax.pcolormesh(xs, ys, speed_raster, cmap="plasma", norm=norm,
                             shading="nearest", alpha=alpha, zorder=2)

    def _arrows_colored(ax):
        return ax.quiver(xs[mask], ys[mask], v_x[mask], v_y[mask], speed[mask],
                         cmap="plasma", norm=norm,
                         scale=arrow_scale, scale_units="xy", width=0.0012, zorder=3)

    def _arrows(ax, color):
        ax.quiver(xs[mask], ys[mask], v_x[mask], v_y[mask],
                  color=color,
                  scale=arrow_scale, scale_units="xy", width=0.0012, zorder=3)

    def _finish(fig, ax, mappable, fname, cbar_label="Speed (m/s)"):
        plt.colorbar(mappable, ax=ax, label=cbar_label, shrink=0.7, extend="max")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.set_aspect("equal")
        ax.set_facecolor((0, 0, 0, 0))
        path = os.path.join(output_dir, fname)
        plt.savefig(path, dpi=300, bbox_inches="tight", transparent=True)
        plt.close()
        print(f"Saved {path}")

    # Uncertainty fields (require temporal std from ensemble_corr=False run)
    has_std = "speed_std" in ds_mean
    if has_std:
        speed_std = ds_mean["speed_std"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(speed > 0, speed_std / speed * 100.0, np.nan)
        std_raster = np.ma.array(speed_std, mask=~mask)
        cv_raster  = np.ma.array(cv,        mask=~mask)
        std_vmax = _nice_upper(float(np.nanpercentile(speed_std[mask], 99)))
        cv_vmax  = min(_nice_upper(float(np.nanpercentile(cv[mask],    99))), 100.0)
        std_norm = mcolors.Normalize(vmin=0.0, vmax=std_vmax)
        cv_norm  = mcolors.Normalize(vmin=0.0, vmax=cv_vmax)

    # Figure 1: colored quiver on background frame
    fig, ax = plt.subplots(figsize=(10, 12))
    _bg(ax)
    _finish(fig, ax, _arrows_colored(ax), "velocity_utm.png")

    # Figure 2: speed raster on background frame
    fig, ax = plt.subplots(figsize=(10, 12))
    _bg(ax)
    _finish(fig, ax, _raster(ax), "velocity_raster_utm.png")

    # Figure 3: speed raster + white arrows on background frame (land-masked)
    fig, ax = plt.subplots(figsize=(10, 12))
    _bg(ax)
    pcm = _raster(ax, alpha=0.7)
    _arrows(ax, color="white")
    _finish(fig, ax, pcm, "velocity_raster_arrows_utm.png")

    # Figure 4: same layout but quality-filtered (s2n + corr + speed + land mask)
    fmask = filter_mask if filter_mask is not None else mask
    speed_raster_f = np.ma.array(speed, mask=~fmask)
    fig, ax = plt.subplots(figsize=(10, 12))
    _bg(ax)
    pcm = ax.pcolormesh(xs, ys, speed_raster_f, cmap="plasma", norm=norm,
                        shading="nearest", alpha=0.7, zorder=2)
    ax.quiver(xs[fmask], ys[fmask], v_x[fmask], v_y[fmask],
              color="white",
              scale=arrow_scale, scale_units="xy", width=0.0012, zorder=3)
    _finish(fig, ax, pcm, "velocity_raster_arrows_filtered_utm.png")

    if has_std:
        # Figure 4: speed_std raster
        fig, ax = plt.subplots(figsize=(10, 12))
        _bg(ax)
        pcm = ax.pcolormesh(xs, ys, std_raster, cmap="plasma", norm=std_norm,
                            shading="nearest", zorder=2)
        _finish(fig, ax, pcm, "velocity_std_utm.png",
                cbar_label="Speed std dev (m/s)")

        # Figure 5: CV of speed (%)
        fig, ax = plt.subplots(figsize=(10, 12))
        _bg(ax)
        pcm = ax.pcolormesh(xs, ys, cv_raster, cmap="YlOrRd", norm=cv_norm,
                            shading="nearest", zorder=2)
        _finish(fig, ax, pcm, "velocity_cv_utm.png",
                cbar_label="Speed CV (%)")


def run_piv(video_path, output_dir, camera_config_path=None,
            start_frame=1, end_frame=None, h_a=0.0, piv_engine="numba",
            window_size=None,
            min_s2n=6.0, min_corr=0.5, min_speed=0.02,
            cv_threshold=100.0,
            dsm_path=None, water_elev_m=None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if end_frame is None:
        end_frame = nframes - 1

    if camera_config_path is not None:
        with open(camera_config_path) as f:
            camera_config = pyorc.CameraConfig(**json.load(f))
        camera_config.set_bbox_from_corners([[0, height], [width, height], [width, 0], [0, 0]])
    else:
        print("WARNING: no camera config provided; using placeholder pixel-scaled GCPs.")
        camera_config = _placeholder_camera_config(width, height)

    video = pyorc.Video(
        video_path,
        camera_config=camera_config,
        start_frame=start_frame,
        end_frame=end_frame,
        h_a=h_a,
    )

    da = video.get_frames()
    da_norm = da.frames.normalize()
    da_norm_proj = da_norm.frames.project(method="numpy")

    piv_kwargs = {"engine": piv_engine, "ensemble_corr": False}
    if window_size is not None:
        piv_kwargs["window_size"] = window_size  # int; pyORC expands to (n, n) internally
    piv = da_norm_proj.frames.get_piv(**piv_kwargs)

    da_rgb = video.get_frames(method="rgb")
    da_rgb_proj = da_rgb.frames.project()
    ds_mean = piv.mean(dim="time", keep_attrs=True)

    # Temporal std as per-cell uncertainty estimate
    import xarray as xr
    speed_all = np.sqrt(piv["v_x"]**2 + piv["v_y"]**2)
    ds_mean["v_x_std"]    = piv["v_x"].std(dim="time")
    ds_mean["v_y_std"]    = piv["v_y"].std(dim="time")
    ds_mean["speed_std"]  = speed_all.std(dim="time")

    plt.figure()
    p = da_rgb_proj[0].frames.plot()
    plt.savefig(os.path.join(output_dir, "Frame.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    p = da_rgb_proj[0].frames.plot()
    ds_mean.velocimetry.plot(ax=p.axes)
    plt.savefig(os.path.join(output_dir, "PIVquiverFrame.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Land mask: noisiness criterion (primary), optionally combined with DSM
    land_mask_np = _noisiness_mask(ds_mean, cv_threshold=cv_threshold)
    if dsm_path is not None:
        dsm_mask_np = _dsm_water_mask(ds_mean, dsm_path, water_elev_m)
        land_mask_np = land_mask_np & dsm_mask_np

    # Quality filter mask for the filtered quiver figure and GeoPackage
    speed_da = np.sqrt(ds_mean["v_x"]**2 + ds_mean["v_y"]**2)
    quality_mask = ((ds_mean["s2n"] >= min_s2n)
                    & (ds_mean["corr"] >= min_corr)
                    & (speed_da >= min_speed)
                    & xr.DataArray(land_mask_np, dims=["y", "x"]))

    ds_filtered = ds_mean.where(quality_mask)
    plt.figure()
    p = da_rgb_proj[0].frames.plot()
    ds_filtered.velocimetry.plot(ax=p.axes)
    plt.savefig(os.path.join(output_dir, "PIVquiverFiltered.png"), dpi=150, bbox_inches="tight")
    plt.close()

    _save_netcdf(ds_mean, output_dir)
    _save_geotiff(ds_mean, output_dir)
    _save_gpkg(ds_mean, output_dir, min_s2n=min_s2n, min_corr=min_corr,
               min_speed=min_speed, land_mask=land_mask_np)

    frame_utm_path = _make_frame_utm(da_rgb_proj[0], ds_mean, output_dir)
    _save_plots_utm(ds_mean, frame_utm_path, output_dir,
                    land_mask=land_mask_np,
                    filter_mask=quality_mask.values)


def main():
    parser = argparse.ArgumentParser(description="Run PIV on a stabilized drone video.")
    parser.add_argument("--video",          required=True,  help="Stabilized input video path")
    parser.add_argument("--camera-config",  default=None,   help="Camera config JSON (from georeferencing step)")
    parser.add_argument("--output-dir",     required=True,  help="Directory to write output files")
    parser.add_argument("--start-frame",    type=int, default=1)
    parser.add_argument("--end-frame",      type=int, default=None)
    parser.add_argument("--h-a",            type=float, default=0.0,  help="Actual water level (m)")
    parser.add_argument("--piv-engine",     default="numba", choices=["numba", "opencv"])
    parser.add_argument("--window-size",    type=int, default=None, help="PIV interrogation window size in pixels (default: pyORC default of 10)")
    parser.add_argument("--min-s2n",        type=float, default=6.0,  help="Min signal-to-noise for point filter (default: 6.0)")
    parser.add_argument("--min-corr",       type=float, default=0.5,  help="Min correlation for point filter (default: 0.5)")
    parser.add_argument("--min-speed",      type=float, default=0.02, help="Min speed (m/s) to include a vector (default: 0.02)")
    parser.add_argument("--cv-threshold",   type=float, default=100.0,
                        help="Max coefficient of variation (%%) to classify a cell as water; "
                             "high-CV cells are treated as stationary land (default: 100)")
    parser.add_argument("--dsm",            default=None,   help="DSM GeoTIFF for land/water masking (optional, secondary to CV criterion)")
    parser.add_argument("--water-elev-m",   type=float, default=None, help="Water surface elevation (m); auto-detected from DSM if omitted")
    args = parser.parse_args()

    run_piv(
        video_path=args.video,
        output_dir=args.output_dir,
        camera_config_path=args.camera_config,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        h_a=args.h_a,
        piv_engine=args.piv_engine,
        window_size=args.window_size,
        min_s2n=args.min_s2n,
        min_corr=args.min_corr,
        min_speed=args.min_speed,
        cv_threshold=args.cv_threshold,
        dsm_path=args.dsm,
        water_elev_m=args.water_elev_m,
    )


if __name__ == "__main__":
    main()
