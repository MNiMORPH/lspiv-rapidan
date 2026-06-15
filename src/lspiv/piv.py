import argparse
import dataclasses
import json
import os
import subprocess
import sys

import cv2
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; figures are saved to disk only
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.transform
from scipy.interpolate import griddata
from shapely.geometry import Point


@dataclasses.dataclass
class CameraConfig:
    """Camera configuration (replaces pyorc.CameraConfig).

    gcps must contain 'src' (image pixel coords) and 'dst' (UTM coords),
    each a list of 4 [x, y] pairs corresponding to the same physical points.
    """
    height: int
    width: int
    crs: int
    gcps: dict
    resolution: float = 0.05


def _compute_projection_params(camera_config):
    """Compute perspective projection parameters from GCPs.

    Returns (M, x_min, y_max, out_w, out_h, res) where M is the homography
    that maps image pixel (col, row) to output pixel (col, row). The output
    grid is a regular north-up UTM raster at `resolution` m/px.
    """
    src_pts = np.float32(camera_config.gcps["src"])   # image (col, row)
    dst_pts = np.float32(camera_config.gcps["dst"])   # UTM (easting, northing)
    res = camera_config.resolution

    x_min = float(dst_pts[:, 0].min())
    x_max = float(dst_pts[:, 0].max())
    y_min = float(dst_pts[:, 1].min())
    y_max = float(dst_pts[:, 1].max())

    out_w = max(1, int(round((x_max - x_min) / res)))
    out_h = max(1, int(round((y_max - y_min) / res)))

    # Output pixel (c, r) ↔ UTM (x_min + c*res, y_max - r*res)
    out_pts = np.float32([
        [(utm_x - x_min) / res, (y_max - utm_y) / res]
        for utm_x, utm_y in dst_pts
    ])

    M = cv2.getPerspectiveTransform(src_pts, out_pts)
    return M, x_min, y_max, out_w, out_h, res


def _load_frames_gray(video_path, start_frame, end_frame):
    """Load grayscale frames [start_frame, end_frame] as float32 (N, H, W)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path} [{start_frame}–{end_frame}]")
    return np.stack(frames)


def _load_rgb_frame(video_path, frame_idx):
    """Load a single BGR frame (uint8) at frame_idx."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def _normalize_frames(frames_np, samples=15):
    """Background subtraction + per-frame normalization, matching pyORC frames.normalize().

    Subtracts the temporal mean of evenly-spaced sample frames, then scales
    each frame independently to [0, 255] float32.
    """
    n = len(frames_np)
    time_interval = max(1, round(n / samples))
    background = frames_np[::time_interval].mean(axis=0).astype(np.float32)
    reduced = frames_np.astype(np.float32) - background
    f_min = reduced.min(axis=(1, 2), keepdims=True)
    f_max = reduced.max(axis=(1, 2), keepdims=True)
    denom = np.where(f_max > f_min, f_max - f_min, 1.0)
    return ((reduced - f_min) / denom * 255.0).astype(np.float32)


def _placeholder_camera_config(width, height, crs=32615):
    """Build a pixel-scaled CameraConfig for use without real GCPs.

    Maps image corners to a local UTM grid at camera_config.resolution m/px.
    Coordinates are meaningless but the pipeline produces valid output.
    """
    res = 0.05
    gcps = {
        "src": [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
        "dst": [
            [0.0,       height * res],
            [width * res, height * res],
            [width * res, 0.0],
            [0.0,       0.0],
        ],
        "h_ref": 0.0,
        "z_0":   0.0,
    }
    return CameraConfig(height=height, width=width, crs=crs, gcps=gcps, resolution=res)


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
    """Interpolate the PIV grid onto a regular UTM grid and write GeoTIFF."""
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
        with np.errstate(divide="ignore", invalid="ignore"):
            cv_pct = np.where(speed > 1e-6,
                              ds_mean["speed_std"].values / speed * 100.0,
                              np.nan).astype("float32")
        bands_data += [
            ("speed_std_m_s", ds_mean["speed_std"].values),
            ("v_x_std_m_s",   ds_mean["v_x_std"].values),
            ("v_y_std_m_s",   ds_mean["v_y_std"].values),
            ("speed_cv_pct",  cv_pct),
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

    corr_ok = np.isnan(corr) | (corr >= min_corr)
    mask = (s2n >= min_s2n) & corr_ok & (speed >= min_speed)
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
    if "speed_std" in ds_mean:
        with np.errstate(divide="ignore", invalid="ignore"):
            attrs["speed_cv_pct"] = np.where(
                speed[mask] > 1e-6,
                ds_mean["speed_std"].values[mask] / speed[mask] * 100.0,
                np.nan,
            ).astype(float)

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


def _save_plots_utm(ds_mean, frame_utm_path, output_dir, land_mask=None):
    """Generate UTM geographic figures in two variants each: noise-masked and full domain.

    land_mask: boolean (ny, nx) — True for cells classified as water by the
               noisiness criterion.  Scale (colorbar range, arrow size) is
               derived from the masked distribution and shared between the two
               variants for direct comparison.

    Each figure type produces a *_utm.png (masked) and *_utm_all.png (all cells).
    The _all suffix sorts immediately after the base name so pairs land adjacent:
      velocity_utm.png / velocity_utm_all.png
      velocity_raster_utm.png / velocity_raster_utm_all.png
      velocity_raster_arrows_utm.png / velocity_raster_arrows_utm_all.png
      velocity_std_utm.png / velocity_std_utm_all.png
      velocity_cv_utm.png / velocity_cv_utm_all.png
    """
    import matplotlib.colors as mcolors
    from matplotlib_scalebar.scalebar import ScaleBar
    from rasterio.plot import reshape_as_image

    clip_name = os.path.basename(os.path.normpath(output_dir))

    xs, ys = _utm_coords(ds_mean)
    v_x, v_y, speed, _bearing, corr, s2n = _velocity_arrays(ds_mean)

    mask     = land_mask if land_mask is not None else np.ones(speed.shape, dtype=bool)
    all_mask = np.ones(speed.shape, dtype=bool)

    speed_vals = speed[mask]

    # Scale derived from masked (water) cells — shared by both variants
    vmax = _nice_upper(float(np.nanpercentile(speed_vals, 99))) if speed_vals.size else 5.0
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)

    cell_spacing = float(abs(ds_mean.x.values[1] - ds_mean.x.values[0]))
    speed_p95    = float(np.nanpercentile(speed_vals, 95)) if speed_vals.size else 1.0
    arrow_scale  = speed_p95 / (0.8 * cell_spacing)

    print(f"UTM plots: vmax={vmax:.1f} m/s  arrow_scale={arrow_scale:.2f}  "
          f"cell_spacing={cell_spacing:.2f} m")

    with rasterio.open(frame_utm_path) as src:
        img     = reshape_as_image(src.read())
        ext     = [src.bounds.left, src.bounds.right,
                   src.bounds.bottom, src.bounds.top]
        n_bands = src.count

    def _bg(ax):
        if n_bands >= 3:
            ax.imshow(img[..., :n_bands], extent=ext, origin="upper", aspect="equal")
        else:
            ax.imshow(img[..., 0], extent=ext, origin="upper", aspect="equal", cmap="gray")

    def _finish(fig, ax, mappable, fname, cbar_label="Speed (m/s)"):
        # Fix the axes and colorbar at identical figure-coordinate positions so
        # every output figure has the same map pixels regardless of label length.
        # subplots_adjust pins the axes box; manual cax pins the colorbar.
        fig.subplots_adjust(left=0.12, right=0.80, top=0.97, bottom=0.08)
        cax = fig.add_axes([0.83, 0.15, 0.025, 0.70])
        fig.colorbar(mappable, cax=cax, label=cbar_label, extend="max")
        ax.set_title(clip_name, fontsize=11)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_facecolor((0, 0, 0, 0))
        ax.add_artist(ScaleBar(1, units="m", location="lower right",
                               color="white", box_color="black", box_alpha=0.5))
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=300, transparent=True)
        plt.close()
        print(f"Saved {path}")

    # Uncertainty fields (require temporal std from ensemble_corr=False run)
    has_std = "speed_std" in ds_mean
    if has_std:
        speed_std = ds_mean["speed_std"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = np.where(speed > 0, speed_std / speed * 100.0, np.nan)
        std_vmax = _nice_upper(float(np.nanpercentile(speed_std[mask], 99)))
        cv_vmax  = min(_nice_upper(float(np.nanpercentile(cv[mask],    99))), 100.0)
        std_norm = mcolors.Normalize(vmin=0.0, vmax=std_vmax)
        cv_norm  = mcolors.Normalize(vmin=0.0, vmax=cv_vmax)

    # Generate each figure type for both mask variants
    for m, suffix in [(mask, "_utm.png"), (all_mask, "_utm_all.png")]:
        sr = np.ma.array(speed, mask=~m)

        # Colored quiver
        fig, ax = plt.subplots(figsize=(12, 10))
        _bg(ax)
        q = ax.quiver(xs[m], ys[m], v_x[m], v_y[m], speed[m],
                      cmap="plasma", norm=norm,
                      scale=arrow_scale, scale_units="xy", width=0.0012, zorder=3)
        _finish(fig, ax, q, f"velocity{suffix}")

        # Speed raster
        fig, ax = plt.subplots(figsize=(12, 10))
        _bg(ax)
        pcm = ax.pcolormesh(xs, ys, sr, cmap="plasma", norm=norm,
                            shading="nearest", zorder=2)
        _finish(fig, ax, pcm, f"velocity_raster{suffix}")

        # Speed raster + white arrows
        fig, ax = plt.subplots(figsize=(12, 10))
        _bg(ax)
        pcm = ax.pcolormesh(xs, ys, sr, cmap="plasma", norm=norm,
                            shading="nearest", alpha=0.7, zorder=2)
        ax.quiver(xs[m], ys[m], v_x[m], v_y[m],
                  color="white",
                  scale=arrow_scale, scale_units="xy", width=0.0012, zorder=3)
        _finish(fig, ax, pcm, f"velocity_raster_arrows{suffix}")

        if has_std:
            # Std dev raster
            fig, ax = plt.subplots(figsize=(12, 10))
            _bg(ax)
            pcm = ax.pcolormesh(xs, ys, np.ma.array(speed_std, mask=~m),
                                cmap="plasma", norm=std_norm,
                                shading="nearest", zorder=2)
            _finish(fig, ax, pcm, f"velocity_std{suffix}",
                    cbar_label="Speed std dev (m/s)")

            # CV raster
            fig, ax = plt.subplots(figsize=(12, 10))
            _bg(ax)
            pcm = ax.pcolormesh(xs, ys, np.ma.array(cv, mask=~m),
                                cmap="YlOrRd", norm=cv_norm,
                                shading="nearest", zorder=2)
            _finish(fig, ax, pcm, f"velocity_cv{suffix}",
                    cbar_label="Speed CV (%)")


def _piv_chunked(video_path, camera_config, start_frame, end_frame, h_a,
                 window_size, fps, chunk_size=50):
    """Run PIV in memory-safe chunks using OpenPIV and return a concatenated Dataset.

    Loads grayscale frames via cv2, removes background via temporal mean subtraction,
    projects to a regular UTM grid via cv2.warpPerspective, then runs FFT
    cross-correlation with OpenPIV on each consecutive frame pair.  50 frames per
    chunk keeps peak RAM ≈1.6 GB.

    corr is set to NaN — OpenPIV does not expose a normalised correlation
    coefficient directly. s2n is the peak2mean ratio from OpenPIV.
    """
    import xarray as xr
    import openpiv.pyprocess

    ws      = window_size if window_size is not None else 10
    overlap = ws // 2

    M, x_min, y_max, out_w, out_h, res = _compute_projection_params(camera_config)

    ds_attrs = {"camera_config": json.dumps({
        "crs":    camera_config.crs,
        "gcps":   camera_config.gcps,
        "height": camera_config.height,
        "width":  camera_config.width,
    })}

    coord_template = None
    piv_chunks = []

    for chunk_start in range(start_frame, end_frame + 1, chunk_size):
        chunk_end = min(chunk_start + chunk_size - 1, end_frame)

        frames_raw  = _load_frames_gray(video_path, chunk_start, chunk_end)
        frames_norm = _normalize_frames(frames_raw)
        frames_proj = np.stack([
            cv2.warpPerspective(f, M, (out_w, out_h))
            for f in frames_norm
        ])   # (N, out_h, out_w)
        del frames_raw, frames_norm

        # Build coordinate arrays once from the first chunk.
        if coord_template is None:
            proj_h, proj_w = frames_proj.shape[1], frames_proj.shape[2]
            x_px, y_px = openpiv.pyprocess.get_coordinates(
                image_size=(proj_h, proj_w),
                search_area_size=ws,
                overlap=overlap,
            )
            # x_px, y_px are output-pixel positions; convert to UTM
            xs_2d  = x_min + x_px * res
            ys_2d  = y_max - y_px * res
            x_1d   = xs_2d[0, :]    # easting along columns
            y_1d   = ys_2d[:, 0]    # northing along rows (decreasing southward)

            transformer = pyproj.Transformer.from_crs(
                camera_config.crs, 4326, always_xy=True)
            lon_2d, lat_2d = transformer.transform(xs_2d, ys_2d)

            coord_template = (xs_2d, ys_2d, lon_2d, lat_2d, x_1d, y_1d)

        xs_2d, ys_2d, lon_2d, lat_2d, x_1d, y_1d = coord_template

        chunk_vx, chunk_vy, chunk_s2n = [], [], []
        for i in range(len(frames_proj) - 1):
            u, v, s2n = openpiv.pyprocess.extended_search_area_piv(
                frames_proj[i], frames_proj[i + 1],
                window_size=ws, overlap=overlap, search_area_size=ws,
                sig2noise_method="peak2mean",
            )
            chunk_vx.append(u * res * fps)
            chunk_vy.append(-v * res * fps)
            chunk_s2n.append(s2n)

        n_pairs = len(chunk_vx)
        print(f"  PIV chunk frames {chunk_start}–{chunk_end}: {n_pairs} pairs")

        ny, nx = xs_2d.shape
        time_coords = np.array([(chunk_start + i) / fps for i in range(n_pairs)])

        ds = xr.Dataset(
            {
                "v_x":  (["time", "y", "x"], np.array(chunk_vx,  dtype=np.float32)),
                "v_y":  (["time", "y", "x"], np.array(chunk_vy,  dtype=np.float32)),
                "s2n":  (["time", "y", "x"], np.array(chunk_s2n, dtype=np.float32)),
                "corr": (["time", "y", "x"],
                         np.full((n_pairs, ny, nx), np.nan, dtype=np.float32)),
            },
            coords={
                "time": time_coords,
                "y":    y_1d,
                "x":    x_1d,
                "xs":   (["y", "x"], xs_2d),
                "ys":   (["y", "x"], ys_2d),
                "lon":  (["y", "x"], lon_2d),
                "lat":  (["y", "x"], lat_2d),
            },
            attrs=ds_attrs,
        )
        piv_chunks.append(ds)
        del frames_proj

    result = xr.concat(piv_chunks, dim="time")
    result.attrs = ds_attrs
    return result


def run_piv(video_path, output_dir, camera_config_path=None,
            start_frame=1, end_frame=None, h_a=0.0, piv_engine="numba",
            window_size=None,
            min_s2n=1.0, min_corr=0.5, min_speed=0.02,
            cv_threshold=100.0,
            dsm_path=None, water_elev_m=None):
    import xarray as xr

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps     = cap.get(cv2.CAP_PROP_FPS)
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if end_frame is None:
        end_frame = nframes - 1

    if camera_config_path is not None:
        with open(camera_config_path) as f:
            data = json.load(f)
        camera_config = CameraConfig(
            height=data["height"],
            width=data["width"],
            crs=data["crs"],
            gcps=data["gcps"],
            resolution=data.get("resolution", 0.05),
        )
    else:
        print("WARNING: no camera config provided; using placeholder pixel-scaled GCPs.")
        camera_config = _placeholder_camera_config(width, height)

    piv = _piv_chunked(video_path, camera_config, start_frame, end_frame, h_a,
                       window_size=window_size, fps=fps, chunk_size=50)

    ds_mean = piv.mean(dim="time", keep_attrs=True)

    # Temporal std as per-cell uncertainty estimate
    speed_all = np.sqrt(piv["v_x"]**2 + piv["v_y"]**2)
    ds_mean["v_x_std"]   = piv["v_x"].std(dim="time")
    ds_mean["v_y_std"]   = piv["v_y"].std(dim="time")
    ds_mean["speed_std"] = speed_all.std(dim="time")
    del piv, speed_all

    # Project two RGB frames: frame 0 for diagnostics, mid-frame for UTM background.
    M, x_min, y_max, out_w, out_h, res = _compute_projection_params(camera_config)
    extent_utm = [x_min, x_min + out_w * res, y_max - out_h * res, y_max]

    mid_frame_abs = start_frame + (end_frame - start_frame) // 2

    frame0_bgr = _load_rgb_frame(video_path, start_frame)
    frame0_proj_bgr = cv2.warpPerspective(frame0_bgr, M, (out_w, out_h))
    frame0_rgb = cv2.cvtColor(frame0_proj_bgr, cv2.COLOR_BGR2RGB)

    frame_mid_bgr = _load_rgb_frame(video_path, mid_frame_abs)
    frame_mid_proj_bgr = cv2.warpPerspective(frame_mid_bgr, M, (out_w, out_h))
    frame_mid_rgb = cv2.cvtColor(frame_mid_proj_bgr, cv2.COLOR_BGR2RGB)

    # Build xarray DataArray for mid-frame (used by _make_frame_utm)
    import xarray as xr
    x_1d_full = x_min + np.arange(out_w) * res
    y_1d_full = y_max - np.arange(out_h) * res   # decreasing (north→south)
    frame_mid_da = xr.DataArray(
        frame_mid_rgb,
        dims=["y", "x", "rgb"],
        coords={"x": x_1d_full, "y": y_1d_full},
    )

    fig, ax = plt.subplots()
    ax.imshow(frame0_rgb, extent=extent_utm, origin="upper", aspect="equal")
    plt.savefig(os.path.join(output_dir, "Frame.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots()
    ax.imshow(frame0_rgb, extent=extent_utm, origin="upper", aspect="equal")
    ax.quiver(ds_mean.xs.values, ds_mean.ys.values,
              ds_mean["v_x"].values, ds_mean["v_y"].values,
              color="r", scale=20, width=0.002)
    plt.savefig(os.path.join(output_dir, "PIVquiverFrame.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Land mask: noisiness criterion (primary), optionally combined with DSM
    land_mask_np = _noisiness_mask(ds_mean, cv_threshold=cv_threshold)
    if dsm_path is not None:
        dsm_mask_np = _dsm_water_mask(ds_mean, dsm_path, water_elev_m)
        land_mask_np = land_mask_np & dsm_mask_np

    # Quality filter mask for the filtered quiver figure and GeoPackage.
    # corr is NaN with OpenPIV; treat NaN corr as passing the threshold.
    speed_da = np.sqrt(ds_mean["v_x"]**2 + ds_mean["v_y"]**2)
    corr_ok = ds_mean["corr"].isnull() | (ds_mean["corr"] >= min_corr)
    quality_mask = ((ds_mean["s2n"] >= min_s2n)
                    & corr_ok
                    & (speed_da >= min_speed)
                    & xr.DataArray(land_mask_np, dims=["y", "x"]))

    ds_filtered = ds_mean.where(quality_mask)

    fig, ax = plt.subplots()
    ax.imshow(frame0_rgb, extent=extent_utm, origin="upper", aspect="equal")
    ax.quiver(ds_filtered.xs.values, ds_filtered.ys.values,
              ds_filtered["v_x"].values, ds_filtered["v_y"].values,
              color="r", scale=20, width=0.002)
    plt.savefig(os.path.join(output_dir, "PIVquiverFiltered.png"), dpi=150, bbox_inches="tight")
    plt.close()

    _save_netcdf(ds_mean, output_dir)
    _save_geotiff(ds_mean, output_dir)
    _save_gpkg(ds_mean, output_dir, min_s2n=min_s2n, min_corr=min_corr,
               min_speed=min_speed, land_mask=land_mask_np)

    frame_utm_path = _make_frame_utm(frame_mid_da, ds_mean, output_dir)
    _save_plots_utm(ds_mean, frame_utm_path, output_dir, land_mask=land_mask_np)


def main():
    parser = argparse.ArgumentParser(description="Run PIV on a stabilized drone video.")
    parser.add_argument("--video",          required=True,  help="Stabilized input video path")
    parser.add_argument("--camera-config",  default=None,   help="Camera config JSON (from georeferencing step)")
    parser.add_argument("--output-dir",     required=True,  help="Directory to write output files")
    parser.add_argument("--start-frame",    type=int, default=1)
    parser.add_argument("--end-frame",      type=int, default=None)
    parser.add_argument("--h-a",            type=float, default=0.0,  help="Actual water level (m)")
    parser.add_argument("--piv-engine",     default="numba", choices=["numba", "opencv"],
                        help="Ignored (retained for backward compatibility; OpenPIV is always used)")
    parser.add_argument("--window-size",    type=int, default=None, help="PIV interrogation window size in pixels (default: 10)")
    parser.add_argument("--min-s2n",        type=float, default=1.0,  help="Min signal-to-noise for point filter (default: 1.0; OpenPIV peak2mean s2n has low dynamic range)")
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
