import argparse
import json
import os
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

    bands_data = [
        ("speed_m_s",       speed),
        ("bearing_deg_cwN", bearing),
        ("v_x_m_s",         v_x),
        ("v_y_m_s",         v_y),
        ("corr",            corr),
        ("s2n",             s2n),
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


def _save_gpkg(ds_mean, output_dir, min_s2n=6.0, min_corr=0.5, dsm_mask=None):
    xs, ys = _utm_coords(ds_mean)
    v_x, v_y, speed, bearing, corr, s2n = _velocity_arrays(ds_mean)
    crs = _crs_from_ds(ds_mean)

    mask = (s2n >= min_s2n) & (corr >= min_corr)
    if dsm_mask is not None:
        mask = mask & dsm_mask

    n_total, n_kept = mask.size, int(mask.sum())

    gdf = gpd.GeoDataFrame(
        {
            "v_x_m_s":         v_x[mask].astype(float),
            "v_y_m_s":         v_y[mask].astype(float),
            "speed_m_s":       speed[mask].astype(float),
            "bearing_deg_cwN": bearing[mask].astype(float),
            "corr":            corr[mask].astype(float),
            "s2n":             s2n[mask].astype(float),
        },
        geometry=[Point(xi, yi) for xi, yi in zip(xs[mask], ys[mask])],
        crs=crs,
    )

    path = os.path.join(output_dir, "velocity.gpkg")
    gdf.to_file(path, driver="GPKG")
    print(f"Velocity GeoPackage saved to {path}  ({n_kept}/{n_total} points after filters)")


def run_piv(video_path, output_dir, camera_config_path=None,
            start_frame=1, end_frame=None, h_a=0.0, piv_engine="numba",
            min_s2n=6.0, min_corr=0.5,
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

    piv = da_norm_proj.frames.get_piv(engine=piv_engine)

    da_rgb = video.get_frames(method="rgb")
    da_rgb_proj = da_rgb.frames.project()
    ds_mean = piv.mean(dim="time", keep_attrs=True)

    plt.figure()
    p = da_rgb_proj[0].frames.plot()
    plt.savefig(os.path.join(output_dir, "Frame.png"))

    plt.figure()
    p = da_rgb_proj[0].frames.plot()
    ds_mean.velocimetry.plot(ax=p.axes)
    plt.savefig(os.path.join(output_dir, "PIVquiverFrame.png"))

    # Build combined filter mask: quality thresholds + optional DSM land mask
    import xarray as xr
    quality_mask = (ds_mean["s2n"] >= min_s2n) & (ds_mean["corr"] >= min_corr)
    dsm_mask = None
    if dsm_path is not None:
        dsm_mask_np = _dsm_water_mask(ds_mean, dsm_path, water_elev_m)
        dsm_mask = dsm_mask_np
        dsm_xr = xr.DataArray(dsm_mask_np, dims=["y", "x"])
        quality_mask = quality_mask & dsm_xr

    ds_filtered = ds_mean.where(quality_mask)
    plt.figure()
    p = da_rgb_proj[0].frames.plot()
    ds_filtered.velocimetry.plot(ax=p.axes)
    plt.savefig(os.path.join(output_dir, "PIVquiverFiltered.png"))

    plt.show()

    _save_netcdf(ds_mean, output_dir)
    _save_geotiff(ds_mean, output_dir)
    _save_gpkg(ds_mean, output_dir, min_s2n=min_s2n, min_corr=min_corr,
               dsm_mask=dsm_mask)


def main():
    parser = argparse.ArgumentParser(description="Run PIV on a stabilized drone video.")
    parser.add_argument("--video",          required=True,  help="Stabilized input video path")
    parser.add_argument("--camera-config",  default=None,   help="Camera config JSON (from georeferencing step)")
    parser.add_argument("--output-dir",     required=True,  help="Directory to write output files")
    parser.add_argument("--start-frame",    type=int, default=1)
    parser.add_argument("--end-frame",      type=int, default=None)
    parser.add_argument("--h-a",            type=float, default=0.0,  help="Actual water level (m)")
    parser.add_argument("--piv-engine",     default="numba", choices=["numba", "opencv"])
    parser.add_argument("--min-s2n",        type=float, default=6.0,  help="Min signal-to-noise for point filter (default: 6.0)")
    parser.add_argument("--min-corr",       type=float, default=0.5,  help="Min correlation for point filter (default: 0.5)")
    parser.add_argument("--dsm",            default=None,   help="DSM GeoTIFF for land/water masking")
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
        min_s2n=args.min_s2n,
        min_corr=args.min_corr,
        dsm_path=args.dsm,
        water_elev_m=args.water_elev_m,
    )


if __name__ == "__main__":
    main()
