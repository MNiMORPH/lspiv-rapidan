"""Integration smoke test: run the full PIV pipeline on 5 frames of MAX_0102.

Requires data files that are not tracked in git:
  data/stabilized/MAX_0102.mp4
  data/camera_configs/MAX_0102.json

Skipped automatically when those files are absent (e.g. in CI without data).
"""

import os
import pytest

VIDEO  = "data/stabilized/MAX_0102.mp4"
CONFIG = "data/camera_configs/MAX_0102.json"

pytestmark = pytest.mark.skipif(
    not (os.path.exists(VIDEO) and os.path.exists(CONFIG)),
    reason="Rapidan data files not present",
)


def test_piv_rapidan_smoke(tmp_path):
    """Pipeline produces all expected outputs for a 5-frame run of MAX_0102."""
    from lspiv.piv import run_piv

    run_piv(
        video_path=VIDEO,
        output_dir=str(tmp_path),
        camera_config_path=CONFIG,
        start_frame=1,
        end_frame=5,
        h_a=0.0,
        piv_engine="numba",
        window_size=20,
        min_s2n=1.0,
        min_corr=0.5,
        min_speed=0.02,
        cv_threshold=100.0,
    )

    expected = [
        "velocity.nc",
        "velocity.tif",
        "velocity.gpkg",
        "frame_utm.tif",
        "Frame.png",
        "PIVquiverFrame.png",
        "PIVquiverFiltered.png",
        "velocity_utm.png",
        "velocity_utm_all.png",
        "velocity_raster_utm.png",
        "velocity_raster_utm_all.png",
        "velocity_raster_arrows_utm.png",
        "velocity_raster_arrows_utm_all.png",
        "velocity_std_utm.png",
        "velocity_std_utm_all.png",
        "velocity_cv_utm.png",
        "velocity_cv_utm_all.png",
    ]
    missing = [f for f in expected if not (tmp_path / f).exists()]
    assert not missing, f"Missing outputs: {missing}"


def test_piv_velocity_nc_structure(tmp_path):
    """velocity.nc contains the expected variables and non-empty spatial dims."""
    import xarray as xr
    from lspiv.piv import run_piv

    run_piv(
        video_path=VIDEO,
        output_dir=str(tmp_path),
        camera_config_path=CONFIG,
        start_frame=1,
        end_frame=5,
        h_a=0.0,
        piv_engine="numba",
        window_size=20,
        min_s2n=1.0,
        min_corr=0.5,
        min_speed=0.02,
        cv_threshold=100.0,
    )

    ds = xr.open_dataset(tmp_path / "velocity.nc")
    for var in ("v_x", "v_y", "corr", "s2n", "v_x_std", "v_y_std", "speed_std"):
        assert var in ds, f"Missing variable: {var}"
    assert ds.sizes["x"] > 0 and ds.sizes["y"] > 0
