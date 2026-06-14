"""lspiv — large-scale PIV from drone video."""

# The conda lspiv-env ships a newer PROJ (9.7+, schema v1.6) than the base
# anaconda environment (schema v1.2).  When the package is invoked without
# full conda activation (e.g. via `python -m lspiv.piv` with the env's
# interpreter, or via `lspiv-piv` as an entry-point), the PROJ C library falls
# back to the base env's proj.db and raises "DATABASE.LAYOUT.VERSION.MINOR"
# errors inside both pyproj and rasterio.
#
# Fix: set PROJ_DATA / PROJ_LIB to the conda env's own share/proj **before**
# any rasterio or pyproj import.  setdefault leaves an explicit user override
# in place.  Running this in __init__.py ensures it fires first regardless of
# which lspiv module is the entry point.

import os
import sys

_proj_data = os.path.join(sys.prefix, "share", "proj")
if os.path.isdir(_proj_data):
    # Always point at this env's PROJ database.  The base-conda PROJ_DATA
    # leaks into the environment when the base env is active; overriding here
    # ensures rasterio and GDAL pick up the correct schema version.
    os.environ["PROJ_DATA"] = _proj_data
    os.environ["PROJ_LIB"]  = _proj_data
    try:
        import pyproj.datadir
        pyproj.datadir.set_data_dir(_proj_data)
    except Exception:
        pass
