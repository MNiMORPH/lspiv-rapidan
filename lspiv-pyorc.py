import pyorc
import numpy as np
import os
import cv2

# 1. Define paths and video details
video_path = "output_stabilized_video.mp4"

# 2. Create the camera configuration

# Create the VideoCapture object
cap = cv2.VideoCapture(video_path)

# Define image dimensions (e.g., from a video frame)
nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define Ground Control Points (GCPs) if available.
gcps = {
    "src": [[1, height], [1, 1], [width, 1], [width, height]], # pixels (col,row)
    "dst": [[0, 0], # LL
            [0, height/100.], # UL
            [width/100., height/100.], # UR
            [width/100., 0]], # LR                     # real-world coords
    "h_ref": 0.0,
    "z_0": 0.0
}


# Define the Coordinate Reference System (CRS) for the real-world coordinates.
# This can be an EPSG code (e.g., 32735 for WGS 84 / UTM zone 35S) or a Proj string.
crs = 32615

# Create the CameraConfig instance
camera_config = pyorc.CameraConfig(
    height=height,
    width=width,
    gcps=gcps,
    crs=crs,
    # Optional parameters can be added here, such as:
    # focal_length=12.0,  # Example focal length in mm
    # pixel_size=0.003,   # Example pixel size in mm
    # distortion_coefficients=[0.1, -0.05, 0.001, 0.001, 0.0] # Example distortion coefficients
)

# bbox from corners (row, col)
# Needed to rotate order for it to work
camera_config.set_bbox_from_corners( [
                                    [0, height], # UL
                                    [width, height], # UR
                                    [width, 0],
                                    [0, 0], # LL
                                    ] )

print("Generic PyORC CameraConfig created successfully.")
print(f"Image dimensions: {camera_config.height}x{camera_config.width}")
print(f"Number of GCPs: {len(camera_config.gcps) if camera_config.gcps is not None else 0}")
print(f"CRS: {camera_config.crs}")


# 3. Create a Video object
# Specify start_frame, end_frame, and actual water level (h_a) if known
# Note: could set box for stabilization here too
video = pyorc.Video(
    video_path,
    camera_config=camera_config,
    start_frame=1,  # Start processing from frame 1
    end_frame=nframes-1, # End processing at the last frame - 1
    h_a = 0.
)

# We don't know the water level, and it varies
#     h_a=1.5           # Actual water level during video capture (in meters) -- unknown

# 4. Get frames and normalize them to enhance contrast: project them
da = video.get_frames()
da_norm = da.frames.normalize()
# Check
p = da_norm[0].frames.plot(cmap="gray")
# Continue
da_norm_proj = da_norm.frames.project(method="numpy")
# Check
da_norm_proj[0].frames.plot(cmap="gray")


# 5. Estimate surface velocities
piv = da_norm_proj.frames.get_piv(engine="numba")


#piv_px_per_frame = frames.get_piv(
#    engine="numba", window_size=64,
#    ensemble_corr=True, s2n_min=3, corr_min=0.2
#)


# 5.5 plot

"""
# extract frames again, but now with rgb
da_rgb = video.get_frames(method="rgb")
# project the rgb frames, same as before
da_rgb_proj = da_rgb.frames.project()

# plot the first frame in geographical mode
p = da_rgb_proj[0].frames.plot(mode="geographical")

ds = piv # alias
ds_mean = ds.mean(dim="time", keep_attrs=True)

# first a pcolormesh
ds_mean.velocimetry.plot.pcolormesh(
    ax=p.axes,
    alpha=0.3,
    cmap="rainbow",
    add_colorbar=True,
    vmax=0.6
)

ds_mean.velocimetry.plot(
    ax=p.axes,
    color="w",
    alpha=0.5,
)
"""

plt.figure()
p = da_rgb_proj[0].frames.plot()
plt.savefig('Frame.png')

plt.figure()
p = da_rgb_proj[0].frames.plot()
ds_mean.velocimetry.plot( ax=p.axes )
plt.savefig('PIVquiverFrame.png')

plt.show()




# You can adjust parameters like window_size, overlap, etc.
velocities = frames.get_piv() # not frames_proj

# 6. (Optional) Filter spurious velocities
# pyorc offers various filtering options
filtered_velocities = pyorc.velocimetry.filter_velocities(velocities)

# 7. (Optional) Extract velocities over a transect and estimate discharge
# First, define a transect (e.g., using geographical coordinates or points in camera view)
# For simplicity, we assume a pre-defined transect in the camera_config
# transect_config = camera_config.get_transect("my_river_transect")
# discharge_results = pyorc.transect.estimate_discharge(filtered_velocities, transect_config)

# 8. (Optional) Plot results
# pyorc allows plotting velocities and frames in various perspectives
# pyorc.plot.plot_velocities(filtered_velocities, in_camera_view=True)
# pyorc.plot.plot_frames(frames, frame_number=0)

# 9. (Optional) Save results
# filtered_velocities.to_netcdf("path/to/save/velocities.nc")

