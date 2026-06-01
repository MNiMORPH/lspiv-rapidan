import argparse
import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyorc


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
        [
            [0, height],
            [width, height],
            [width, 0],
            [0, 0],
        ]
    )
    return camera_config


def run_piv(video_path, output_dir, camera_config_path=None,
            start_frame=1, end_frame=None, h_a=0.0, piv_engine="numba"):
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

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run PIV on a stabilized drone video.")
    parser.add_argument("--video",         required=True,  help="Stabilized input video path")
    parser.add_argument("--camera-config", default=None,   help="Camera config JSON (from georeferencing step)")
    parser.add_argument("--output-dir",    required=True,  help="Directory to write output figures")
    parser.add_argument("--start-frame",   type=int, default=1)
    parser.add_argument("--end-frame",     type=int, default=None)
    parser.add_argument("--h-a",           type=float, default=0.0, help="Actual water level (m)")
    parser.add_argument("--piv-engine",    default="numba", choices=["numba", "opencv"])
    args = parser.parse_args()

    run_piv(
        video_path=args.video,
        output_dir=args.output_dir,
        camera_config_path=args.camera_config,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        h_a=args.h_a,
        piv_engine=args.piv_engine,
    )


if __name__ == "__main__":
    main()
