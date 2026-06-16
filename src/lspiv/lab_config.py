import argparse
import json
import os

import cv2


def make_lab_config(video_path, output_path,
                    width_m=None, height_m=None,
                    corners_px=None, crs=32615):
    """
    Create a camera config for an overhead lab or test setup.

    Without --width/--height, uses pixel/100 placeholder scaling — velocities
    will not be in real units, but the pipeline will run end-to-end for testing.
    With --width/--height, output velocities are in m/s.
    """
    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Default: full frame corners in (col, row) order — LL, UL, UR, LR
    if corners_px is None:
        corners_px = [
            [0,       frame_h],
            [0,       0      ],
            [frame_w, 0      ],
            [frame_w, frame_h],
        ]

    placeholder = width_m is None or height_m is None
    if placeholder:
        width_m  = width_m  or frame_w / 100.0
        height_m = height_m or frame_h / 100.0
        print(
            "WARNING: --width / --height not provided; using pixel/100 placeholder scaling.\n"
            "         Output velocities will NOT be in real units."
        )

    # Physical corners matching pixel order: LL, UL, UR, LR
    corners_world = [
        [0.0,     0.0     ],
        [0.0,     height_m],
        [width_m, height_m],
        [width_m, 0.0     ],
    ]

    config = {
        "height": frame_h,
        "width":  frame_w,
        "gcps": {
            "src":   corners_px,
            "dst":   corners_world,
            "h_ref": 0.0,
            "z_0":   0.0,
        },
        "crs": crs,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    scale = "placeholder" if placeholder else f"{width_m} m × {height_m} m"
    print(f"Lab camera config written to {output_path}  ({scale})")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create a camera config for an overhead lab setup. "
            "Omit --width/--height for a placeholder test config."
        )
    )
    parser.add_argument("--video",   required=True,
                        help="Video path (used only to read frame dimensions)")
    parser.add_argument("--output",  required=True,
                        help="Output camera_config.json path")
    parser.add_argument("--width",   type=float, default=None,
                        help="Physical width of field of view (m); omit for test/placeholder run")
    parser.add_argument("--height",  type=float, default=None,
                        help="Physical height of field of view (m); omit for test/placeholder run")
    parser.add_argument("--corners", default=None,
                        help=(
                            "JSON file containing [[col,row],...] pixel coordinates of "
                            "LL, UL, UR, LR corners. Omit to use full frame."
                        ))
    parser.add_argument("--crs",     type=int, default=32615,
                        help="EPSG code (default: 32615). Ignored for local lab coordinates "
                             "but required for the camera config JSON format.")
    args = parser.parse_args()

    corners_px = None
    if args.corners:
        with open(args.corners) as f:
            corners_px = json.load(f)
        if len(corners_px) != 4:
            raise ValueError("--corners file must contain exactly 4 [col, row] pairs (LL, UL, UR, LR)")

    make_lab_config(
        video_path=args.video,
        output_path=args.output,
        width_m=args.width,
        height_m=args.height,
        corners_px=corners_px,
        crs=args.crs,
    )


if __name__ == "__main__":
    main()
