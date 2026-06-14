import argparse
import os

import cv2
import numpy as np
from stabilo import Stabilizer


def stabilize_video(input_path, output_path, border_threshold=10):
    """Stabilize a video and crop out the black stabilization border.

    Two-pass approach:
      Pass 1 — stabilize all frames to a temp file, accumulate the tightest
                crop rectangle (max inset on each side across all frames).
      Pass 2 — re-read the temp file and write the cropped final output.

    border_threshold: pixel-value threshold below which a row/column is
    considered part of the black stabilization fill.
    """
    stabilizer = Stabilizer()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp_path = output_path + ".tmp.mp4"

    # ── Pass 1: stabilize → temp file, track crop insets ──────────────────
    out_tmp = cv2.VideoWriter(
        tmp_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    crop_top    = 0
    crop_left   = 0
    crop_bottom = height
    crop_right  = width

    first_frame = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame:
            stabilizer.set_ref_frame(frame)
            first_frame = False
            stabilized = frame
        else:
            stabilizer.stabilize(frame)
            stabilized = stabilizer.warp_cur_frame()

        out_tmp.write(stabilized)

        gray     = cv2.cvtColor(stabilized, cv2.COLOR_BGR2GRAY)
        nonblack = gray > border_threshold
        rows     = np.any(nonblack, axis=1)   # True for rows with real content
        cols     = np.any(nonblack, axis=0)

        if rows.any() and cols.any():
            crop_top    = max(crop_top,    int(np.argmax(rows)))
            crop_bottom = min(crop_bottom, height - int(np.argmax(rows[::-1])))
            crop_left   = max(crop_left,   int(np.argmax(cols)))
            crop_right  = min(crop_right,  width  - int(np.argmax(cols[::-1])))

    cap.release()
    out_tmp.release()

    crop_h = crop_bottom - crop_top
    crop_w = crop_right  - crop_left
    removed_top    = crop_top
    removed_bottom = height - crop_bottom
    removed_left   = crop_left
    removed_right  = width  - crop_right
    print(
        f"Stabilization crop: removed {removed_top}px top, {removed_bottom}px bottom, "
        f"{removed_left}px left, {removed_right}px right  →  {crop_w}×{crop_h} px output"
    )

    # ── Pass 2: read temp file, write cropped output ───────────────────────
    cap2    = cv2.VideoCapture(tmp_path)
    out_final = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (crop_w, crop_h)
    )

    while cap2.isOpened():
        ret, frame = cap2.read()
        if not ret:
            break
        out_final.write(frame[crop_top:crop_bottom, crop_left:crop_right])

    cap2.release()
    out_final.release()
    os.remove(tmp_path)

    print(f"Stabilization complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stabilize a drone video.")
    parser.add_argument("--input",            required=True, help="Input video path")
    parser.add_argument("--output",           required=True, help="Output stabilized video path")
    parser.add_argument("--border-threshold", type=int, default=10,
                        help="Pixel brightness below which a row/col is considered "
                             "black stabilization fill (default: 10)")
    args = parser.parse_args()
    stabilize_video(args.input, args.output, border_threshold=args.border_threshold)


if __name__ == "__main__":
    main()
