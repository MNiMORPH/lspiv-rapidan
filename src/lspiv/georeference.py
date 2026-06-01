import argparse
import json
import os

import cv2
import numpy as np
import rasterio
import rasterio.enums
import rasterio.transform


def extract_frame(video_path, frame_number=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def load_orthophoto_gray(orthophoto_path, max_pixels=4_000_000):
    """Load orthophoto as uint8 grayscale, downsampling to at most max_pixels."""
    with rasterio.open(orthophoto_path) as src:
        h, w = src.height, src.width
        scale = min(1.0, (max_pixels / (h * w)) ** 0.5)
        out_h = max(1, int(h * scale))
        out_w = max(1, int(w * scale))

        data = src.read(
            out_shape=(src.count, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )
        # Affine transform that maps downsampled pixel coords → world coords
        scaled_transform = src.transform * rasterio.transform.Affine.scale(
            w / out_w, h / out_h
        )
        crs = src.crs

    print(f"Orthophoto loaded at {out_w}x{out_h} (scale {scale:.3f})")

    if data.shape[0] >= 3:
        rgb = np.moveaxis(data[:3], 0, -1).astype(np.float32)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-9) * 255
        gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        band = data[0].astype(np.float32)
        band = (band - band.min()) / (band.max() - band.min() + 1e-9) * 255
        gray = band.astype(np.uint8)

    return gray, scaled_transform, crs


def match_frame_to_orthophoto(frame_bgr, ortho_gray, ratio_threshold=0.7):
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)
    kp_ortho, des_ortho = sift.detectAndCompute(ortho_gray, None)

    print(f"Keypoints — frame: {len(kp_frame)}, orthophoto: {len(kp_ortho)}")

    if des_frame is None or des_ortho is None or len(kp_frame) < 4 or len(kp_ortho) < 4:
        raise RuntimeError("Too few keypoints detected in one or both images.")

    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),   # FLANN_INDEX_KDTREE
        dict(checks=50),
    )
    matches = flann.knnMatch(des_frame, des_ortho, k=2)
    good = [m for m, n in matches if m.distance < ratio_threshold * n.distance]
    print(f"Good matches (Lowe ratio test): {len(good)}")

    if len(good) < 4:
        raise RuntimeError(f"Only {len(good)} good matches; need at least 4.")

    src_pts = np.float32([kp_frame[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ortho[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")

    inlier_mask = mask.ravel().astype(bool)
    n_inliers = inlier_mask.sum()
    print(f"RANSAC inliers: {n_inliers} / {len(good)}")

    if n_inliers < 4:
        raise RuntimeError(f"Only {n_inliers} RANSAC inliers; georeferencing unreliable.")

    return (
        src_pts[inlier_mask].reshape(-1, 2),
        dst_pts[inlier_mask].reshape(-1, 2),
    )


def select_spread_gcps(frame_pts, ortho_pts, n_gcps, frame_shape):
    """Select up to n_gcps spatially distributed points using a grid over the frame."""
    if len(frame_pts) <= n_gcps:
        return frame_pts, ortho_pts

    h, w = frame_shape[:2]
    grid = int(n_gcps ** 0.5) + 1
    cell_w, cell_h = w / grid, h / grid

    selected = {}
    for fp, op in zip(frame_pts, ortho_pts):
        cell = (int(fp[1] / cell_h), int(fp[0] / cell_w))
        if cell not in selected:
            selected[cell] = (fp, op)

    pairs = list(selected.values())[:n_gcps]
    return (
        np.array([p[0] for p in pairs]),
        np.array([p[1] for p in pairs]),
    )


def ortho_pixels_to_world(ortho_pts, transform):
    """Convert orthophoto pixel (col, row) → world (x, y) via rasterio affine transform."""
    xs, ys = rasterio.transform.xy(transform, ortho_pts[:, 1], ortho_pts[:, 0])
    return np.column_stack([xs, ys])


def save_debug_image(frame_bgr, frame_pts, path):
    """Save a copy of the frame with GCP locations marked."""
    debug = frame_bgr.copy()
    for i, (col, row) in enumerate(frame_pts):
        cv2.circle(debug, (int(col), int(row)), 8, (0, 255, 0), -1)
        cv2.putText(
            debug, str(i),
            (int(col) + 10, int(row)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
    cv2.imwrite(path, debug)
    print(f"Debug image saved to {path}")


def georeference(video_path, orthophoto_path, output_path,
                 frame_number=0, n_gcps=20,
                 h_ref=0.0, z_0=0.0,
                 max_ortho_pixels=4_000_000,
                 debug_image_path=None):

    print(f"Extracting frame {frame_number} from {video_path}")
    frame = extract_frame(video_path, frame_number)
    h, w = frame.shape[:2]

    print(f"Loading orthophoto: {orthophoto_path}")
    ortho_gray, ortho_transform, crs = load_orthophoto_gray(orthophoto_path, max_ortho_pixels)

    print("Matching features (SIFT + RANSAC)...")
    frame_pts, ortho_pts = match_frame_to_orthophoto(frame, ortho_gray)

    frame_pts, ortho_pts = select_spread_gcps(frame_pts, ortho_pts, n_gcps, frame.shape)
    print(f"Selected {len(frame_pts)} spatially distributed GCPs.")

    world_pts = ortho_pixels_to_world(ortho_pts, ortho_transform)

    if debug_image_path:
        save_debug_image(frame, frame_pts, debug_image_path)

    epsg = crs.to_epsg()
    if epsg is None:
        raise RuntimeError(
            f"Could not determine EPSG code from orthophoto CRS: {crs}. "
            "Reproject the orthophoto to a CRS with a known EPSG code."
        )

    config = {
        "height": h,
        "width": w,
        "gcps": {
            "src": frame_pts.tolist(),
            "dst": world_pts.tolist(),
            "h_ref": h_ref,
            "z_0": z_0,
        },
        "crs": epsg,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Camera config written to {output_path}  (CRS: EPSG:{epsg})")


def main():
    parser = argparse.ArgumentParser(
        description="Georeference a stabilized video frame against a GeoTIFF orthophoto."
    )
    parser.add_argument("--video",              required=True,
                        help="Stabilized video path")
    parser.add_argument("--orthophoto",         required=True,
                        help="Georeferenced GeoTIFF orthophoto")
    parser.add_argument("--output",             required=True,
                        help="Output camera_config.json path")
    parser.add_argument("--frame",              type=int, default=0,
                        help="Frame index to use for matching (default: 0)")
    parser.add_argument("--n-gcps",             type=int, default=20,
                        help="Number of GCPs to extract (default: 20)")
    parser.add_argument("--h-ref",              type=float, default=0.0,
                        help="Reference water surface elevation (m)")
    parser.add_argument("--z-0",               type=float, default=0.0,
                        help="Bed elevation (m)")
    parser.add_argument("--max-ortho-pixels",   type=int, default=4_000_000,
                        help="Downsample orthophoto to at most this many pixels for SIFT (default: 4M)")
    parser.add_argument("--debug-image",        default=None,
                        help="Optional path to save GCP coverage image for QC")
    args = parser.parse_args()

    georeference(
        video_path=args.video,
        orthophoto_path=args.orthophoto,
        output_path=args.output,
        frame_number=args.frame,
        n_gcps=args.n_gcps,
        h_ref=args.h_ref,
        z_0=args.z_0,
        max_ortho_pixels=args.max_ortho_pixels,
        debug_image_path=args.debug_image,
    )


if __name__ == "__main__":
    main()
