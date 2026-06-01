import argparse
import cv2
from stabilo import Stabilizer


def stabilize_video(input_path, output_path):
    stabilizer = Stabilizer()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    first_frame = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame:
            stabilizer.set_ref_frame(frame)
            first_frame = False
            stabilized_frame = frame
        else:
            stabilizer.stabilize(frame)
            stabilized_frame = stabilizer.warp_cur_frame()
        out.write(stabilized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Stabilization complete: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stabilize a drone video.")
    parser.add_argument("--input",  required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output stabilized video path")
    args = parser.parse_args()
    stabilize_video(args.input, args.output)


if __name__ == "__main__":
    main()
