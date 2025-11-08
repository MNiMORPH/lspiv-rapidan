import cv2
import numpy as np
from stabilo import Stabilizer 

# Create an instance of the Stabilizer class with default parameters
stabilizer = Stabilizer() 

# Open the video file
video_path = 'MAX_0102.MP4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Optional: Set up output video writer
out = cv2.VideoWriter('output_stabilized_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 3. Process the video frame by frame
first_frame = True
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale if your stabilization method requires it
    # Some stabilo internal methods handle color images, check documentation for specifics
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if first_frame:
        # Set the first frame (or any chosen frame) as the reference
        stabilizer.set_ref_frame(frame) # Use frame or gray_frame
        first_frame = False
        stabilized_frame = frame # No transformation needed for the first frame
    else:
        # Stabilize the current frame relative to the reference
        stabilizer.stabilize(frame) # Use frame or gray_frame
        # Get the stabilized (warped) frame
        stabilized_frame = stabilizer.warp_cur_frame()

    # Optional: Display the stabilized frame (for live viewing or debugging)
    # cv2.imshow('Original', frame)
    # cv2.imshow('Stabilized', stabilized_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # 4. Write the stabilized frame to the output video
    out.write(stabilized_frame)

# 5. Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video stabilization complete. Output saved to output_stabilized_video.mp4")


