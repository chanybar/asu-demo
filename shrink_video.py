import cv2
from pathlib import Path

input_video = "/Users/chandler.white/Desktop/Demo video /mobility_demo/CIS515-Project/Library-Recording.mp4"
output_video = "/Users/chandler.white/Desktop/Demo video /mobility_vision/data/videos/demo_library.mp4"

Path("/Users/chandler.white/Desktop/Demo video /mobility_vision/data/videos").mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Downscale to 720p for the demo to save size
new_height = 720
new_width = int(width * (720 / height))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))

count = 0
max_frames = 300 # 10 seconds at 30fps

while cap.isOpened() and count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    resized = cv2.resize(frame, (new_width, new_height))
    out.write(resized)
    count += 1

cap.release()
out.release()
print(f"Saved {count} frames to {output_video}")
