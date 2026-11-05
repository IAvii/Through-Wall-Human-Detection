import cv2
import os

video_path = r'd:\College projects\Mini Project Wifi RADAR esp 32 c6\WhatsApp Video 2026-04-26 at 5.33.14 PM.mp4'
out_dir = r'd:\College projects\Mini Project Wifi RADAR esp 32 c6\data'

v = cv2.VideoCapture(video_path)
total = int(v.get(cv2.CAP_PROP_FRAME_COUNT))
fps = v.get(cv2.CAP_PROP_FPS)
print(f"Frames: {total}, FPS: {fps:.1f}, Duration: {total/fps:.1f}s")

# extract 6 evenly-spaced frames
for i in range(6):
    pos = int((i + 0.5) * total / 6)
    v.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = v.read()
    if ret:
        path = os.path.join(out_dir, f"frame_{i}.jpg")
        cv2.imwrite(path, frame)
        print(f"Saved frame {i} at position {pos} -> {path}")
    else:
        print(f"Failed to read frame at position {pos}")

v.release()
print("Done")
