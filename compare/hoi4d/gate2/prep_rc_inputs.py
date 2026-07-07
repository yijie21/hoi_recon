"""Prepare RC inputs for the substrate matrix: per-clip mp4 (all frames, 15fps)
and an object SAM2 point prompt derived from the validated kill-test masks
(distance-transform peak of the frame-0 object mask = deepest interior point).

Writes <clip>/gate2/clip_rc.mp4 (kettle_N15 keeps its original kettle_5s.mp4)
and gate2/rc_inputs.json with {clip: {video, obj_prompt, frames}}.
"""
import glob, json, os, subprocess
import numpy as np
import cv2

CLIPS_ROOT = "/workspace/hoi4d/clips"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rc_inputs.json")


def main():
    inputs = {}
    for clip in sorted(glob.glob(os.path.join(CLIPS_ROOT, "*"))):
        if not os.path.isdir(clip):
            continue
        name = os.path.basename(clip)
        frames = sorted(glob.glob(os.path.join(clip, "rgb", "*.jpg")))
        video = os.path.join(clip, "gate2", "clip_rc.mp4")
        if name == "kettle_N15":
            video = os.path.join(clip, "kettle_5s.mp4")
        elif not os.path.exists(video):
            os.makedirs(os.path.dirname(video), exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", "15",
                 "-i", os.path.join(clip, "rgb", "%06d.jpg"),
                 "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", video],
                check=True)
        m = cv2.imread(os.path.join(clip, "masks", "frame_000000_masks", "object.png"),
                       cv2.IMREAD_GRAYSCALE)
        mb = (m > 127).astype(np.uint8)
        dt = cv2.distanceTransform(mb, cv2.DIST_L2, 5)
        y, x = np.unravel_index(np.argmax(dt), dt.shape)
        inputs[name] = {"video": video, "obj_prompt": [float(x), float(y)],
                        "frames": len(frames)}
        print(f"{name}: {len(frames)} frames, obj_prompt=({x},{y}), video={os.path.basename(video)}")
    with open(OUT, "w") as f:
        json.dump(inputs, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
