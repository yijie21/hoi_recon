"""
Reproject HORT's per-frame object point cloud (+ hand mesh) back onto the
original HOT3D RGB frame, and render an original-vs-overlay side-by-side video.

Why not naive "real K" backprojection:
  HORT/demo.py never reads intrinsics.npy. Its object cloud + hand mesh live in
  the coordinate frame of WiLoR's *virtual crop camera*
  (cam_intr = fx=fy=4375/224*... , cx=cy=112, for the 224x224 hand+object crop),
  not the real HOT3D camera. The crop's hand/object 3D points are consistent
  WITHIN that crop-camera frame (correct 2D shape when projected with cam_intr),
  but the absolute depth is a heuristic (~7-9m here) that has no relation to the
  real ~0.3-0.6m capture distance, and directly plugging those XYZ into the real
  HOT3D K (fx=fy=512, cx=cy=511.5, full 1024x1024 frame) would just dump every
  frame near the image center regardless of where the hand actually is (because
  the crop camera always centers optical axis on the crop), giving a static,
  uninformative overlay that does not track hand/object motion.

  Instead we "un-crop": project each 3D point with the crop's own cam_intr to
  get its 224x224 crop-pixel location, then map that crop-pixel back into the
  original 1024x1024 frame using the *known* crop bounding box (`ho_bbox`,
  saved by demo.py) that WiLoR/LangSAM detected in that frame. This reproduces
  correct 2D placement + shape of the reconstruction on the real frame; the
  only thing NOT recovered is true metric depth/scale (documented caveat).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def unproject_and_map(d):
    """Return (Nx2 hand pixel, Nx2 obj pixel) in ORIGINAL (unflipped) frame coords."""
    cam_intr = np.array(d["cam_intr"], dtype=np.float64)  # (3,4) or (3,3)
    if cam_intr.shape[-1] == 4:
        K = cam_intr[:, :3]
    else:
        K = cam_intr
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    ho_bbox = np.array(d["ho_bbox"], dtype=np.float64)  # x,y,w,h in the (possibly flipped) frame
    bx, by, bw, bh = ho_bbox
    img_h, img_w = d["img_hw"]
    flipped = bool(d.get("flipped", False))

    pts = np.array(d["pointclouds_up"], dtype=np.float64)
    palm = np.array(d["handpalm"], dtype=np.float64)
    objtrans = np.array(d["objtrans"], dtype=np.float64)
    obj_cam = pts + palm + objtrans  # crop-camera frame

    def project(xyz):
        z = xyz[:, 2]
        u = fx * xyz[:, 0] / z + cx
        v = fy * xyz[:, 1] / z + cy
        # crop-pixel (224x224) -> full-frame pixel via the known crop bbox
        full_u = bx + (u / 224.0) * bw
        full_v = by + (v / 224.0) * bh
        if flipped:
            full_u = img_w - full_u
        return np.stack([full_u, full_v], axis=1)

    obj_px = project(obj_cam)
    return obj_px


def load_mesh_verts(obj_path):
    verts = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    return np.array(verts, dtype=np.float64)


def project_hand(verts, d):
    cam_intr = np.array(d["cam_intr"], dtype=np.float64)
    K = cam_intr[:, :3] if cam_intr.shape[-1] == 4 else cam_intr
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    ho_bbox = np.array(d["ho_bbox"], dtype=np.float64)
    bx, by, bw, bh = ho_bbox
    img_h, img_w = d["img_hw"]
    flipped = bool(d.get("flipped", False))
    z = verts[:, 2]
    u = fx * verts[:, 0] / z + cx
    v = fy * verts[:, 1] / z + cy
    full_u = bx + (u / 224.0) * bw
    full_v = by + (v / 224.0) * bh
    if flipped:
        full_u = img_w - full_u
    return np.stack([full_u, full_v], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True, help="folder of source RGB frames fed to demo.py")
    ap.add_argument("--hort_out_dir", required=True, help="demo.py --out_folder for this clip")
    ap.add_argument("--out_video", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    args = ap.parse_args()

    json_paths = sorted(glob.glob(os.path.join(args.hort_out_dir, "*.json")))
    frames = []
    for jp in json_paths:
        stem = os.path.splitext(os.path.basename(jp))[0]
        img_path = os.path.join(args.frames_dir, stem + ".png")
        if not os.path.exists(img_path):
            print(f"missing source frame for {stem}, skipping")
            continue
        d = json.load(open(jp))
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        obj_px = unproject_and_map(d)
        obj_path = os.path.join(args.hort_out_dir, stem + ".obj")
        hand_px = None
        if os.path.exists(obj_path):
            verts = load_mesh_verts(obj_path)
            # hand verts are already in the SAME crop-camera frame (verts + cam_t applied by demo.py)
            # but load_mesh_verts loads world verts (verts+cam_t); reuse unproject with cam_intr/ho_bbox directly
            hand_px = project_hand(verts, d)

        overlay = img.copy()
        # object point cloud: blue dots
        for (u, v) in obj_px:
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                cv2.circle(overlay, (ui, vi), 2, (255, 120, 30), -1, lineType=cv2.LINE_AA)
        # hand mesh verts: purple dots
        if hand_px is not None:
            for (u, v) in hand_px:
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < w and 0 <= vi < h:
                    cv2.circle(overlay, (ui, vi), 1, (170, 60, 160), -1, lineType=cv2.LINE_AA)

        bx, by, bw, bh = np.array(d["ho_bbox"], dtype=np.int32)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)

        side = np.concatenate([img, overlay], axis=1)
        cv2.putText(side, "RGB (real frame)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side, "HORT overlay (obj=blue, hand=purple)", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        frames.append(side)

    if not frames:
        print("NO FRAMES produced, aborting")
        return

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out_video, fourcc, args.fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()
    print(f"wrote {args.out_video} with {len(frames)} frames at {args.fps} fps")


if __name__ == "__main__":
    main()
