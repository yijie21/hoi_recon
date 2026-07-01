"""Real 2D/3D perception pass on wild6.mp4 frames — the genuinely-extractable parts of the
EgoAERO front-end from a monocular RGB clip:
  - WiLoR  -> per-frame MANO hand (778 verts + 21 joints + camera)   [SP1 hand component]
  - YOLO   -> manipulated-object (bottle) bounding box               [SP1 semantic/track stub]
  - Depth-Anything -> monocular relative depth map                   [the missing depth channel]
NOT the full 4D-HOI contract: no metric object mesh / 6-DoF track / ego-SLAM (models absent).
Saves per-frame npz + a few raw arrays for visualization."""
import os, sys, glob, json
import numpy as np
import cv2
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAMES = sorted(glob.glob("/tmp/wild6/frames/*.png"))
OUT = "/tmp/wild6/perception"
os.makedirs(OUT, exist_ok=True)

# ---- models ----
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import \
    WiLorHandPose3dEstimationPipeline
wilor = WiLorHandPose3dEstimationPipeline(device=DEVICE, dtype=torch.float32)

from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")  # COCO; class 39 = 'bottle'

from transformers import pipeline as hf_pipeline
depth_est = hf_pipeline("depth-estimation",
                        model="depth-anything/Depth-Anything-V2-Small-hf",
                        device=0 if DEVICE == "cuda" else -1)

records = []
for fi, fp in enumerate(FRAMES):
    bgr = cv2.imread(fp)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    # --- hand (WiLoR / MANO) ---
    hand = None
    try:
        dets = wilor.predict(rgb)
        if dets:
            # pick the largest-bbox hand
            d = max(dets, key=lambda r: (r["hand_bbox"][2]-r["hand_bbox"][0]) *
                                        (r["hand_bbox"][3]-r["hand_bbox"][1]))
            wp = d["wilor_preds"]
            hand = {
                "verts3d": wp["pred_vertices"][0].astype(np.float32),       # (778,3) cam frame
                "joints3d": wp["pred_keypoints_3d"][0].astype(np.float32),  # (21,3)
                "kp2d": wp["pred_keypoints_2d"][0].astype(np.float32),      # (21,2) px
                "cam_t": wp["pred_cam_t_full"][0].astype(np.float32),       # (3,)
                "focal": float(np.ravel(wp["scaled_focal_length"])[0]),
                "is_right": float(d["is_right"]),
                "bbox": np.array(d["hand_bbox"], np.float32),
            }
    except Exception as e:
        print(f"  frame {fi}: hand err {repr(e)[:80]}")

    # --- object (YOLO bottle) ---
    obj_bbox = None
    res = yolo.predict(bgr, verbose=False, conf=0.25)[0]
    if res.boxes is not None and len(res.boxes):
        cls = res.boxes.cls.cpu().numpy().astype(int)
        sel = np.where(cls == 39)[0]  # bottle
        if len(sel):
            b = res.boxes.xyxy.cpu().numpy()[sel]
            conf = res.boxes.conf.cpu().numpy()[sel]
            obj_bbox = b[int(np.argmax(conf))].astype(np.float32)

    # --- monocular depth (save downsized, every 5th frame to limit size) ---
    depth = None
    if fi % 5 == 0:
        from PIL import Image
        dout = depth_est(Image.fromarray(rgb))["depth"]
        depth = np.asarray(dout, np.float32)
        depth = cv2.resize(depth, (W // 2, H // 2))

    np.savez(os.path.join(OUT, f"f{fi:04d}.npz"),
             **({f"hand_{k}": v for k, v in hand.items()} if hand else {}),
             **({"obj_bbox": obj_bbox} if obj_bbox is not None else {}),
             **({"depth": depth} if depth is not None else {}),
             has_hand=np.array(hand is not None),
             has_obj=np.array(obj_bbox is not None),
             img_hw=np.array([H, W]))
    records.append({"frame": fi, "file": os.path.basename(fp),
                    "has_hand": hand is not None, "has_obj": obj_bbox is not None})
    if fi % 10 == 0:
        print(f"frame {fi:3d}/{len(FRAMES)}  hand={hand is not None}  bottle={obj_bbox is not None}", flush=True)

json.dump(records, open(os.path.join(OUT, "index.json"), "w"), indent=2)
nh = sum(r["has_hand"] for r in records); no = sum(r["has_obj"] for r in records)
print(f"\nDONE: {len(records)} frames | hand in {nh} | bottle in {no} | depth on {len(records)//5+1}")
