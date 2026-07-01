"""Option 1: per-clip hand-scale + translation correction for render_and_compare on wild6.

WHY: the reconstructed MANO hand is metrically undersized (longest span ~0.147 m vs a real adult
hand ~0.19 m -> ~0.77x), set at the WiLoR hand stage, so it backprojects too small and the
fingertips fall short (apparent position offset). See compare/README.md "hand scale".

TARGET CHOICE (important): on wild6 the obvious 2D targets are BOTH unreliable for hand SIZE:
  - stage2 kp2d (WiLoR 2D keypoints) is garbage here (scattered across the frame, 15% out-of-frame);
  - the stage1 YOLO hand box is ~2.25x the projected hand because it includes forearm/wrist, so
    fitting size to it would grossly oversize the hand.
What IS reliable is (a) the box CENTER (hand location tracks well) and (b) the canonical hand-size
prior. So we correct with a single GLOBAL scale from the metric prior + a single GLOBAL image-plane
translation that snaps the projected hand centroid onto the detection-box centre. (This is the
standard way to resolve monocular hand-scale ambiguity; it is a global 2-parameter-family fix, not a
per-frame re-fit.)

Writes render_and_compare/runs/wild6_real/stage7_contact_optim/arrays_handfit.npz (a copy of
arrays.npz with hand_verts/hand_joints corrected). backproject.py and rc_to_scene.py prefer it.

Usage: python compare/fit_hand_scale.py [canonical_span_m=0.19]
"""
import sys, numpy as np

ROOT = "render_and_compare/runs/wild6_real"


def main(canonical=0.19):
    K = np.load(f"{ROOT}/stage0_preprocess/arrays.npz")["intrinsics"].astype(np.float64)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    a = dict(np.load(f"{ROOT}/stage7_contact_optim/arrays.npz", allow_pickle=True))
    z1 = np.load(f"{ROOT}/stage1_detect_track/arrays.npz", allow_pickle=True)
    box, bvalid = z1["hand_boxes"][:, 0], z1["hand_valid"][:, 0]   # slot 0 = left hand
    hv, hj = a["hand_verts"].astype(np.float64), a["hand_joints"].astype(np.float64)
    T = len(hv)

    # --- global scale from canonical metric prior ---
    span0 = np.median([(hv[t].max(0) - hv[t].min(0)).max() for t in range(T)])
    s = float(canonical / span0)

    # scale each frame about its wrist (joint 0): preserves pose + wrist position
    wrist = hj[:, 0:1, :]
    hv_s = wrist + s * (hv - wrist)
    hj_s = wrist + s * (hj - wrist)

    def centroid_uv(V):
        z = np.clip(V[:, 2], 1e-4, None)
        return np.array([(V[:, 0] / z * fx + cx).mean(), (V[:, 1] / z * fy + cy).mean()])

    # --- global image-plane translation: projected hand centroid -> box centre ---
    duv, zc = [], []
    for t in range(T):
        if not bvalid[t]:
            continue
        bc = np.array([(box[t][0] + box[t][2]) / 2, (box[t][1] + box[t][3]) / 2])
        duv.append(bc - centroid_uv(hv_s[t]))
        zc.append(hv_s[t][:, 2].mean())
    duv = np.median(np.stack(duv), 0)          # robust px offset
    zbar = float(np.mean(zc))
    delta = np.array([duv[0] * zbar / fx, duv[1] * zbar / fy, 0.0])   # px -> metres at mean depth

    hv_f = (hv_s + delta).astype(np.float32)
    hj_f = (hj_s + delta).astype(np.float32)

    # --- report before/after ---
    def center_err(V):
        e = []
        for t in range(T):
            if not bvalid[t]:
                continue
            bc = np.array([(box[t][0] + box[t][2]) / 2, (box[t][1] + box[t][3]) / 2])
            e.append(np.linalg.norm(bc - centroid_uv(V[t])))
        return float(np.median(e))
    span1 = np.median([(hv_f[t].max(0) - hv_f[t].min(0)).max() for t in range(T)])
    print(f"hand longest span: {span0:.3f} m -> {span1:.3f} m   (scale s = {s:.3f})")
    print(f"global translation delta = [{delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f}] m  "
          f"(from {duv.round(0)} px @ z={zbar:.3f})")
    print(f"projected-centroid vs box-centre error (median px): {center_err(hv):.0f} -> {center_err(hv_f):.0f}")

    a["hand_verts"], a["hand_joints"] = hv_f, hj_f
    out = f"{ROOT}/stage7_contact_optim/arrays_handfit.npz"
    np.savez(out, **a)
    print("wrote", out)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.19)
