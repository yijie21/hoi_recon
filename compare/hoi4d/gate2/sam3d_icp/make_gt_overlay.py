"""Overlay video: GT CAD @ GT HOI4D pose (green, left) vs icpj3 estimate
(orange, right) splatted over the original frames -> gt_cad_overlay.mp4.
Shaded z-buffered surface splat at half res, alpha 0.58. Note when reading
it: the GT annotation lives in the depth-camera frame, so GT itself shows a
constant ~20-40 px leftward bias vs the RGB kettle (the same depth<->RGB
misregistration measured in RESULTS.md); and a 2D overlay hides depth and
attitude error - the 3D verdict is gt_pose_eval.py's, not this video's."""
import json, numpy as np, trimesh, cv2
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hoi4d"
SEQ = "ZY20210800002/H2/C12/N15/S196/s04/T2"
GT = "/workspace/code/hoi_recon/render_and_compare/runs/kettle_gt"
RC = "/workspace/code/hoi_recon/render_and_compare/runs"
K = np.load(f"{GT}/stage0_preprocess/arrays.npz")["intrinsics"]
fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
B = 2; Wd, Hd = 1920 // B, 1080 // B
N = 200000

cad = trimesh.load(f"{DS}/HOI4D_CAD_Model_for_release/rigid/Kettle/015.obj", force="mesh")
Pc, fidx = trimesh.sample.sample_surface(cad, N, seed=0)
Pc = np.asarray(Pc) - cad.bounds.mean(0)   # HOI4D poses the bbox CENTER
Nc = np.asarray(cad.face_normals)[fidx]

z8 = np.load(f"{RC}/kettle_gt_icpj3/stage8_eval/pseudo_gt.npz")
est = trimesh.Trimesh(z8["obj_verts"], z8["obj_faces"], process=False)
Pe, fe = trimesh.sample.sample_surface(est, N, seed=0)
Pe = np.asarray(Pe); Ne = np.asarray(est.face_normals)[fe]

def overlay(img, X, Nrm, base):
    z = X[:, 2]; ok = z > 0.1
    u = np.clip((X[ok, 0] / z[ok] * fx + cx) / B, 0, Wd - 1).astype(int)
    v = np.clip((X[ok, 1] / z[ok] * fy + cy) / B, 0, Hd - 1).astype(int)
    lam = 0.35 + 0.65 * np.clip(-(Nrm[ok] @ np.array([0.3, -0.5, -0.81])), 0, 1)
    col = (np.asarray(base)[None] * lam[:, None]).astype(np.uint8)
    order = np.argsort(z[ok])[::-1]
    lay = np.zeros((Hd, Wd, 3), np.uint8); cov = np.zeros((Hd, Wd), bool)
    lay[v[order], u[order]] = col[order]; cov[v, u] = True
    cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    hole = cov & (lay.sum(2) == 0); lay[hole] = cv2.blur(lay, (3, 3))[hole]
    sm = cv2.resize(img, (Wd, Hd))
    out = sm.copy(); out[cov] = (0.42 * sm[cov] + 0.58 * lay[cov]).astype(np.uint8)
    return out

FONT = cv2.FONT_HERSHEY_SIMPLEX
vw = cv2.VideoWriter("gt_cad_overlay_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 15, (Wd * 2, Hd))
for t in range(75):
    img = cv2.imread(f"{GT}/stage0_preprocess/frames/{t:05d}.jpg")
    d = json.load(open(f"{DS}/HOI4D_annotations/{SEQ}/objpose/{t}.json"))["dataList"][0]
    R = Rotation.from_euler("XYZ", [d["rotation"]["x"], d["rotation"]["y"], d["rotation"]["z"]]).as_matrix()
    c = np.array([d["center"]["x"], d["center"]["y"], d["center"]["z"]])
    a = overlay(img, Pc @ R.T + c, Nc @ R.T, (80, 220, 80))
    M = z8["obj_poses"][t]
    b = overlay(img, Pe @ M[:3, :3].T + M[:3, 3], Ne @ M[:3, :3].T, (80, 160, 255))
    for im, txt in ((a, "GT CAD + GT pose (HOI4D annotation)"), (b, "icpj3 estimate")):
        cv2.putText(im, txt, (12, 30), FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(im, txt, (12, 30), FONT, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im, f"frame {t}", (12, Hd - 14), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    vw.write(np.hstack([a, b]))
vw.release()
print("wrote gt_cad_overlay_raw.mp4; transcode: ffmpeg -i ... -c:v libx264 -crf 20 -pix_fmt yuv420p gt_cad_overlay.mp4")
