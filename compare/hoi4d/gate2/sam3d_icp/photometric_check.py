"""Held-out photometric alignment metric: does the TEXTURED mesh, rendered
with each arm's final poses, look like the real RGB video where it lands?

Why this exists: every other metric scores against the evidence the optimizer
fits (GT depth, mask), so it is partially circular; and none is
rotation-sensitive on a rotation-symmetric depth patch. RGB appearance is
never used by the pipeline's object registration, and the kettle's texture
(spout, handle, lid pattern) breaks the symmetry — so photometric similarity
is a genuinely held-out, rotation-sensitive verdict on "does the mesh align
with the object in the video".

Method per frame: splat the dense SAM-3D vertex-colored mesh (417k verts,
z-buffered at BIN px) with the arm's stage-7 pose + stage-4 canonical scale;
compare rendered vs real RGB. Two regions: ALL rendered pixels ("cov" —
punishes overhang onto background), and rendered ∩ object mask ("mask" —
pure texture alignment). Scores: zero-mean normalized cross-correlation on
gray (robust to albedo-vs-shaded lighting gap), SSIM on the bbox crop, and
LPIPS (AlexNet) if installed.

Usage: photometric_check.py [run_suffix ...]  (default: icp2 icp4 icp5 icpj3)
"""
import json
import os
import sys

import cv2
import numpy as np
import trimesh

RC = "/workspace/code/hoi_recon/render_and_compare/runs"
GT = f"{RC}/kettle_gt"
OUT = os.path.dirname(os.path.abspath(__file__))
BIN = 4
FLIP = np.diag([1.0, -1.0, -1.0])

# which stage-3 cache (glb + npz) each arm's mesh came from
MESH_SRC = {"base": "kettle_gt", "icp": "kettle_gt",
            "base2": "kettle_gt_icp2", "icp2": "kettle_gt_icp2",
            "icp3": "kettle_gt_icp2", "icp4": "kettle_gt_icp2",
            "icp5": "kettle_gt_icp2", "icpj3": "kettle_gt_icp2"}


def dense_canonical(run):
    """Dense vertex-colored GLB mapped into the pipeline canonical frame with
    exactly the stage-3 transform (flip, center, scale — derived from the
    decimated npz verts, which is what stage 3 used)."""
    d = np.load(f"{RC}/{run}/stage3_object/sam3d/object.npz")
    v_dec = d["verts"].astype(np.float64) @ FLIP.T
    mu, ext = v_dec.mean(0), (v_dec.max(0) - v_dec.min(0)).max()
    s3 = np.load(f"{RC}/{run}/stage3_object/arrays.npz")
    f = float((s3["verts"].max(0) - s3["verts"].min(0)).max() / ext)

    g = trimesh.load(f"{RC}/{run}/stage3_object/sam3d/object.glb")
    m = (trimesh.util.concatenate(list(g.geometry.values()))
         if isinstance(g, trimesh.Scene) else g)
    V = (np.asarray(m.vertices, np.float64) @ FLIP.T - mu) * f
    C = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.uint8)
    # sanity: must reproduce the decimated canonical verts' bbox
    err = np.abs((v_dec - mu) * f - s3["verts"]).max()
    assert err < 1e-6, f"canonical map mismatch: {err}"
    return V, C


def arm_scale(name):
    with open(f"{RC}/kettle_gt_{name}/stage4_align/meta.json") as fp:
        meta = json.load(fp)
    icp = meta.get("object_icp") or {}
    ax = icp.get("global_scale_axes")
    if ax is not None:
        return np.asarray(ax, np.float64)
    return np.full(3, float(icp.get("global_scale", 1.0)))


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def main():
    names = sys.argv[1:] or ["icp2", "icp4", "icp5", "icpj3"]
    K = np.load(f"{GT}/stage0_preprocess/arrays.npz")["intrinsics"]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Wd, Hd = 1920 // BIN, 1080 // BIN

    try:
        from skimage.metrics import structural_similarity as ssim_fn
    except ImportError:
        ssim_fn = None
    lpips_fn = None
    try:
        import lpips
        import torch
        _net = lpips.LPIPS(net="alex", spatial=True, verbose=False).cuda().eval()

        def lpips_fn(a, b, region):                            # uint8 HxWx3
            ta = torch.from_numpy(a).permute(2, 0, 1)[None].float().cuda()
            tb = torch.from_numpy(b).permute(2, 0, 1)[None].float().cuda()
            with torch.no_grad():
                m = _net(ta / 127.5 - 1, tb / 127.5 - 1)[0, 0].cpu().numpy()
            reg = cv2.resize(region.astype(np.uint8), m.shape[::-1]) > 0
            return float(m[reg].mean()) if reg.any() else np.nan
    except Exception:
        pass

    meshes, scales, runs = {}, {}, {}
    for n in names:
        V, C = dense_canonical(MESH_SRC[n])
        meshes[n] = (V * arm_scale(n), C)
        runs[n] = np.load(f"{RC}/kettle_gt_{n}/stage8_eval/pseudo_gt.npz")
    T = len(runs[names[0]]["obj_poses"])

    res = {n: {"ncc_cov": [], "ncc_mask": [], "ssim": [], "lpips": []}
           for n in names}
    for t in range(T):
        img = cv2.imread(f"{GT}/stage0_preprocess/frames/{t:05d}.jpg")
        real = cv2.resize(img, (Wd, Hd), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(np.load(f"{GT}/stage1_detect_track/masks/{t:05d}.npy")
                          .astype(np.uint8), (Wd, Hd),
                          interpolation=cv2.INTER_AREA) > 0
        for n in names:
            V, C = meshes[n]
            M = runs[n]["obj_poses"][t]
            X = V @ M[:3, :3].T + M[:3, 3]
            z = X[:, 2]
            ok = z > 0.1
            u = np.clip((X[ok, 0] / z[ok] * fx + cx) / BIN, 0, Wd - 1).astype(int)
            v = np.clip((X[ok, 1] / z[ok] * fy + cy) / BIN, 0, Hd - 1).astype(int)
            order = np.argsort(z[ok])[::-1]          # far-to-near overwrite
            rend = np.zeros((Hd, Wd, 3), np.uint8)
            rend[v[order], u[order]] = C[ok][order][:, ::-1]   # RGB->BGR
            cov = np.zeros((Hd, Wd), bool)
            cov[v, u] = True
            cov = cv2.morphologyEx(cov.astype(np.uint8), cv2.MORPH_CLOSE,
                                   np.ones((3, 3), np.uint8)) > 0
            hole = cov & (rend.sum(2) == 0)
            rend[hole] = cv2.blur(rend, (3, 3))[hole]  # fill splat pinholes

            # chroma NCC: the decal/handle pattern is chroma; shading (which
            # the albedo render cannot reproduce) lives mostly in L
            lab_r = cv2.cvtColor(rend, cv2.COLOR_BGR2LAB).astype(np.float64)
            lab_t = cv2.cvtColor(real, cv2.COLOR_BGR2LAB).astype(np.float64)
            both = cov & mask

            def cncc(reg):
                return ncc(lab_r[reg][:, 1:].ravel(), lab_t[reg][:, 1:].ravel())

            res[n]["ncc_cov"].append(cncc(cov) if cov.sum() > 100 else np.nan)
            res[n]["ncc_mask"].append(cncc(both) if both.sum() > 100 else np.nan)
            ys, xs = np.nonzero(cov)
            if len(ys) > 100:
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                a, b = rend[y0:y1, x0:x1], real[y0:y1, x0:x1]
                reg = cov[y0:y1, x0:x1]
                if ssim_fn is not None and min(a.shape[:2]) >= 8:
                    _, smap = ssim_fn(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY),
                                      cv2.cvtColor(b, cv2.COLOR_BGR2GRAY),
                                      full=True)
                    res[n]["ssim"].append(float(smap[reg].mean()))
                if lpips_fn is not None:
                    res[n]["lpips"].append(lpips_fn(
                        cv2.resize(a, (128, 128)), cv2.resize(b, (128, 128)),
                        cv2.resize(reg.astype(np.uint8), (128, 128)) > 0))

    summary = {}
    for n in names:
        r = res[n]
        summary[n] = {k: (float(np.nanmedian(v)) if len(v) else None)
                      for k, v in r.items()}
        summary[n]["series"] = {k: np.round(np.asarray(v, float), 4).tolist()
                                for k, v in r.items()}
        s = summary[n]
        print(f"{n:6s} NCC(cov) {s['ncc_cov']:.3f}  NCC(mask) {s['ncc_mask']:.3f}"
              + (f"  SSIM {s['ssim']:.3f}" if s["ssim"] is not None else "")
              + (f"  LPIPS {s['lpips']:.3f}" if s["lpips"] is not None else ""))
    with open(os.path.join(OUT, f"photometric_ab_{'_'.join(names)}.json"), "w") as fp:
        json.dump(summary, fp, indent=1)


if __name__ == "__main__":
    main()
