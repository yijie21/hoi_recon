# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Standalone SAM-3D-Objects inference for the hoi_recon pipeline.

Runs INSIDE the `sam3d-objects` conda env (torch 2.5.1/cu121). Takes one RGB
frame + a binary object mask and produces a textured 3D mesh, saved as a plain
.npz (verts, faces, vertex_colors) plus a .glb for reference — so the hoi_recon
env (torch 2.11/cu128) can load the result without importing any sam3d deps.

Usage:
  python sam3d_infer.py --image frame.jpg --mask mask.npy --out object.npz \
      [--config checkpoints/hf/pipeline.yaml] [--seed 42]
"""
import os
import sys
import argparse

import numpy as np

# inference.py lives in notebook/ and pulls in the pipeline helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_HERE, "notebook"))


def _load_mask(path):
    if path.endswith(".npy"):
        m = np.load(path)
    else:
        from inference import load_image
        m = load_image(path)
    m = m > 0
    if m.ndim == 3:
        m = m[..., -1]
    return m


def _to_trimesh(glb):
    import trimesh
    if isinstance(glb, trimesh.Trimesh):
        return glb
    if isinstance(glb, trimesh.Scene):
        geoms = tuple(glb.geometry.values())
        return geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
    raise TypeError(f"unexpected glb type: {type(glb)}")


def _decimate(verts, faces, colors, max_faces):
    """Quadric-decimate to <= max_faces, carrying per-vertex colors. The raw SLAT
    mesh has ~450k faces — far too heavy for the downstream per-frame kNN contact
    stages and the viewer. open3d preserves/interpolates vertex colors; trimesh is
    the fallback."""
    if len(faces) <= max_faces:
        return verts, faces, colors
    try:
        import open3d as o3d
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32)))
        m.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3].astype(np.float64) / 255.0)
        m = m.simplify_quadric_decimation(int(max_faces))
        v = np.asarray(m.vertices, np.float32)
        f = np.asarray(m.triangles, np.int64)
        c = (np.asarray(m.vertex_colors) * 255.0).clip(0, 255).astype(np.uint8) \
            if m.has_vertex_colors() else np.tile(np.uint8([200, 200, 200]), (len(v), 1))
        print(f"[sam3d_infer] decimated {len(faces)}->{len(f)} faces (open3d)")
        return v, f, c
    except Exception as e:
        print(f"[sam3d_infer] open3d decimation failed ({e}); trying trimesh")
    try:
        import trimesh
        m = trimesh.Trimesh(verts, faces, process=False)
        m.visual.vertex_colors = colors
        m = m.simplify_quadric_decimation(max_faces)
        c = np.asarray(m.visual.vertex_colors)[:, :3].astype(np.uint8)
        print(f"[sam3d_infer] decimated to {len(m.faces)} faces (trimesh)")
        return (np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64), c)
    except Exception as e:
        print(f"[sam3d_infer] decimation failed ({e}); keeping full-res mesh")
        return verts, faces, colors


def _vertex_colors(mesh):
    """Per-vertex RGB uint8 from whatever visual the mesh carries (texture or
    vertex colors). Falls back to light grey."""
    import trimesh
    try:
        vis = mesh.visual
        if isinstance(vis, trimesh.visual.TextureVisuals):
            vc = vis.to_color().vertex_colors
        else:
            vc = vis.vertex_colors
        vc = np.asarray(vc)
        if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices):
            return vc[:, :3].astype(np.uint8)
    except Exception as e:
        print(f"[sam3d_infer] vertex-color extraction failed ({e}); using grey")
    return np.tile(np.array([200, 200, 200], np.uint8), (len(mesh.vertices), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=os.path.join(_HERE, "checkpoints/hf/pipeline.yaml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-texture", action="store_true",
                    help="skip multiview texture baking; use raw vertex colors (faster)")
    ap.add_argument("--max-faces", type=int, default=8000,
                    help="decimate the mesh to at most this many faces (0 = keep full-res)")
    a = ap.parse_args()

    from inference import Inference, load_image
    import torch

    image = load_image(a.image)[..., :3]
    mask = _load_mask(a.mask)
    assert image.shape[:2] == mask.shape[:2], \
        f"image {image.shape[:2]} vs mask {mask.shape[:2]} size mismatch"

    inf = Inference(a.config, compile=False)
    rgba = inf.merge_mask_to_rgba(image, mask)

    bake = not a.no_texture
    out = inf._pipeline.run(
        rgba, None, a.seed,
        with_mesh_postprocess=bake,        # clean geometry only meaningful w/ texture path
        with_texture_baking=bake,
        with_layout_postprocess=False,
        use_vertex_color=not bake,
    )
    mesh = _to_trimesh(out["glb"])
    verts = np.asarray(mesh.vertices, np.float32)
    faces = np.asarray(mesh.faces, np.int64)
    colors = _vertex_colors(mesh)
    if a.max_faces and len(faces) > a.max_faces:
        verts, faces, colors = _decimate(verts, faces, colors, a.max_faces)

    # layout pose (object canonical -> camera/scene), if predicted
    def _np(x):
        return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    rot = _np(out["rotation"]).reshape(-1)[:4] if out.get("rotation") is not None else None
    transl = _np(out["translation"]).reshape(-1)[:3] if out.get("translation") is not None else None
    scale = float(_np(out["scale"]).reshape(-1)[0]) if out.get("scale") is not None else None

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, verts=verts, faces=faces, vertex_colors=colors,
             rotation=np.asarray(rot if rot is not None else [], np.float32),
             translation=np.asarray(transl if transl is not None else [], np.float32),
             scale=np.asarray([scale] if scale is not None else [], np.float32))
    try:
        mesh.export(a.out.rsplit(".", 1)[0] + ".glb")
    except Exception as e:
        print(f"[sam3d_infer] glb export skipped ({e})")

    print(f"[sam3d_infer] saved {a.out}: verts={verts.shape} faces={faces.shape} "
          f"colors={colors.shape} (texture_baked={bake})")


if __name__ == "__main__":
    main()
