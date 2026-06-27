"""Workbench method-contract writer: emit the comparable per-clip output
(MANO hand, object mesh, object 6-DoF trajectory, contact maps) under <run>/contract/."""
from __future__ import annotations
import json
import os

import numpy as np


def _write_obj(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a triangulated OBJ with 1-indexed faces."""
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for tri in faces:
            # OBJ spec: 1-indexed; faces from stage6 are already 0-indexed int arrays
            f.write(f"f {int(tri[0]) + 1} {int(tri[1]) + 1} {int(tri[2]) + 1}\n")


def write(ctx) -> dict:
    """Write contract artefacts to <run>/contract/ and return the manifest dict."""
    s6 = ctx.load("stage6_contact")
    d = os.path.join(ctx.run_dir, "contract")
    os.makedirs(d, exist_ok=True)

    np.savez(
        os.path.join(d, "hand_mano.npz"),
        verts=s6["hand_verts_t"],
        joints=s6["hand_joints_t"],
    )
    np.savez(os.path.join(d, "object_traj.npz"), poses=s6["obj_poses_t"])
    np.savez(os.path.join(d, "contact.npz"), mask=s6["contact_mask"])
    _write_obj(
        os.path.join(d, "object_mesh.obj"),
        s6["obj_verts"],
        s6["obj_faces"].astype(int),
    )

    manifest: dict = {
        "hand_mano": "hand_mano.npz",
        "object_mesh": "object_mesh.obj",
        "object_traj": "object_traj.npz",
        "contact": "contact.npz",
        "frames": int(s6["hand_verts_t"].shape[0]),
    }
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def validate(run_dir: str) -> bool:
    """Return True iff all required contract files exist under <run_dir>/contract/."""
    d = os.path.join(run_dir, "contract")
    needed = [
        "hand_mano.npz",
        "object_mesh.obj",
        "object_traj.npz",
        "contact.npz",
        "manifest.json",
    ]
    return all(os.path.exists(os.path.join(d, n)) for n in needed)
