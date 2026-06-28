"""EgoDex-R per-sequence record (App F): assemble raw observations, reconstructed
hand-object states, quality diagnostics, and task/difficulty metadata into one
sequence directory, with a writer / validator / reader."""
from __future__ import annotations
import json, os, shutil
import numpy as np

_CONTRACT_FILES = ["hand_mano.npz", "object_traj.npz", "object_mesh.obj", "contact.npz"]
_META_KEYS = {"task_description", "manipulated_object", "relational_objects",
              "difficulty", "decision", "frames", "seq_id"}


def _load_npz(path):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def write_sequence(dataset_dir, seq_id, run_dir, metadata):
    d = os.path.join(dataset_dir, seq_id)
    os.makedirs(d, exist_ok=True)
    contract_dir = os.path.join(run_dir, "contract")
    for fn in _CONTRACT_FILES:
        shutil.copyfile(os.path.join(contract_dir, fn), os.path.join(d, fn))
    # quality diagnostics
    shutil.copyfile(os.path.join(run_dir, "quality.json"), os.path.join(d, "quality.json"))
    # raw observations from the stage0 bundle
    s0 = _load_npz(os.path.join(run_dir, "stage0_ego_io", "arrays.npz"))
    with open(os.path.join(run_dir, "stage0_ego_io", "meta.json")) as f:
        s0_meta = json.load(f)
    fps = float(s0_meta.get("fps", 30.0)); T = int(s0_meta.get("T", s0["depth"].shape[0]))
    np.savez(os.path.join(d, "raw_obs.npz"),
             depth=s0["depth"], obj_mask=s0["obj_mask"], hand_mask=s0["hand_mask"],
             intrinsics=s0["intrinsics"], cam_traj=s0["cam_traj"],
             timestamps=np.arange(T) / fps)
    with open(os.path.join(d, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    manifest = {"hand_mano": "hand_mano.npz", "object_traj": "object_traj.npz",
                "object_mesh": "object_mesh.obj", "contact": "contact.npz",
                "quality": "quality.json", "raw_obs": "raw_obs.npz",
                "metadata": "metadata.json", "frames": int(metadata.get("frames", T))}
    with open(os.path.join(d, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def validate_sequence(dataset_dir, seq_id) -> bool:
    d = os.path.join(dataset_dir, seq_id)
    need = _CONTRACT_FILES + ["quality.json", "raw_obs.npz", "metadata.json", "manifest.json"]
    if not all(os.path.exists(os.path.join(d, n)) for n in need):
        return False
    try:
        with open(os.path.join(d, "metadata.json")) as f:
            meta = json.load(f)
    except Exception:
        return False
    return _META_KEYS.issubset(meta.keys())


def read_metadata(dataset_dir, seq_id) -> dict:
    with open(os.path.join(dataset_dir, seq_id, "metadata.json")) as f:
        return json.load(f)
