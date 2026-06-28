"""Build the MuJoCo manipulation task from an SP1 reconstruction run.

Shadow Hand mounted via a mocap-driven wrist (rh_forearm welded to a 'wrist_target'
mocap body) + the reconstructed object as a free body on a table plane.

The mocap wrist replaces a floating 6-DoF base: at policy rollout time the caller
sets ``data.mocap_pos[task.wrist_mocap_id]`` and ``data.mocap_quat[task.wrist_mocap_id]``
each step to follow the reference wrist trajectory; the equality constraint then
kinematically pins rh_forearm to that mocap body.  This lets the 20 hand actuators
(18 finger + 2 wrist) drive finger posture while the whole hand tracks the
reconstructed wrist path.

The composed model exposes:
  - ``task.fingertip_body_ids``  — dict{finger -> body_id} on the composed model
  - ``task.finger_act_ids``       — actuator indices with no 'WRJ' in their name (18)
  - ``task.wrist_mocap_id``       — mocap index of 'wrist_target'
  - ``task.obj_qadr``             — qpos address of joint 'obj_free'
"""
from __future__ import annotations

import json
import os

import numpy as np

_TIP_JOINTS = [4, 8, 12, 16, 20]   # MANO joint indices for the 5 fingertips

_FINGERTIP_BODY_NAMES: dict[str, str] = {
    "thumb":  "rh_thdistal",
    "index":  "rh_ffdistal",
    "middle": "rh_mfdistal",
    "ring":   "rh_rfdistal",
    "little": "rh_lfdistal",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_npz(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _read_obj_verts(path: str) -> np.ndarray:
    """Return (N,3) vertex positions from a Wavefront OBJ file."""
    vs = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    return np.asarray(vs, dtype=float)


# ---------------------------------------------------------------------------
# load_reference — pure numpy; no mujoco dependency
# ---------------------------------------------------------------------------

def load_reference(run_dir: str) -> dict:
    """Load the SP1 contract into a reference dict.

    Keys
    ----
    wrist_pos      : (T, 3)      wrist/root position in metres
    fingertips_h   : (T, 5, 3)  MANO fingertip positions [thumb,index,middle,ring,little]
    obj_pos        : (T, 3)      object centroid position
    obj_R          : (T, 3, 3)  object rotation matrices
    contact_active : list[list[int]]  per-frame active finger indices (sphere-proxy heuristic)
    mesh_path      : str         absolute path to object_mesh.obj
    T              : int         number of frames
    """
    d = os.path.join(run_dir, "contract")
    with open(os.path.join(d, "manifest.json")) as _mf:
        man = json.load(_mf)

    hand = _load_npz(os.path.join(d, man["hand_mano"]))
    obj = _load_npz(os.path.join(d, man["object_traj"]))

    joints = hand["joints"]       # (T, 21, 3)
    poses = obj["obj_poses_t"]    # (T, 4, 4)
    T = int(joints.shape[0])

    wrist_pos = joints[:, 0, :]
    fingertips_h = joints[:, _TIP_JOINTS, :]   # (T, 5, 3)
    obj_pos = poses[:, :3, 3]
    obj_R = poses[:, :3, :3]

    # contact_active: fingertips within ~20 mm of the object surface (sphere proxy).
    # The contract carries no per-finger vertex groups; we approximate the surface
    # as a sphere of radius = max distance from centroid.  See ASSUMPTIONS.md.
    mesh_path = os.path.join(d, man["object_mesh"])
    verts = _read_obj_verts(mesh_path)
    centroid = verts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(verts - centroid, axis=1)))
    thresh = 0.02   # 20 mm band around the sphere surface

    contact_active: list[list[int]] = []
    for t in range(T):
        c = obj_pos[t]
        active = [
            i for i in range(5)
            if abs(np.linalg.norm(fingertips_h[t, i] - c) - radius) < thresh
        ]
        contact_active.append(active)

    return {
        "wrist_pos":      wrist_pos,
        "fingertips_h":   fingertips_h,
        "obj_pos":        obj_pos,
        "obj_R":          obj_R,
        "contact_active": contact_active,
        "mesh_path":      os.path.abspath(mesh_path),
        "T":              T,
    }


# ---------------------------------------------------------------------------
# Task — wraps the compiled MjModel with convenience look-ups
# ---------------------------------------------------------------------------

class Task:
    """Composed MuJoCo task: Shadow Hand + mocap wrist + free object.

    Attributes
    ----------
    model              : mujoco.MjModel  — the compiled scene
    ref                : dict            — reference trajectory (from load_reference)
    cfg                : dict            — policy configuration
    fingertip_body_ids : dict[str, int]  — {finger -> body_id} on the composed model
    finger_act_ids     : list[int]       — actuator indices with no 'WRJ' in their name
    wrist_mocap_id     : int             — model.body_mocapid[wrist_target]
    obj_qadr           : int             — model.jnt_qposadr[obj_free]
    """

    def __init__(self, model, ref: dict, cfg) -> None:
        import mujoco  # lazy: mujoco is optional at module-import time

        self.model = model
        self.ref = ref
        self.cfg = cfg

        # Fingertip body IDs resolved on the composed model
        self.fingertip_body_ids: dict[str, int] = {}
        for finger, body_name in _FINGERTIP_BODY_NAMES.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if bid < 0:
                raise RuntimeError(
                    f"Fingertip body '{body_name}' (finger '{finger}') "
                    "not found in composed model."
                )
            self.fingertip_body_ids[finger] = int(bid)

        # Finger actuator indices: all actuators whose name lacks 'WRJ'
        self.finger_act_ids: list[int] = [
            i for i in range(model.nu)
            if "WRJ" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
        ]

        # Mocap index of 'wrist_target'
        wrist_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_target")
        if wrist_bid < 0:
            raise RuntimeError("Body 'wrist_target' not found in composed model.")
        self.wrist_mocap_id: int = int(model.body_mocapid[wrist_bid])

        # qpos address of the object free joint
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        if jid < 0:
            raise RuntimeError("Joint 'obj_free' not found in composed model.")
        self.obj_qadr: int = int(model.jnt_qposadr[jid])

    @staticmethod
    def from_run(run_dir: str, hand_xml: str, cfg) -> "Task":
        """Convenience constructor — delegates to build_task."""
        return build_task(run_dir, hand_xml, cfg)


# ---------------------------------------------------------------------------
# build_task — scene construction via MjSpec
# ---------------------------------------------------------------------------

def build_task(run_dir: str, hand_xml: str, cfg) -> Task:
    """Build the MuJoCo manipulation task from an SP1 reconstruction run.

    Scene layout
    ------------
    1. Shadow Hand loaded from *hand_xml* (meshdir resolved by MjSpec relative to
       the XML's own directory).
    2. A mocap body ``wrist_target`` added to the worldbody; a WELD equality
       constrains ``rh_forearm`` to follow it.  Moving
       ``data.mocap_pos / mocap_quat`` at rollout time drives the whole hand.
    3. A table ground plane (``mjGEOM_PLANE``).
    4. The reconstructed object as a free body (``obj_free`` joint) with its
       mesh (``obj_mesh``), placed at the first reference position.

    Parameters
    ----------
    run_dir  : path to a completed SP1 run (must contain contract/ artefacts)
    hand_xml : path to right_hand.xml (Shadow Hand MJCF)
    cfg      : policy config dict (e.g. load_config()["quality"])

    Returns
    -------
    Task  — compiled model + reference + resolved IDs
    """
    import mujoco  # lazy

    ref = load_reference(run_dir)

    # Load the hand spec; MjSpec.from_file resolves meshdir relative to the XML.
    spec = mujoco.MjSpec.from_file(os.path.abspath(hand_xml))
    wb = spec.worldbody

    # --- Table ground plane ---
    wb.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.0, 1.0, 0.05],
        pos=[0.0, 0.0, 0.0],
    )

    # --- Mocap wrist target body ---
    wb.add_body(
        name="wrist_target",
        mocap=1,
        pos=list(map(float, ref["wrist_pos"][0])),
    )

    # --- Weld the forearm to the mocap body ---
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name1="wrist_target",
        name2="rh_forearm",
        objtype=mujoco.mjtObj.mjOBJ_BODY,
    )

    # --- Reconstructed object mesh + free body ---
    spec.add_mesh(name="obj_mesh", file=os.path.abspath(ref["mesh_path"]))
    # Clamp object z above table plane: ensure clearance by object radius
    obj_pos0 = list(map(float, ref["obj_pos"][0]))
    verts = _read_obj_verts(ref["mesh_path"])
    centroid = verts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(verts - centroid, axis=1)))
    obj_pos0[2] = max(obj_pos0[2], radius + 0.005)
    obj_body = wb.add_body(
        name="object",
        pos=obj_pos0,
    )
    obj_body.add_freejoint(name="obj_free")
    obj_body.add_geom(
        name="obj_geom",
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname="obj_mesh",
        density=200,
    )

    model = spec.compile()
    return Task(model, ref, cfg)
