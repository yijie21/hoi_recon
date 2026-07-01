"""Interactive 4D-HOI viewer (viser) for EgoAERO runs.

Plays the reconstructed hand-object interaction over time in the browser:

  * object  -> rigid mesh, transformed per frame by its 6-DoF pose
  * hand    -> MANO mesh if ``hand_faces`` is present, else a point cloud;
               vertices in active contact this frame are turned red
  * contacts-> optional line segments from each in-contact hand vertex to its
               nearest object-surface point
  * timeline slider + play/pause + speed, and live contact / gap readouts

Data source: any EgoAERO run directory. By default it reads the final contact
stage bundle (``stage6_contact``); a run's flat ``contract/`` dir also works.

Run (needs ``pip install viser`` — present in the ``forehoi`` env)::

    python -m egoaero.viz.viser_app --run runs/demo
    python -m egoaero.viz.viser_app --run /tmp/egoaero_viz/run --stage stage5_ego_comp  # coarse vs final
    egoaero-view --run runs/demo                                                        # console script
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ..bundle import Bundle
from ..core.geometry import transform_points, knn

SKIN = np.array([235, 190, 160], np.uint8)
HIT = np.array([220, 30, 30], np.uint8)       # active contact
OBJECT_COLOR = (90, 150, 230)


def R_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return np.array([w, x, y, z])


def _hand_colors(Nh: int, contact_row: np.ndarray | None) -> np.ndarray:
    """Per-vertex colors for one frame given a boolean [Nh] active-contact mask."""
    col = np.tile(SKIN, (Nh, 1))
    if contact_row is not None and contact_row.any():
        col[contact_row.astype(bool)] = HIT
    return col


def _load(run_dir: str, stage: str) -> dict:
    """Return a unified scene dict from an EgoAERO run dir.

    Three sources, in order:
      1. a self-contained scene file (``<run>/viser_scene.npz`` or a path ending
         in ``.npz``) — used for exported reconstructions (e.g. wild6 real MANO
         hand + depth object point cloud);
      2. the stage bundle ``<run>/<stage>/`` (canonical object verts + per-frame
         pose + contact mask);
      3. the flat ``contract/`` dir.
    """
    # ---- 1) self-contained scene npz ----
    npz = run_dir if run_dir.endswith(".npz") else os.path.join(run_dir, "viser_scene.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        d = {k: z[k] for k in z.files}
        d["source"] = str(d["source"]) if "source" in d else os.path.basename(run_dir)
        for k in ("hand_joints", "hand_faces", "contact_mask", "obj_verts",
                  "obj_faces", "obj_poses", "obj_points", "obj_point_colors"):
            d.setdefault(k, None)
            if d[k] is not None and np.asarray(d[k]).dtype == object:
                d[k] = None
        return d

    bdir = os.path.join(run_dir, stage)
    if os.path.exists(os.path.join(bdir, "arrays.npz")):
        b = Bundle.load(bdir)
        return dict(
            hand_verts=b["hand_verts_t"], hand_joints=b.get("hand_joints_t"),
            obj_verts=b["obj_verts"], obj_faces=b["obj_faces"].astype(np.int32),
            obj_poses=b["obj_poses_t"], contact_mask=b.get("contact_mask"),
            hand_faces=b.get("hand_faces"),
            source=f"{os.path.basename(run_dir.rstrip('/'))}/{stage}")

    # ---- contract/ fallback ----
    c = os.path.join(run_dir, "contract")
    if not os.path.isdir(c):
        c = run_dir
    hm = np.load(os.path.join(c, "hand_mano.npz"))
    obj_poses = np.load(os.path.join(c, "object_traj.npz"))["obj_poses_t"]
    contact = np.load(os.path.join(c, "contact.npz"))["contact_mask"]
    ov, of = _read_obj(os.path.join(c, "object_mesh.obj"))
    hf = None
    hfp = os.path.join(c, "hand_faces.npy")
    if os.path.exists(hfp):
        hf = np.load(hfp).astype(np.int32)
    return dict(hand_verts=hm["verts"], hand_joints=hm.get("joints"),
                obj_verts=ov, obj_faces=of.astype(np.int32), obj_poses=obj_poses,
                contact_mask=contact, hand_faces=hf,
                source=os.path.basename(run_dir.rstrip("/")))


def _read_obj(path: str):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            F.append([int(p.split("/")[0]) - 1 for p in ln.split()[1:4]])
    return np.asarray(V, np.float64), np.asarray(F, np.int64)


def launch(run_dir: str, stage: str = "stage6_contact", port: int = 8080,
           point_size: float = 0.004, block: bool = True):
    try:
        import viser
    except ImportError:
        raise SystemExit("viser is not installed.  ->  pip install viser")

    s = _load(run_dir, stage)
    hand_verts = np.asarray(s["hand_verts"])           # [T,Nh,3] world
    contact_mask = s.get("contact_mask")               # [T,Nh] bool or None
    hand_joints = s.get("hand_joints")
    hand_faces = s.get("hand_faces")
    # object: either canonical mesh + per-frame pose, or per-frame point cloud
    obj_points = s.get("obj_points")                   # [T,M,3] or None
    obj_is_mesh = obj_points is None
    if obj_is_mesh:
        obj_verts = np.asarray(s["obj_verts"])         # [No,3] canonical
        obj_faces = np.asarray(s["obj_faces"]).astype(np.int32)
        obj_poses = np.asarray(s["obj_poses"])         # [T,4,4]
        grid_z = float(obj_poses[0][2, 3])
    else:
        obj_points = np.asarray(obj_points)
        opc = s.get("obj_point_colors")
        obj_pt_colors = (np.asarray(opc, np.uint8) if opc is not None
                         else np.tile(OBJECT_COLOR, (obj_points.shape[1], 1)).astype(np.uint8))
        grid_z = float(np.median(obj_points[0][:, 2]))
    T, Nh = hand_verts.shape[0], hand_verts.shape[1]

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")                # camera looks down +z, y is down
    server.scene.add_grid("/grid", width=1.0, height=1.0, plane="xz",
                          cell_size=0.05, position=(0.0, 0.06, grid_z))

    if hand_faces is not None:
        hand_handle = server.scene.add_mesh_simple(
            "/hand", hand_verts[0], np.asarray(hand_faces).astype(np.int32),
            color=tuple(int(c) for c in SKIN), side="double")
    else:
        hand_handle = server.scene.add_point_cloud(
            "/hand", hand_verts[0], _hand_colors(Nh, None), point_size=point_size)
    if obj_is_mesh:
        obj_handle = server.scene.add_mesh_simple(
            "/object", obj_verts, obj_faces, color=OBJECT_COLOR,
            opacity=0.65, flat_shading=False, side="double")
    else:
        obj_handle = server.scene.add_point_cloud(
            "/object", obj_points[0], obj_pt_colors, point_size=point_size)
    njoint = hand_joints.shape[1] if hand_joints is not None else 1
    joints_handle = server.scene.add_point_cloud(
        "/joints", hand_joints[0] if hand_joints is not None else np.zeros((1, 3)),
        np.tile((40, 90, 220), (njoint, 1)), point_size=point_size * 2.2, visible=False)
    lines_handle = server.scene.add_line_segments(
        "/contacts", np.zeros((1, 2, 3)), np.zeros((1, 2, 3), np.uint8), visible=False)

    gui_frame = server.gui.add_slider("frame", 0, T - 1, 1, 0)
    gui_play = server.gui.add_button("play / pause")
    gui_fps = server.gui.add_slider("speed (fps)", 1, 60, 1, 15)
    gui_obj = server.gui.add_checkbox("object", True)
    gui_hand = server.gui.add_checkbox("hand", True)
    gui_joints = server.gui.add_checkbox("joints", False)
    gui_contacts = server.gui.add_checkbox("contact lines", False)
    gui_info = server.gui.add_markdown("")
    state = {"playing": True}

    def render(t: int):
        t = int(t) % T
        if obj_is_mesh:
            P = obj_poses[t]
            obj_handle.position = tuple(P[:3, 3])
            obj_handle.wxyz = tuple(R_to_wxyz(P[:3, :3]))
            ow = transform_points(obj_verts, P)            # object surface points (world)
        else:
            obj_handle.points = obj_points[t]
            ow = obj_points[t]
        row = contact_mask[t].astype(bool) if contact_mask is not None else None
        if hand_faces is not None:
            try:
                hand_handle.vertices = hand_verts[t]
            except Exception:
                server.scene.add_mesh_simple("/hand", hand_verts[t], np.asarray(hand_faces).astype(np.int32),
                                             color=tuple(int(c) for c in SKIN), side="double")
        else:
            hand_handle.points = hand_verts[t]
            hand_handle.colors = _hand_colors(Nh, row)
        if hand_joints is not None:
            joints_handle.points = hand_joints[t]
        n_contact = int(row.sum()) if row is not None else 0
        if gui_contacts.value and n_contact:
            hc = hand_verts[t][row]
            nidx = knn(hc, ow, k=1)[1][:, 0]
            seg = np.stack([hc, ow[nidx]], axis=1)
            lines_handle.points = seg
            lines_handle.colors = np.tile(HIT, (seg.shape[0], 2, 1))
        # per-frame min hand->object surface gap
        gap_mm = float(knn(hand_verts[t], ow, k=1)[0].min() * 1000)
        status = "▶ playing" if state["playing"] else "⏸ paused"
        gui_info.content = (f"**{s['source']}**  \n{status} — **frame {t}/{T-1}**  \n"
                            f"active contacts: **{n_contact}**  \n"
                            f"min surface gap: **{gap_mm:.1f} mm**")

    @gui_frame.on_update
    def _(_): render(gui_frame.value)

    @gui_play.on_click
    def _(_):
        state["playing"] = not state["playing"]; render(gui_frame.value)

    @gui_obj.on_update
    def _(_): obj_handle.visible = gui_obj.value

    @gui_hand.on_update
    def _(_): hand_handle.visible = gui_hand.value

    @gui_joints.on_update
    def _(_): joints_handle.visible = gui_joints.value

    @gui_contacts.on_update
    def _(_):
        lines_handle.visible = gui_contacts.value; render(gui_frame.value)

    render(0)
    if not block:
        return server, render
    print(f"viser running — open the URL above (port {port}). Ctrl-C to quit.")
    try:
        while True:
            if state["playing"]:
                gui_frame.value = (gui_frame.value + 1) % T
                render(gui_frame.value)
            time.sleep(1.0 / max(1, gui_fps.value))
    except KeyboardInterrupt:
        print("\nbye.")


def main(argv=None):
    p = argparse.ArgumentParser("egoaero-view", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="EgoAERO run directory (has stage bundles or contract/)")
    p.add_argument("--stage", default="stage6_contact", help="stage bundle to view (default: final contact)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--point-size", type=float, default=0.004)
    args = p.parse_args(argv)
    launch(args.run, args.stage, args.port, args.point_size)


if __name__ == "__main__":
    main()
