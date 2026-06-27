"""Interactive 4D HOI viewer (viser).

Loads a stage bundle (default: the final stage7 reconstruction) and plays the
hand-object interaction over time in the browser:

  * object  -> rigid mesh, transformed per frame by its 6D pose
  * hand    -> point cloud (MANO mesh if `hand_faces` is present), with contact
               candidates highlighted and active contacts turned red
  * contacts-> optional line segments from each in-contact hand vertex to its
               nearest object-surface point
  * timeline slider + play/pause + speed, and live contact / gap readouts

Run:
  pip install viser
  python -m hoi_recon.viz.viser_app --run runs/demo
  python -m hoi_recon.viz.viser_app --run runs/demo --stage stage5_coarse_fit  # compare coarse vs final
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from ..bundle import Bundle
from ..geometry import transform_points, knn

SKIN = np.array([235, 190, 160], np.uint8)
CAND = np.array([245, 140, 40], np.uint8)     # contact candidate (not touching)
HIT = np.array([220, 30, 30], np.uint8)       # active contact
OBJECT_COLOR = (90, 150, 230)


def R_to_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> quaternion (w, x, y, z)."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def _hand_colors(Nh, contact_idx, contact_row):
    """Per-vertex colors for one frame given the contact map row [Nc]."""
    col = np.tile(SKIN, (Nh, 1))
    col[contact_idx] = CAND
    if contact_row is not None:
        hit = contact_idx[contact_row.astype(bool)]
        col[hit] = HIT
    return col


def launch(run_dir: str, stage: str = "stage7_contact_optim",
           port: int = 8080, point_size: float = 0.0025, block: bool = True):
    try:
        import viser
    except ImportError:
        raise SystemExit("viser is not installed.  ->  pip install viser")

    b = Bundle.load(f"{run_dir}/{stage}")
    hand_verts = b["hand_verts"]                       # [T,Nh,3]
    obj_verts = b["obj_verts"]                         # [No,3] canonical
    obj_faces = b["obj_faces"].astype(np.int32)
    obj_poses = b["obj_poses"]                         # [T,4,4]
    contact_idx = b["contact_idx"].astype(int)
    obj_colors = b.get("obj_colors")                   # [No,3] uint8 (textured backends) or None
    contact_map = b.get("contact_map")                 # [T,Nc] or None
    gaps = b.get("gaps")
    hand_joints = b.get("hand_joints")
    hand_faces = b.get("hand_faces")                   # present for real MANO
    T, Nh = hand_verts.shape[0], hand_verts.shape[1]

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")                # camera looks down +z, y is down
    server.scene.add_grid("/grid", width=1.0, height=1.0, plane="xz",
                          cell_size=0.05, position=(0.0, 0.06, 0.6))

    # --- static handles, updated per frame ---
    if obj_colors is not None:
        # textured object (e.g. SAM-3D): render real per-vertex colors
        import trimesh
        _om = trimesh.Trimesh(vertices=np.asarray(obj_verts), faces=obj_faces,
                              process=False)
        _om.visual.vertex_colors = np.asarray(obj_colors, np.uint8)
        obj_handle = server.scene.add_mesh_trimesh("/object", _om)
    else:
        obj_handle = server.scene.add_mesh_simple(
            "/object", obj_verts, obj_faces, color=OBJECT_COLOR,
            opacity=0.65, flat_shading=False, side="double")
    if hand_faces is not None:
        hand_handle = server.scene.add_mesh_simple(
            "/hand", hand_verts[0], hand_faces.astype(np.int32),
            color=tuple(int(c) for c in SKIN), side="double")
    else:
        hand_handle = server.scene.add_point_cloud(
            "/hand", hand_verts[0], _hand_colors(Nh, contact_idx, None),
            point_size=point_size)
    joints_handle = server.scene.add_point_cloud(
        "/joints", hand_joints[0] if hand_joints is not None else np.zeros((1, 3)),
        np.tile((40, 90, 220), ((hand_joints.shape[1] if hand_joints is not None else 1), 1)),
        point_size=point_size * 2.5, visible=False)
    lines_handle = server.scene.add_line_segments(
        "/contacts", np.zeros((1, 2, 3)), np.zeros((1, 2, 3), np.uint8), visible=False)

    # --- GUI ---
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
        P = obj_poses[t]
        obj_handle.position = tuple(P[:3, 3])
        obj_handle.wxyz = tuple(R_to_wxyz(P[:3, :3]))
        row = contact_map[t] if contact_map is not None else None
        if hand_faces is not None:
            # MeshHandle vertex updates aren't supported on all viser versions;
            # fall back to re-adding the mesh node if direct assignment fails.
            try:
                hand_handle.vertices = hand_verts[t]
            except Exception:
                server.scene.add_mesh_simple(
                    "/hand", hand_verts[t], hand_faces.astype(np.int32),
                    color=tuple(int(c) for c in SKIN), side="double")
        else:
            hand_handle.points = hand_verts[t]
            hand_handle.colors = _hand_colors(Nh, contact_idx, row)
        if hand_joints is not None:
            joints_handle.points = hand_joints[t]
        # contact line segments: in-contact hand vert -> nearest object surface pt
        n_contact = int(row.sum()) if row is not None else 0
        if gui_contacts.value and n_contact:
            ow = transform_points(obj_verts, P)
            hc = hand_verts[t][contact_idx[row.astype(bool)]]
            nidx = knn(hc, ow, k=1)[1][:, 0]
            seg = np.stack([hc, ow[nidx]], axis=1)
            lines_handle.points = seg
            lines_handle.colors = np.tile(HIT, (seg.shape[0], 2, 1))
        gap_mm = float(gaps[t] * 1000) if gaps is not None else float("nan")
        status = "▶ playing" if state["playing"] else "⏸ paused"
        gui_info.content = (f"{status} — **frame {t}/{T-1}**  \n"
                            f"active contacts: **{n_contact}**  \n"
                            f"min surface gap: **{gap_mm:.1f} mm**")

    @gui_frame.on_update
    def _(_):
        render(gui_frame.value)

    @gui_play.on_click
    def _(_):
        state["playing"] = not state["playing"]
        render(gui_frame.value)

    @gui_obj.on_update
    def _(_): obj_handle.visible = gui_obj.value

    @gui_hand.on_update
    def _(_): hand_handle.visible = gui_hand.value

    @gui_joints.on_update
    def _(_): joints_handle.visible = gui_joints.value

    @gui_contacts.on_update
    def _(_):
        lines_handle.visible = gui_contacts.value
        render(gui_frame.value)

    render(0)
    if not block:
        return server, render, gui_contacts
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
    p = argparse.ArgumentParser("hoi-recon-view", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory (e.g. runs/demo)")
    p.add_argument("--stage", default="stage7_contact_optim",
                   help="which stage bundle to view (default: final)")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--point-size", type=float, default=0.0025)
    args = p.parse_args(argv)
    launch(args.run, args.stage, args.port, args.point_size)


if __name__ == "__main__":
    main()
