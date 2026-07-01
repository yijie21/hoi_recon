"""HOI methods workbench — a method-agnostic viser viewer that plays several reconstructed
hand-object scenes in a clean, uniformly-scaled row on a shared timeline (render_and_compare
style). Each method is normalized to the same size and given a floating text label, so methods
with very different metric scales/depths line up neatly for visual comparison.

Scene schema (compare/scenes/<m>.npz), produced by compare/adapters/*:
  hand_verts [T,Nh,3] (req)  hand_faces [F,3] (opt -> mesh; else point cloud)
  obj_verts/obj_faces/obj_poses  (canonical mesh + per-frame 4x4)  OR  obj_points [T,M,3]
  contact_mask [T,Nh] (opt)  source str

Run (forehoi env has viser):
  python compare/viser_workbench.py compare/scenes/*.npz --port 8080
"""
from __future__ import annotations
import argparse, os, time
import numpy as np

SKIN = (235, 190, 160)
OBJ = (70, 130, 230)
TARGET = 0.22     # each method normalized to ~this bounding size (metres)
SPACING = 0.34    # centre-to-centre distance between methods along +x


def R_to_wxyz(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1) * 2
        return np.array([0.25*s, (m[2,1]-m[1,2])/s, (m[0,2]-m[2,0])/s, (m[1,0]-m[0,1])/s])
    if m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = np.sqrt(1+m[0,0]-m[1,1]-m[2,2]) * 2
        return np.array([(m[2,1]-m[1,2])/s, 0.25*s, (m[0,1]+m[1,0])/s, (m[0,2]+m[2,0])/s])
    if m[1,1] > m[2,2]:
        s = np.sqrt(1+m[1,1]-m[0,0]-m[2,2]) * 2
        return np.array([(m[0,2]-m[2,0])/s, (m[0,1]+m[1,0])/s, 0.25*s, (m[1,2]+m[2,1])/s])
    s = np.sqrt(1+m[2,2]-m[0,0]-m[1,1]) * 2
    return np.array([(m[1,0]-m[0,1])/s, (m[0,2]+m[2,0])/s, (m[1,2]+m[2,1])/s, 0.25*s])


def load_scene(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    for k in ("hand_faces", "hand_joints", "contact_mask", "obj_verts", "obj_faces",
              "obj_poses", "obj_points", "obj_point_colors"):
        if k not in d or np.asarray(d[k]).dtype == object:
            d[k] = None
    d["source"] = str(d["source"]) if "source" in d else os.path.basename(path)[:-4]
    d["hand_verts"] = np.asarray(d["hand_verts"], np.float64)
    d["T"] = d["hand_verts"].shape[0]
    d["obj_is_mesh"] = d.get("obj_points") is None
    return d


def _obj_world(d, t):
    if d["obj_is_mesh"]:
        P = np.asarray(d["obj_poses"])[t]
        return np.asarray(d["obj_verts"]) @ P[:3, :3].T + P[:3, 3]
    return np.asarray(d["obj_points"])[t]


def normalize(d, origin):
    """Bake a similarity transform so the method fits in a TARGET-sized box centred at `origin`.
    Rotations are preserved; only uniform scale + translation are applied."""
    T = d["T"]
    samp = [d["hand_verts"][ti] for ti in (0, T // 2, T - 1)]
    for ti in (0, T // 2, T - 1):
        ow = _obj_world(d, ti)
        samp.append(ow[:: max(1, len(ow) // 3000)])
    allp = np.vstack(samp)
    c = (allp.min(0) + allp.max(0)) / 2.0
    s = float((allp.max(0) - allp.min(0)).max()) or 1.0
    k = TARGET / s
    origin = np.asarray(origin, float)
    d["hand_verts"] = (d["hand_verts"] - c) * k + origin
    if d["obj_is_mesh"]:
        d["obj_verts"] = np.asarray(d["obj_verts"]) * k
        P = np.asarray(d["obj_poses"], float).copy()
        P[:, :3, 3] = (P[:, :3, 3] - c) * k + origin
        d["obj_poses"] = P
    else:
        d["obj_points"] = (np.asarray(d["obj_points"]) - c) * k + origin
    d["origin"] = origin


def launch(paths, port=8080, block=True):
    import viser
    scenes = [load_scene(p) for p in paths]
    for i, s in enumerate(scenes):
        normalize(s, [i * SPACING, 0.0, 0.0])
    Tmax = max(s["T"] for s in scenes)

    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("-y")
    server.scene.add_grid("/grid", width=SPACING * len(scenes), height=0.4, plane="xz",
                          cell_size=0.05, position=((len(scenes) - 1) * SPACING / 2, 0.14, 0.0))

    handles = []
    for i, s in enumerate(scenes):
        r = f"/m{i}"
        Nh = s["hand_verts"].shape[1]
        if s["hand_faces"] is not None:
            hh = server.scene.add_mesh_simple(f"{r}/hand", s["hand_verts"][0],
                    np.asarray(s["hand_faces"]).astype(np.int32), color=SKIN, side="double")
        else:
            hh = server.scene.add_point_cloud(f"{r}/hand", s["hand_verts"][0],
                    np.tile(SKIN, (Nh, 1)).astype(np.uint8), point_size=0.003)
        if s["obj_is_mesh"]:
            oh = server.scene.add_mesh_simple(f"{r}/obj", np.asarray(s["obj_verts"]),
                    np.asarray(s["obj_faces"]).astype(np.int32), color=OBJ, opacity=0.75, side="double")
        else:
            M = np.asarray(s["obj_points"]).shape[1]
            oh = server.scene.add_point_cloud(f"{r}/obj", np.asarray(s["obj_points"])[0],
                    np.tile(OBJ, (M, 1)).astype(np.uint8), point_size=0.003)
        # floating label above each method (screen-space font so it's always readable)
        server.scene.add_label(f"{r}/label", s["source"].split(" (")[0],
                               position=(i * SPACING, -0.19, 0.0),
                               font_size_mode="screen", font_screen_scale=1.4,
                               anchor="bottom-center")
        handles.append((hh, oh))

    cx = (len(scenes) - 1) * SPACING / 2
    @server.on_client_connect
    def _(client):
        client.camera.position = (cx, -0.28, -0.55)
        client.camera.look_at = (cx, 0.0, 0.02)

    gui_t = server.gui.add_slider("timeline", 0.0, 1.0, 1.0 / max(Tmax - 1, 1), 0.0)
    gui_play = server.gui.add_button("play / pause")
    gui_fps = server.gui.add_slider("fps", 1, 60, 1, 10)
    gui_info = server.gui.add_markdown("")
    state = {"playing": True}

    def render(frac):
        rows = []
        for i, s in enumerate(scenes):
            t = int(round(frac * (s["T"] - 1)))
            hh, oh = handles[i]
            if s["hand_faces"] is not None:
                try: hh.vertices = s["hand_verts"][t]
                except Exception: pass
            else:
                hh.points = s["hand_verts"][t]
            if s["obj_is_mesh"]:
                P = np.asarray(s["obj_poses"])[t]
                oh.position = tuple(P[:3, 3]); oh.wxyz = tuple(R_to_wxyz(P[:3, :3]))
            else:
                oh.points = np.asarray(s["obj_points"])[t]
            rows.append(f"**{s['source']}** — frame {t}/{s['T']-1}")
        gui_info.content = "  \n".join(rows)

    @gui_t.on_update
    def _(_): render(gui_t.value)
    @gui_play.on_click
    def _(_): state["playing"] = not state["playing"]

    render(0.0)
    print(f"workbench viser on :{port} — {len(scenes)} methods: {[s['source'] for s in scenes]}")
    if not block:
        return server, render
    try:
        while True:
            if state["playing"]:
                gui_t.value = min(gui_t.value + 1.0 / max(Tmax - 1, 1), 1.0)
                if gui_t.value >= 1.0: gui_t.value = 0.0
                render(gui_t.value)
            time.sleep(1.0 / max(1, gui_fps.value))
    except KeyboardInterrupt:
        print("bye.")


def main():
    ap = argparse.ArgumentParser("hoi-workbench-view")
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    launch(a.scenes, a.port)


if __name__ == "__main__":
    main()
