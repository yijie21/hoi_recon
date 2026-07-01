# HOI methods workbench — collect & compare

All HOI-reconstruction methods live as sibling subfolders under `hoi_recon/`, each runnable
on its own. This `compare/` folder provides a **method-agnostic viser viewer** + per-method
**adapters** so you can view any/all of them on one timeline, in the same style as
`render_and_compare`'s viewer.

## Methods (subfolders)

| Method | Subfolder | Status | Env | Real perception? |
|---|---|---|---|---|
| render_and_compare (CHOIR) | `render_and_compare/` | ✅ mock runs; real = heavy 2-env backends | `forehoi` (numpy mock) | optimization, backends wired |
| egoaero (EgoAERO) | `egoaero/` | ✅ mock only (real perception stubbed) | base / `forehoi` | no (GT-driven mock) |
| **ForeHOI** | `forehoi/` | ✅ **fully runnable** (feed-forward video) | `forehoi` | **yes** |
| **HORT** | `hort/` | ✅ **fully runnable** (feed-forward, 1 image) | `hort` | **yes** |
| HOLD | `hold/` | ⚠️ env ready; real recon **blocked** (account-walled data) | `hold` | yes (when data unblocked) |
| EasyHOI | `easyhoi/` | ⚠️ hand + object shape run; **fusion/alignment blocked** (LISA + afford-diffusion envs) | `easyhoi` (+`instantmesh`) | yes (hand + shape) |

Each subfolder has a `RUN.md` with the exact, verified run command + output format.

## The common scene format

Adapters convert each method's native output to a self-contained `compare/scenes/<m>.npz`:

```
hand_verts   [T,Nh,3]   required, metres, OpenCV camera frame
hand_faces   [F,3]      optional -> MANO mesh; absent -> point cloud
obj_verts/obj_faces/obj_poses   canonical mesh + per-frame 4x4 pose   (mesh objects)
  OR obj_points [T,M,3] (+ obj_point_colors)                          (point-cloud objects)
contact_mask [T,Nh]     optional -> in-contact verts turn red
source       str        label
```

## View it (interactive, like render_and_compare)

```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python    # has viser
# all methods side by side on one timeline:
$PY compare/viser_workbench.py compare/scenes/*.npz --port 8080
# pick a subset:
$PY compare/viser_workbench.py compare/scenes/hort.npz compare/scenes/forehoi.npz
```
Reach it from your laptop: `ssh -p $VAST_TCP_PORT_22 -L 8080:127.0.0.1:8080 root@$PUBLIC_IPADDR` → http://localhost:8080

Static preview (no server): `$PY compare/snapshot.py compare/out.png compare/scenes/*.npz`
→ see `compare/compare_all.png`.

ForeHOI also keeps its own native viewer: `forehoi/viser_4d_demo.py output_4d/wild1`.

## Rebuild the scenes (adapters)

```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python
$PY compare/adapters/hort_to_scene.py     hort/out_demo/f0030                    compare/scenes/hort.npz
$PY compare/adapters/forehoi_to_scene.py  forehoi/output_4d/wild1                compare/scenes/forehoi.npz
$PY compare/adapters/rc_to_scene.py       render_and_compare/runs/demo/stage8_eval/pseudo_gt.npz compare/scenes/render_and_compare.npz
$PY compare/adapters/egoaero_to_scene.py  /tmp/egoaero_viz/run                   compare/scenes/egoaero.npz
$PY compare/adapters/easyhoi_to_scene.py  easyhoi/data_run                       compare/scenes/easyhoi.npz
```

## Same-clip comparison (wild6.mp4) ✅ — all 4 methods REAL

All four methods reconstruct the **same** clip (`egoaero/assets/wild6.mp4`, a hand grasping a
~15 cm bottle) with real perception → directly comparable:

| Scene | Method on wild6 | Object output | Object world size |
|---|---|---|---|
| `render_and_compare_wild6.npz` | render_and_compare (CHOIR real: MoGe+WiLoR+SAM2+HaMeR+SAM-3D, 74f) | textured mesh, but **~5× oversized** | **0.813 m** ⚠️ |
| `hort_wild6.npz`    | HORT, per-frame on 33 frames | sparse point cloud (no temporal consistency) | 0.154 m |
| `easyhoi.npz`       | EasyHOI, frame f0030         | shape mesh, **pose not aligned** | 0.120 m |
| `forehoi_wild6.npz` | ForeHOI, 12 frames           | **dense coherent bottle mesh (186K verts)** | 0.251 m |

```bash
PY=/workspace/miniconda3/envs/forehoi/bin/python
$PY compare/viser_workbench.py compare/scenes/render_and_compare_wild6.npz \
    compare/scenes/hort_wild6.npz compare/scenes/easyhoi.npz compare/scenes/forehoi_wild6.npz
$PY compare/snapshot.py compare/compare_wild6_4way.png compare/scenes/render_and_compare_wild6.npz \
    compare/scenes/hort_wild6.npz compare/scenes/easyhoi.npz compare/scenes/forehoi_wild6.npz
```
→ `compare/compare_wild6_4way.png`.

**Takeaways (real bottle ≈ 0.15 m), after the stage-1 fix below:**
- **render_and_compare (robust, fixed)** — bottle-sized object (0.188 m) held firmly in the grasp; hand good, contact tight. Now competitive/best.
- **ForeHOI** — cleanest, most complete object mesh (0.251 m); strong all-round.
- **HORT / EasyHOI** — object size close to true, but HORT's is a noisy per-frame cloud and EasyHOI's isn't pose-aligned.
- **render_and_compare (combined/differentiable)** — better hand keypoint reproj (kp2d 4.9 px) but its differentiable object-pose **drifts off the hand** at some frames; not strictly better than the robust path here.

### The object-scale bug (found + fixed)
The first same-clip run exposed a real failure: render_and_compare's object came out **0.813 m (~5×)**.
Root cause was **not** the scaling formula but **stage 1**: `_object_prompt` returned the *hand-box
centre* as SAM2's object prompt, so for a hand *holding* an object it segmented the **hand** (all 74
masks covered 55–73 % of the frame). SAM-3D then reconstructed the hand-blob. Fix: a proper
`--object-prompt X Y` override (the "user click" the code's own docstring recommends), wired
CLI→config→stage 1. Re-running with `--object-prompt 502 482` (the bottle, from YOLO on frame 0):
object mask → 8.5 % of frame, object size → **0.188 m**. `render_and_compare_wild6_before.npz`
keeps the broken result for comparison. Code changes: `hoi_recon/{cli.py,stages/stage1_detect_track.py,
backends/real_perception.py}`; re-run recipe in `render_and_compare/RUN_REAL.md`.

Scenes: `render_and_compare_wild6.npz` (robust, fixed), `render_and_compare_wild6_combined.npz`
(differentiable), `render_and_compare_wild6_before.npz` (pre-fix). Final 5-panel: `compare/compare_wild6_final.png`.

render_and_compare real run: `render_and_compare/runs/wild6_real/` (robust) and `runs/wild6_combined/`
(full differentiable path); setup in `render_and_compare/RUN_REAL.md`. To rebuild its scene:
`$PY compare/adapters/rc_to_scene.py render_and_compare/runs/wild6_real/stage7_contact_optim/arrays.npz compare/scenes/render_and_compare_wild6.npz`

How wild6 was produced for each (under shared-GPU contention, ForeHOI needs a ~19GB window):
```bash
# HORT: per-frame on wild6 frames -> temporal scene
conda activate hort && CUDA_VISIBLE_DEVICES=0 python hort/demo.py --img_folder <wild6 frames> --out_folder hort/out_wild6
$PY compare/adapters/hort_seq_to_scene.py hort/out_wild6 compare/scenes/hort_wild6.npz
# ForeHOI: on the video (obj/hand pixel prompts on frame 0)
cd forehoi && CUDA_VISIBLE_DEVICES=<free gpu> python export_4d_hoi.py --video .../wild6.mp4 \
    --obj-points 502,483 502,387 --hand-points 90,607 --frames 12 --name wild6
$PY compare/adapters/forehoi_to_scene.py forehoi/output_4d/wild6 compare/scenes/forehoi_wild6.npz
```

## Caveats for a *fair* comparison

The scenes above are each method's **native demo on a different clip** (HORT→wild6 bottle,
ForeHOI→wild1, the two mock methods→synthetic sphere). For a true head-to-head, run the
real-perception methods on the **same** clip (e.g. HORT per-frame + ForeHOI on `wild6.mp4`),
and run `render_and_compare`'s real backends instead of mock. That common-clip run is the
natural next step; the infra here already supports it (just point the adapters at same-clip outputs).
