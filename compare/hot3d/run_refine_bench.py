"""Tier-2 driver: run the trained hoi_flow bridge refiner behind the REAL coarse stack on
the 6 HOT3D benchmark clips, score object + hand before/after against the mocap GT, and
write standard `render_and_compare/runs/hot3d_<clip>_flowref/` run dirs.

This is the benchmark counterpart of `hoi_flow/sample.py` (which refines one training
segment). It differs from sample.py in three benchmark-specific ways:

1. **Object canonical gauge fix (the load-bearing step).** The coarse `obj_poses` pose the
   *run's own SAM-3D mesh* (SAM canonical), but the refiner's object state block AND its
   `obj_points` conditioning both live in the *GT eval-GLB canonical* (that is what training
   saw). So per clip we solve one constant SE3 gauge `A` by MESH-TO-MESH registration only —
   24 octahedral-rotation ICP inits between surface samples of the SAM mesh and the GT GLB,
   lowest-cost wins — and remap the coarse poses into GT canonical as `M_gt = M_coarse @ A`.
   GT poses are NEVER used to fit `A` (that would leak GT into the input). We verify with a
   frame-0 point-cloud chamfer between the posed SAM mesh and the posed GT mesh and report it
   (expect <~15 mm; symmetric objects are ambiguous about their symmetry axis — acceptable,
   scoring is chamfer-based). The refined GT-canonical poses then pose the GT GLB mesh, which
   is what `gt_pose_eval_hot3d.py` scores.

2. **Conditioning modalities = rgb + depth + obj_points + K. seg_mask is DROPPED.** The
   seg_mask channel's target class is GT-derived localization, i.e. it is leakage at test
   time; depth (ray-cast from the posed CAD, the benchmark's standard RGB-D substrate that
   the coarse stack already consumes) carries the same evidence without leaking. The
   conditioning encoders skip any modality absent from the batch (train-time modality
   dropout), so omitting seg_mask needs no model change.

3. **Hand into state** reuses sample.py's real-coarse hand builder (`_CoarseHand`): the
   shipped hand-reprojection optimizer's MANO params (mano_global_aa/mano_pose_aa[T,45]/
   mano_transl/mano_side, camera frame) are FK'd, mapped through `wrist_offset.json` into the
   UmeTrack-wrist state convention, and the full pose is projected to PCA-15. `visible=False`
   frames -> invalid. presence is taken from the coarse hand's own validity (honest test-time
   presence), NOT from GT hands_present.

Outputs per clip (under render_and_compare/runs/hot3d_<clip>_flowref/stage8_eval/):
  pseudo_gt.npz        {obj_verts, obj_faces = GT GLB mesh; obj_poses = refined GT-canon [T,4,4]}
  hand_refined.npz     {hand_wrist [T,2,4,4], hand_thetas [T,2,15]}
  coarse_state_debug.npz  the pre-refine state (obj_pose/hand_wrist/hand_thetas) for diffing

CLI:  run_refine_bench.py --ckpt PATH [--clips all|comma-list] [--n_steps 8]
                          [--out_suffix flowref] [--device cuda:0]
"""
import argparse
import json
import os
import subprocess
import sys

# keep GPU1 (training) untouched — only GPU0 is ever visible to this process
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

HOT3D_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HOT3D_DIR))
sys.path.insert(0, REPO)

from hoi_flow.state import pack_state, unpack_state, Normalizer          # noqa: E402
from hoi_flow.bridge import sample as bridge_sample                      # noqa: E402
from hoi_flow.geometry import decompose_along_ray                        # noqa: E402
from hoi_flow.train import load_config, make_cond_fn, state_metrics, _COLS  # noqa: E402
from hoi_flow.sample import load_model_enc, _CoarseHand                  # noqa: E402

DS = "/workspace/datasets/hot3d"
RUNS = os.path.join(REPO, "render_and_compare/runs")
PY = sys.executable

# clip key -> (source-clip id, rc_input dir, object coarse run, hand coarse run)
CLIPS = {
    "bottle_bbq_002034":    ("clip-002034", "rc_input_002034_bottle_bbq",
                             "hot3d_bottle_bbq_002034_fpauto", "hot3d_bottle_bbq_002034_icpjgr"),
    "mug_white_001970":     ("clip-001970", "rc_input_001970_mug_white",
                             "hot3d_mug_white_001970_fpauto", "hot3d_mug_white_001970_icpjgr"),
    "vase_002500":          ("clip-002500", "rc_input_002500_vase",
                             "hot3d_vase_002500_fpauto", "hot3d_vase_002500_icpjgr"),
    "spatula_red_001990":   ("clip-001990", "rc_input_001990_spatula_red",
                             "hot3d_spatula_red_001990_fpauto", "hot3d_spatula_red_001990_icpjgr"),
    "puzzle_toy_001964":    ("clip-001964", "rc_input_001964_puzzle_toy",
                             "hot3d_puzzle_toy_001964_fpauto", "hot3d_puzzle_toy_001964_icpjgr"),
    "potato_masher_002349": ("clip-002349", "rc_input_002349_potato_masher",
                             "hot3d_potato_masher_002349_icpjgr", "hot3d_potato_masher_002349_icpjgr"),
}
_PER_FRAME = ("rgb", "depth", "seg_mask")


# ------------------------------------------------------------------- object canonical gauge
def _sample_mesh(mesh, n, seed=0):
    return np.asarray(trimesh.sample.sample_surface(mesh, n, seed=seed)[0])


def _chamfer_med_mm(a, b):
    """symmetric median chamfer (mm) between two point clouds in meters."""
    d1 = cKDTree(b).query(a, workers=-1)[0]
    d2 = cKDTree(a).query(b, workers=-1)[0]
    return (np.median(d1) + np.median(d2)) / 2 * 1000.0


def solve_gauge(sam_mesh, gt_mesh, coarse_poses, n_pts=2000):
    """Mesh-to-mesh SE3 gauge `A` (GT-canonical -> SAM-canonical) by multi-start ICP over the
    24 octahedral rotation inits; lowest cost wins. Returns (A, mapped_poses, frame0_chamfer_mm,
    side) where mapped_poses[t] = coarse_poses[t] @ A poses the GT mesh (verified empirically to
    overlay the posed SAM mesh at frame 0; falls back to the inverse composition if that scores
    lower). GT object poses are never used here — the gauge is shape-only."""
    Vgt = _sample_mesh(gt_mesh, n_pts, seed=0)
    Vsam = _sample_mesh(sam_mesh, n_pts, seed=0)
    mu_gt, mu_sam = Vgt.mean(0), Vsam.mean(0)
    best = (np.inf, None)
    for R0 in Rotation.create_group("O").as_matrix():
        init = np.eye(4)
        init[:3, :3] = R0
        init[:3, 3] = mu_sam - R0 @ mu_gt                 # centroid-align the init
        try:
            M, _, cost = trimesh.registration.icp(Vgt, Vsam, initial=init,
                                                  max_iterations=40, scale=False)
        except Exception:
            continue
        if cost < best[0]:
            best = (cost, M)
    A = best[1] if best[1] is not None else np.eye(4)     # GT-canon -> SAM-canon

    # empirical composition side: after mapping, posed GT mesh must overlay posed SAM mesh @ f0
    M0 = coarse_poses[0]
    Xsam0 = Vsam @ M0[:3, :3].T + M0[:3, 3]
    cand = {"right": A, "left": np.linalg.inv(A)}         # M_gt = M_c @ A   vs   M_c @ A^-1
    scored = {}
    for name, G in cand.items():
        Mg0 = M0 @ G
        Xgt0 = Vgt @ Mg0[:3, :3].T + Mg0[:3, 3]
        scored[name] = _chamfer_med_mm(Xsam0, Xgt0)
    side = min(scored, key=scored.get)
    G = cand[side]
    mapped = np.einsum("tij,jk->tik", coarse_poses, G).astype(np.float64)
    return G, mapped, scored[side], side


# ------------------------------------------------------------------- coarse hand -> state
def _hand_adapter(out, T):
    """hand_reproj_opt/out.npz -> the key names _CoarseHand.to_state expects, with invisible
    frames set NaN so they are marked invalid."""
    vis = out["visible"].astype(bool)[:T]
    g = out["mano_global_aa"].astype(np.float64)[:T].copy()
    p = out["mano_pose_aa"].astype(np.float64)[:T].copy()
    tr = out["mano_transl"].astype(np.float64)[:T].copy()
    side = out["mano_side"].astype(np.float64)[:T].copy()
    inv = ~vis
    g[inv] = np.nan; p[inv] = np.nan; tr[inv] = np.nan; side[inv] = np.nan
    return {"hand_mano_global": g, "hand_mano_pose": p,
            "hand_mano_transl": tr, "hand_mano_side": side}


def _state_hand_verts(ch, wrist, thetas15, s):
    """Inverse of _CoarseHand.to_state: state (wrist 4x4, thetas PCA-15, side slot s) ->
    camera-frame MANO verts [778,3]. Used only for the hand-error metric (validated against the
    coarse optimizer's own hand_verts)."""
    mx = np.eye(3) if s == 1 else np.diag([-1.0, 1.0, 1.0])
    off = ch.offsets["right" if s == 1 else "left"]
    M_root = wrist @ np.linalg.inv(off)
    R_root, root_cam = M_root[:3, :3], M_root[:3, 3]
    Rg = mx @ R_root @ mx                                  # invert R_root = mx@Rg@mx
    g = Rotation.from_matrix(Rg).as_rotvec()
    full_p = (thetas15.astype(np.float64) @ ch.comp15 + ch.hands_mean).astype(np.float32)
    hp = torch.tensor(full_p[None])
    j0 = ch.layer(betas=torch.zeros(1, 10), global_orient=torch.zeros(1, 3),
                  hand_pose=hp, transl=torch.zeros(1, 3),
                  return_verts=True).joints[0, 0].detach().numpy()
    tr = root_cam - mx @ j0
    v = ch.layer(betas=torch.zeros(1, 10),
                 global_orient=torch.tensor(g[None], dtype=torch.float32),
                 hand_pose=hp, transl=torch.zeros(1, 3),
                 return_verts=True).vertices[0].detach().numpy()
    return (mx @ v.T).T + tr


# ------------------------------------------------------------------- conditioning (no seg)
def build_cond(seg, modalities, device):
    """bench segment npz -> cond_full {k: [1,...] on device}. DROP seg_mask (leakage)."""
    cond = {}
    if "rgb" in modalities:
        cond["rgb"] = torch.from_numpy(seg["rgb"]).permute(0, 3, 1, 2).float() / 255.0
    if "depth" in modalities and "depth_mm" in seg.files:
        cond["depth"] = torch.from_numpy(seg["depth_mm"].astype(np.float32) / 1000.0)
    if "obj_points" in modalities:
        cond["obj_points"] = torch.from_numpy(seg["obj_verts"].astype(np.float32))
    if "K" in modalities:
        cond["K"] = torch.from_numpy(seg["K"].astype(np.float32))
    return {k: v.unsqueeze(0).to(device) for k, v in cond.items()}


# ------------------------------------------------------------------- hand 3D error vs GT
def hand_error(ch, wrist, thetas, presence, coarse_hand_verts, gt_verts):
    """Per-frame nearest-GT-hand centroid offset (mm), along/perp split. coarse hand centroid
    comes from the optimizer's own hand_verts; refined from FK of the refined state. Returns
    dict of median totals + along/perp, and n frames used."""
    T = len(gt_verts)
    ce = {"along": [], "perp": [], "total": []}
    re = {"along": [], "perp": [], "total": []}
    used = 0
    for t in range(T):
        slots = np.where(presence[t].cpu().numpy())[0]
        if len(slots) == 0:
            continue
        s = int(slots[0])
        # gt_verts[t] is [2,788,3]; split into per-hand centroids, keep finite ones
        hands = np.asarray(gt_verts[t])
        cents = [h.mean(0) for h in hands if np.isfinite(h).all()]
        if not cents:
            continue
        # coarse centroid (optimizer hand) and refined centroid (FK)
        c_cent = coarse_hand_verts[t].mean(0)
        r_cent = _state_hand_verts(ch, wrist[t, s], thetas[t, s], s).mean(0)
        for cent, acc in ((c_cent, ce), (r_cent, re)):
            gtc = min(cents, key=lambda g: np.linalg.norm(cent - g))
            e = torch.from_numpy((cent - gtc).astype(np.float32))[None]
            al, pe = decompose_along_ray(e, torch.from_numpy(gtc.astype(np.float32))[None])
            acc["along"].append(abs(float(al)) * 1000)
            acc["perp"].append(float(pe.norm()) * 1000)
            acc["total"].append(float(e.norm()) * 1000)
        used += 1

    def med(d):
        return {k: (float(np.median(v)) if v else float("nan")) for k, v in d.items()}
    return {"coarse": med(ce), "refined": med(re), "n": used}


# ------------------------------------------------------------------- object scoring subprocess
def score_objects(rc_input, run_dirs):
    """gt_pose_eval_hot3d.py on run_dirs; returns {basename: metrics}."""
    cmd = [PY, "gt_pose_eval_hot3d.py", rc_input] + run_dirs
    r = subprocess.run(cmd, cwd=HOT3D_DIR, capture_output=True, text=True)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr.rstrip())
        raise SystemExit(f"gt_pose_eval failed for {rc_input}")
    tag = "_".join(os.path.basename(d) for d in run_dirs)
    with open(os.path.join(HOT3D_DIR, f"gt_pose_hot3d_{tag}.json")) as f:
        return json.load(f)


def write_run(run_dir, gt_mesh, poses, wrist, thetas, coarse_state):
    d = os.path.join(run_dir, "stage8_eval")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "pseudo_gt.npz"),
             obj_verts=np.asarray(gt_mesh.vertices, np.float64),
             obj_faces=np.asarray(gt_mesh.faces, np.int64),
             obj_poses=poses.astype(np.float64))
    np.savez(os.path.join(d, "hand_refined.npz"),
             hand_wrist=wrist.astype(np.float32), hand_thetas=thetas.astype(np.float32))
    np.savez(os.path.join(d, "coarse_state_debug.npz"), **coarse_state)
    return os.path.join(d, "pseudo_gt.npz")


# ------------------------------------------------------------------------------- per clip
def run_clip(key, model, enc, norm, cfg, modalities, ch, args, device):
    clip, rc_name, obj_run, hand_run = CLIPS[key]
    rc_input = os.path.join(DS, rc_name)
    seg = np.load(os.path.join(DS, "bench_segments", f"seg_bench_{clip}.npz"), allow_pickle=True)
    coarse = np.load(os.path.join(RUNS, obj_run, "stage8_eval", "pseudo_gt.npz"))
    out = np.load(os.path.join(RUNS, hand_run, "hand_reproj_opt", "out.npz"), allow_pickle=True)
    gt = np.load(os.path.join(rc_input, "gt_target.npz"))
    uid = int(gt["uid"])
    gt_glb = trimesh.load(os.path.join(DS, "object_models_eval", f"obj_{uid:06d}.glb"))
    gt_mesh = (trimesh.util.concatenate(list(gt_glb.geometry.values()))
               if isinstance(gt_glb, trimesh.Scene) else gt_glb)
    sam_mesh = trimesh.Trimesh(coarse["obj_verts"], coarse["obj_faces"], process=False)

    T = min(len(coarse["obj_poses"]), seg["obj_pose"].shape[0])
    print(f"\n===== {key}  (obj arm {obj_run.split('_')[-1]}, uid {uid}, T={T}) =====")

    # 1) object gauge fix: SAM-canon coarse poses -> GT-canon
    A, mapped, gf_cham, side = solve_gauge(sam_mesh, gt_mesh, coarse["obj_poses"][:T])
    print(f"  gauge: composition={side}  frame0 chamfer(posedSAM,posedGT)={gf_cham:.1f} mm")

    # 2) hand coarse -> state
    wrist_c, thetas_c = ch.to_state(_hand_adapter(out, T), T)

    # 3) pack coarse state x0 (+ presence from coarse validity), build GT x1 for the internal table
    x0, valid0 = pack_state(torch.from_numpy(mapped).float(),
                            torch.from_numpy(wrist_c).float(), torch.from_numpy(thetas_c).float())
    presence = torch.stack([valid0[:, 9], valid0[:, 33]], dim=-1).to(device)
    x1, _ = pack_state(torch.from_numpy(seg["obj_pose"][:T].astype(np.float64)).float(),
                       torch.from_numpy(seg["hand_wrist"][:T].astype(np.float64)).float(),
                       torch.from_numpy(seg["hand_mano"][:T, :, :15].astype(np.float64)).float())

    # 4) conditioning + refine
    cond_full = build_cond(seg, modalities, device)
    cond_full = {k: (v[:, :T] if k in _PER_FRAME else v) for k, v in cond_full.items()}
    cond_fn = make_cond_fn(cond_full, enc, {}, device)
    x0n = torch.nan_to_num(norm.transform(x0)).to(device)
    vcfg = cfg["val"]
    n_list = [args.n_steps, 1] if args.n_steps != 1 else [1]
    ref = {}
    for n in n_list:
        ref[n] = bridge_sample(model, x0n, cond_fn, presence, n_steps=n,
                               window=vcfg["window"], overlap=vcfg["overlap"],
                               sigma0=vcfg["sigma0"] if args.sigma0 is None else args.sigma0,
                               seed=0)

    # internal normalized-space table (sanity, vs bench GT) — same helper train.py uses
    x1n = norm.transform(x1).to(device)
    print(f"  [internal state metrics vs bench GT]  " + "".join(f"{c:>10}" for c in _COLS))
    for name, xx in [("coarse", x0n)] + [(f"refined@{n}", ref[n]) for n in n_list]:
        mm = state_metrics(xx, x1n, presence, norm)
        row = {c: (float(mm[c].median()) if mm[c].numel() else float("nan")) for c in _COLS}
        print(f"    {name:<11}" + "".join(f"{row[c]:>10.2f}" for c in _COLS))

    # 5) denormalize + write run dirs (main = refined@n_steps; r1 scratch for scoring only)
    def denorm(xn):
        u = unpack_state(norm.inverse(xn.detach().cpu()))
        return u["obj_pose"].numpy(), u["hand_wrist"].numpy(), u["hand_thetas"].numpy()

    p8, w8, t8 = denorm(ref[args.n_steps])
    coarse_state = dict(obj_pose=mapped.astype(np.float32),
                        hand_wrist=wrist_c.astype(np.float32), hand_thetas=thetas_c.astype(np.float32))
    main_dir = os.path.join(RUNS, f"hot3d_{key}_{args.out_suffix}")
    write_run(main_dir, gt_mesh, p8, w8, t8, coarse_state)
    run_scored = [main_dir]
    if 1 in ref and args.n_steps != 1:
        p1, _, _ = denorm(ref[1])
        r1_dir = os.path.join(RUNS, f"hot3d_{key}_{args.out_suffix}_r1")
        write_run(r1_dir, gt_mesh, p1, w8, t8, coarse_state)
        run_scored.append(r1_dir)
    incumbent = os.path.join(RUNS, obj_run)
    run_scored.append(incumbent)

    # 6) score objects (flowref@8, flowref@1, incumbent) + hand error
    obj = score_objects(rc_input, run_scored)
    hb = os.path.basename(main_dir)
    he = hand_error(ch, w8, t8, presence, out["hand_verts"], seg_gt_hands(rc_input))

    return {"key": key, "obj_arm": obj_run.split("_")[-1], "gauge_mm": gf_cham,
            "obj": obj, "main": hb,
            "r1": os.path.basename(run_scored[1]) if len(run_scored) == 3 else None,
            "incumbent": os.path.basename(incumbent), "hand": he}


def seg_gt_hands(rc_input):
    p = os.path.join(rc_input, "gt_hands.npz")
    return np.load(p, allow_pickle=True)["verts"] if os.path.exists(p) else []


# --------------------------------------------------------------------------- consolidated table
def print_table(results):
    print("\n\n## Consolidated benchmark table (object via gt_pose_eval, hand via GT-centroid offset)\n")
    h = ("| clip | arm | gauge mm | chamfer coarse | chamfer ref@8 | chamfer ref@1 | "
         "rot_traj coarse | rot_traj ref@8 | hand off coarse | hand off ref |")
    print(h)
    print("|" + "---|" * 10)
    for r in results:
        o = r["obj"]
        inc = o[r["incumbent"]]
        m8 = o[r["main"]]
        m1 = o.get(r["r1"], {}) if r["r1"] else {}

        def g(d, k):
            return f"{d[k]:.1f}" if k in d else "—"
        hc = r["hand"]["coarse"]["total"]
        hr = r["hand"]["refined"]["total"]
        hand_c = f"{hc:.0f}" if hc == hc else "—"
        hand_r = f"{hr:.0f}" if hr == hr else "—"
        print(f"| {r['key']} | {r['obj_arm']} | {r['gauge_mm']:.1f} | "
              f"{g(inc,'chamfer_mm_med')} | {g(m8,'chamfer_mm_med')} | {g(m1,'chamfer_mm_med')} | "
              f"{g(inc,'rot_traj_deg_med')} | {g(m8,'rot_traj_deg_med')} | {hand_c} | {hand_r} |")
    print("\n(chamfer/rot in mm/deg median; hand off = nearest-GT-hand centroid offset, mm median. "
          "ref@8 = flowref shipped output; coarse = incumbent object arm as it ships.)")


# ------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clips", default="all")
    ap.add_argument("--n_steps", type=int, default=8)
    ap.add_argument("--sigma0", type=float, default=None,
                    help="override sample-time source noise (default: config val.sigma0)")
    ap.add_argument("--out_suffix", default="flowref")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device.startswith("cpu") or torch.cuda.is_available())
                          else "cpu")
    keys = list(CLIPS) if args.clips == "all" else [k.strip() for k in args.clips.split(",")]
    for k in keys:
        if k not in CLIPS:
            raise SystemExit(f"unknown clip '{k}'. known: {list(CLIPS)}")

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.ckpt)), "config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(REPO, "hoi_flow/configs/base.yaml")
    cfg = load_config(cfg_path, [])
    modalities = tuple(m for m, on in cfg["conditioning"]["modalities"].items()
                       if on and m != "seg_mask")               # DROP seg_mask
    np_path = cfg["data"]["normalizer"]
    norm = Normalizer.load(np_path if os.path.isabs(np_path) else os.path.join(REPO, np_path))

    print(f"ckpt={args.ckpt}\nmodalities (seg_mask dropped)={modalities}\ndevice={device}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model, enc = load_model_enc(ckpt, cfg, device)
    ch = _CoarseHand()

    results = []
    for k in keys:
        results.append(run_clip(k, model, enc, norm, cfg, modalities, ch, args, device))
    print_table(results)


if __name__ == "__main__":
    main()
