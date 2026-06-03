#!/usr/bin/env python
"""Validate that the single `hoi_recon` conda env can run ALL related code.

Two parts:
  A. import matrix — core libs, this package, every third-party real backend, and
     the known-conflict `chumpy`.
  B. fake-load pipeline — monkeypatch the model loaders to return stubs with random
     weights, then run the REAL stage 0-8 code path on a tiny synthetic video. This
     exercises the real adapters + glue + geometry without any downloaded checkpoint.

Run:  python scripts/check_env.py        (from the repo root, inside `hoi_recon`)
Exit code 0 if everything except the documented chumpy conflict passes.
"""
import importlib
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

results = []


def check(label, fn, expect_fail=False):
    try:
        fn()
        status = "OK"
        results.append((label, True, expect_fail, ""))
    except Exception as e:
        status = "FAIL"
        results.append((label, False, expect_fail, f"{type(e).__name__}: {str(e)[:90]}"))
    mark = {"OK": "  ok ", "FAIL": "FAIL "}[status]
    tag = "  (expected)" if (status == "FAIL" and expect_fail) else ""
    print(f"  {mark} {label}{tag}")
    if status == "FAIL":
        print(f"         └─ {results[-1][3]}")


def imp(module, path=None):
    if path and path not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, path))
    importlib.import_module(module)


# ---------------------------------------------------------------- A. imports
def section(name):
    print(f"\n── {name} " + "─" * (60 - len(name)))


section("A. core libraries")
for m in ["torch", "numpy", "cv2", "scipy", "trimesh", "viser", "yaml",
          "matplotlib", "imageio"]:
    check(f"core: {m}", lambda m=m: imp(m))
check("torch CUDA available", lambda: __import__("torch").cuda.is_available()
      or (_ for _ in ()).throw(RuntimeError("torch.cuda.is_available() is False")))

section("B. hoi_recon package")
for m in ["hoi_recon.cli", "hoi_recon.pipeline", "hoi_recon.config",
          "hoi_recon.geometry", "hoi_recon.backends.real_perception",
          "hoi_recon.viz.viser_app"]:
    check(f"pkg: {m}", lambda m=m: imp(m))
for i in range(9):
    check(f"stage: {i}", lambda i=i: imp(
        f"hoi_recon.stages.stage{i}_" + ["preprocess", "detect_track", "hand",
        "object", "align", "coarse_fit", "rectify", "contact_optim", "eval"][i]))

section("C. third-party real backends")
check("depth: MoGe (moge.model.v2)", lambda: imp("moge.model.v2"))
check("depth/camera: Depth-Anything-3 (depth_anything_3.api)",
      lambda: imp("depth_anything_3.api"))
check("seg: SAM 2 (sam2.build_sam)", lambda: imp("sam2.build_sam"))
check("detect: ultralytics (YOLO)", lambda: imp("ultralytics"))
check("hand: HaMeR (hamer.models)", lambda: imp("hamer.models", "third_party/hamer"))
check("hand: HaMeR (hamer.datasets.vitdet_dataset)",
      lambda: imp("hamer.datasets.vitdet_dataset", "third_party/hamer"))
check("hand: WiLoR (wilor)", lambda: imp("wilor", "third_party/WiLoR"))
check("hand: smplx", lambda: imp("smplx"))

section("D. known conflict (numpy>=2)")
check("chumpy (needed only to load the official MANO .pkl)",
      lambda: imp("chumpy"), expect_fail=True)


# ------------------------------------------------------- B. fake-load pipeline
def run_fake_pipeline():
    import cv2
    from hoi_recon.config import load_config
    from hoi_recon import pipeline
    from hoi_recon.backends import real_perception as rp

    d = tempfile.mkdtemp()
    vid = os.path.join(d, "clip.mp4")
    vw = cv2.VideoWriter(vid, cv2.VideoWriter_fourcc(*"mp4v"), 5, (128, 128))
    rng = np.random.default_rng(0)
    for _ in range(4):
        vw.write((rng.random((128, 128, 3)) * 255).astype(np.uint8))
    vw.release()

    # --- stub the heavy model entry points with fake (random) outputs ---
    def fake_geo(cfg, frame_paths, out_dir):
        T = len(frame_paths)
        dd = os.path.join(out_dir, "depth"); os.makedirs(dd, exist_ok=True)
        dp = []
        for i in range(T):
            depth = (rng.random((128, 128)) * 0.2 + 0.5).astype(np.float16)
            p = os.path.join(dd, f"{i:05d}.npy"); np.save(p, depth); dp.append(p)
        K = np.array([[128., 0, 64], [0, 128., 64], [0, 0, 1.]])
        return {"intrinsics": K, "extrinsics": np.tile(np.eye(4), (T, 1, 1)),
                "depth_dir": dd, "depth_paths": dp, "image_size": (128, 128),
                "camera_source": "fake"}

    def fake_detect(cfg, frame_paths):
        T = len(frame_paths)
        b = np.full((T, 2, 4), np.nan); v = np.zeros((T, 2), bool)
        b[:, 1] = [40, 40, 95, 95]; v[:, 1] = True
        return b, v

    def fake_seg(cfg, frames_dir, frame_paths, prompt, out_dir):
        md = os.path.join(out_dir, "masks"); os.makedirs(md, exist_ok=True); mp = []
        for i in range(len(frame_paths)):
            m = np.zeros((128, 128), bool); m[55:95, 55:95] = True
            p = os.path.join(md, f"{i:05d}.npy"); np.save(p, m); mp.append(p)
        return md, mp

    rp.run_stage0_geometry = fake_geo
    rp.detect_hands = fake_detect
    rp.segment_object = fake_seg

    cfg = load_config(None, {"mock": False, "video": vid, "force": True,
        "backend": {"hand": "depthlift", "object": "sam3d",
                    "depth": "moge", "camera": "vipe"}})
    run_dir = os.path.join(d, "run")
    pipeline.run_pipeline(cfg, run_dir, "all")
    assert os.path.exists(os.path.join(run_dir, "stage8_eval", "meta.json")), \
        "pipeline did not reach stage8"


section("E. fake-load run of the REAL pipeline (stages 0-8, stub weights)")
# silence stage logging noise during the run
import hoi_recon.logging_utils as _lg
_lg.log = lambda *a, **k: None
check("real pipeline executes end-to-end with stub models", run_fake_pipeline)


# ------------------------------------------------------------------- summary
print("\n" + "=" * 64)
hard_fail = [r for r in results if not r[1] and not r[2]]
expected = [r for r in results if not r[1] and r[2]]
passed = [r for r in results if r[1]]
print(f"  passed: {len(passed)}   hard-fail: {len(hard_fail)}   "
      f"expected-fail: {len(expected)}")
if hard_fail:
    print("\n  HARD FAILURES (block the single-env goal):")
    for label, _, _, err in hard_fail:
        print(f"    - {label}: {err}")
print("\n  Verdict:")
if not hard_fail:
    print("  ✅ One conda env runs ALL related code (ours + MoGe + DA3 + SAM2 +")
    print("     ultralytics + HaMeR + WiLoR) and the full real pipeline (fake weights).")
    print("  ⚠️  Only `chumpy` cannot coexist with numpy>=2 — it is needed solely to")
    print("     deserialize the official MANO .pkl for --hand hamer/wilor. Use")
    print("     --hand depthlift (no MANO), or a patched chumpy / side-env for that step.")
sys.exit(1 if hard_fail else 0)
