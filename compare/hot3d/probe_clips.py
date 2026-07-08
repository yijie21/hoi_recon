"""Probe-download BOP train Aria clips from HIT-rich sequences and index them
by within-clip GT motion of the HIT target category, to select single-object
interaction clips for the batch pipeline run.

HIT frame indices live on the full-sequence axis (needs the VRS timestamp
list we don't have), so selection is two-stage: HIT picks the sequence and
target object (human-verified interaction-rich), the clip's own mocap GT
confirms the target actually moves within the 5-s window.
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download
from scipy.spatial.transform import Rotation

DS = "/workspace/datasets/hot3d"
# sequence -> HIT target category to check
PROBE = {
    "P0002_016222d1": ["plate_bamboo", "spatula_red"],
    "P0012_e97d31b6": ["carton_oj", "spatula_red"],
    "P0002_65085bfc": ["bowl", "bottle_bbq"],
    "P0001_f6cc0cc8": ["mug_white", "puzzle_toy"],
    "P0003_c701bd11": ["coffee_pot", "mug_white"],
    "P0009_9e03121a": ["potato_masher", "bowl"],
}
N_PER_SEQ = 3


def T_from(d):
    R = Rotation.from_quat(np.roll(d["quaternion_wxyz"], -1)).as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, d["translation_xyz"]
    return T


def main():
    splits = json.load(open(f"{DS}/clip_splits.json"))
    clips = json.load(open(f"{DS}/clip_definitions.json"))
    names = json.load(open(f"{DS}/object_models_models_info.json"))
    name2uid = {v["name"]: k for k, v in names.items()}
    train_aria = set(splits["train"]["Aria"])

    byseq = defaultdict(list)
    for cid, v in clips.items():
        if v["device"] == "Aria" and int(cid) in train_aria and \
                v["sequence_id"] in PROBE:
            ts0 = list(v["per_frame_timestamps_ns"][0].values())[0]
            byseq[v["sequence_id"]].append((ts0, int(cid)))

    chosen = []
    for sq, lst in byseq.items():
        lst.sort()
        idx = np.linspace(0, len(lst) - 1, N_PER_SEQ).astype(int)
        chosen += [(sq, lst[i][1]) for i in sorted(set(idx))]
    print(f"probing {len(chosen)} clips:", [c for _, c in chosen], flush=True)

    report = []
    for sq, cid in chosen:
        name = f"clip-{cid:06d}"
        cdir = f"{DS}/clips/{name}"
        if not os.path.isdir(cdir):
            tar = hf_hub_download("bop-benchmark/hot3d", f"train_aria/{name}.tar",
                                  repo_type="dataset", local_dir=f"{DS}/_tars")
            os.makedirs(cdir, exist_ok=True)
            subprocess.run(["tar", "xf", tar, "-C", cdir], check=True)
            os.remove(tar)
        # index: per target cat, motion + view angle over the clip
        import glob
        frames = sorted(glob.glob(f"{cdir}/*.objects.json"))
        stats = {}
        for cat in PROBE[sq]:
            uid = name2uid[cat]
            tr, ang = [], []
            for f in frames[::5]:
                d = json.load(open(f))
                if uid not in d:
                    continue
                cams = json.load(open(f.replace(".objects.json", ".cameras.json")))["214-1"]
                T_cw = np.linalg.inv(T_from(cams["T_world_from_camera"]))
                Tc = T_cw @ T_from(d[uid][0]["T_world_from_object"])
                c = Tc[:3, 3]
                tr.append(T_from(d[uid][0]["T_world_from_object"])[:3, 3])
                ang.append(np.degrees(np.arccos(np.clip(c[2] / np.linalg.norm(c), -1, 1))))
            if not tr:
                stats[cat] = None
                continue
            tr = np.array(tr)
            stats[cat] = {"move_cm": float(np.linalg.norm(tr.max(0) - tr.min(0)) * 100),
                          "ang_med": float(np.median(ang)),
                          "ang_max": float(np.max(ang))}
        report.append({"seq": sq, "clip": name, "stats": stats})
        print(json.dumps(report[-1]), flush=True)
    json.dump(report, open(f"{DS}/probe_report.json", "w"), indent=1)
    print("wrote", f"{DS}/probe_report.json")


if __name__ == "__main__":
    main()
