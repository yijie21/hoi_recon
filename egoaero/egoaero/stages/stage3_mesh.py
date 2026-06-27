"""Stage 3 (§2.1.2 / App B): neural-field coarse mesh + SAM3D fine mesh, aligned.
App B is a stub in the paper; mock uses the tracked object geometry as coarse and
recovers the SAM3D->coarse rigid+scale alignment (the one specified operation)."""
from __future__ import annotations
import numpy as np
from ..bundle import Bundle
from ..core.geometry import umeyama, rotvec_to_R, transform_points

NAME = "stage3_mesh"; INDEX = 3


def run(ctx) -> Bundle:
    cfg = ctx.cfg
    s2 = ctx.load("stage2_track")
    if not cfg.mock:
        raise NotImplementedError("real neural-field + SAM3D backend — backends/real.py")
    rng = np.random.default_rng(int(cfg.seed) + 2)
    coarse = s2["obj_verts"]                                  # M_O^coarse (canonical)
    # synthetic SAM3D mesh: coarse under an unknown rigid+scale (+ small detail noise)
    s, R, t = 1.15, rotvec_to_R(np.array([0.1, -0.2, 0.05])), np.array([0.3, -0.1, 0.2])
    sam = s * transform_points(coarse, np.eye(4)) @ R.T + t
    sam = sam + rng.normal(0, 1e-4, sam.shape)
    s_hat, R_hat, t_hat = umeyama(sam, coarse, with_scale=True)
    aligned = s_hat * (sam @ R_hat.T) + t_hat
    resid = float(np.median(np.linalg.norm(aligned - coarse, axis=1)))
    return Bundle(arrays={"obj_verts": aligned, "obj_faces": s2["obj_faces"],
                          "align_s": np.array(s_hat), "align_R": R_hat, "align_t": t_hat},
                  meta={"align_residual_m": resid})
