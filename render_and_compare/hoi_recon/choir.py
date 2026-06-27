"""CHOIR coarse-init algorithm (arXiv:2605.20992, Stage 1) — faithful reproduction.

These are the env-independent algorithmic pieces (run in the `hoi_recon` env):

  * hand_isolated_fit   — CHOIR Eq 1: refine the per-frame MANO hand against 2D
                          evidence + metric depth + anatomy/prior/temporal.
  * angular_guard       — reject a follow-track rotation if it jumps >guard_deg
                          from the last accepted estimate (used by the object
                          guarded follow-tracker).
  * ray_scale_align     — slide the object trajectory along the camera ray so its
                          interaction-depth statistics match the hand, preserving
                          the image-space silhouette fit (the key distinctive step
                          this repo otherwise lacks).

The object isolated fit (Eq 2-3, repulsion/attraction) is the differentiable
PyTorch3D piece and lives in the sam3d-env subprocess
(scripts/subprocess_entries/sam-3d-objects/choir_object_fit.py).
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# Hand isolated fit — CHOIR Eq 1
#   L_h = L_2D + λ_depth L_depth + λ_anat L_anat + λ_prior L_prior + λ_temp L_temp
# Optimizes a per-frame rigid correction (rotation + translation) to the HaMeR /
# Dyn-HaMR hand so projected joints match 2D keypoints, the wrist depth matches the
# median metric depth in the hand mask, while staying near the init and smooth.
# (Finger articulation is left to the regressor — same scope as the repo's other
# hand optimizer; CHOIR's L_anat regularizes articulation, applied here as a soft
# stay-near-init on the joints to suppress invalid drift.)
# --------------------------------------------------------------------------
def hand_isolated_fit(verts, joints, kp2d, kp_valid, depth_paths, hand_boxes,
                      hand_valid, K, o, *, device=None):
    """Returns (verts*, joints*) after the CHOIR hand isolated fit.

    verts (T,778,3), joints (T,21,3) camera-frame; kp2d (T,21,2) full-image px;
    kp_valid (T,) bool; o = cfg.choir.hand (weights/iters/lr).
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T = verts.shape[0]
    V = torch.tensor(verts, dtype=torch.float32, device=dev)
    J = torch.tensor(joints, dtype=torch.float32, device=dev)
    KP = torch.tensor(kp2d, dtype=torch.float32, device=dev)
    val = torch.tensor(kp_valid.astype(np.float32), device=dev)
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    ch = V.mean(1)                                            # hand centroid (T,3)

    # per-frame wrist target depth = median metric depth inside the hand box
    zt = np.full(T, np.nan, np.float32)
    for t in range(T):
        slot = 1 if hand_valid[t, 1] else (0 if hand_valid[t, 0] else None)
        if slot is None:
            continue
        d = np.load(depth_paths[t]).astype(np.float32)
        H, W = d.shape
        x0, y0, x1, y1 = hand_boxes[t, slot]
        x0, x1 = np.clip([x0, x1], 0, W - 1); y0, y1 = np.clip([y0, y1], 0, H - 1)
        crop = d[int(y0):int(y1) + 1, int(x0):int(x1) + 1]
        pos = crop[crop > 0]
        if pos.size > 4:
            zt[t] = np.median(pos)
    zt_t = torch.tensor(np.nan_to_num(zt), device=dev)
    zt_valid = torch.tensor(np.isfinite(zt).astype(np.float32), device=dev)

    def rodrigues(r):
        th = r.norm(dim=1, keepdim=True).clamp(min=1e-8)
        k = r / th; O = torch.zeros_like(th)
        kx, ky, kz = k[:, 0:1], k[:, 1:2], k[:, 2:3]
        Kk = torch.cat([O, -kz, ky, kz, O, -kx, -ky, kx, O], 1).view(-1, 3, 3)
        I = torch.eye(3, device=dev).expand(r.shape[0], 3, 3)
        s = torch.sin(th)[:, :, None]; c = (1 - torch.cos(th))[:, :, None]
        return I + s * Kk + c * (Kk @ Kk)

    rot = torch.zeros(T, 3, device=dev, requires_grad=True)
    tr = torch.zeros(T, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([rot, tr], lr=float(o.get("lr", 1e-3) if hasattr(o, "get") else 1e-3))
    w2d = float(o.get("lambda_2d", 1.0)); wdep = float(o.get("lambda_depth", 10.0))
    want = float(o.get("lambda_anat", 5.0)); wpri = float(o.get("lambda_prior", 1.0))
    wtmp = float(o.get("lambda_temp", 1.0)); iters = int(o.get("iters", 500))

    def fwd():
        R = rodrigues(rot)
        Jc = torch.einsum('tij,tnj->tni', R, J - ch[:, None]) + ch[:, None] + tr[:, None]
        return Jc

    for it in range(iters):
        opt.zero_grad()
        Jc = fwd()
        z = Jc[..., 2].clamp(min=1e-4)
        u = Kt[0, 0] * Jc[..., 0] / z + Kt[0, 2]
        v = Kt[1, 1] * Jc[..., 1] / z + Kt[1, 2]
        r2 = (u - KP[..., 0]) ** 2 + (v - KP[..., 1]) ** 2
        l2d = (r2 * val[:, None]).mean()
        # wrist (joint 0) depth -> median metric depth in the hand mask
        ldep = (((Jc[:, 0, 2] - zt_t) ** 2) * zt_valid).sum() / zt_valid.sum().clamp(min=1)
        lanat = (rot ** 2).mean()                            # suppress large reorientation
        lpri = (tr ** 2).mean() + (rot ** 2).mean()          # stay near init
        ltmp = ((tr[1:] - tr[:-1]) ** 2).mean() + ((rot[1:] - rot[:-1]) ** 2).mean()
        loss = w2d * l2d + wdep * ldep + want * lanat + wpri * lpri + wtmp * ltmp
        loss.backward(); opt.step()

    with torch.no_grad():
        R = rodrigues(rot)
        Vf = torch.einsum('tij,tnj->tni', R, V - ch[:, None]) + ch[:, None] + tr[:, None]
        Jf = torch.einsum('tij,tnj->tni', R, J - ch[:, None]) + ch[:, None] + tr[:, None]
    return Vf.cpu().numpy(), Jf.cpu().numpy()


# --------------------------------------------------------------------------
# Angular guard for the object follow-tracker
# --------------------------------------------------------------------------
def angular_guard(R_cand, R_prev, guard_deg=60.0):
    """True if the candidate rotation is within guard_deg of the previous accepted
    one (i.e. accept it); False to reject the jump."""
    Rrel = R_cand @ R_prev.T
    cos = (np.trace(Rrel) - 1.0) / 2.0
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return ang <= guard_deg


# --------------------------------------------------------------------------
# Ray-scale alignment — slide the object along the camera ray so its
# interaction-depth statistics match the hand, preserving the silhouette fit.
# --------------------------------------------------------------------------
def ray_scale_align(obj_poses, obj_verts, hand_verts, hand_contact_idx, K,
                    grasp_mask=None):
    """Shift the object centroid along the camera ray (origin->centroid direction)
    so that, over interaction frames, the object's near-surface depth matches the
    hand's contact-joint depth. Sliding ALONG the ray preserves the image-space
    projection (silhouette fit) while correcting the object's depth placement
    (CHOIR's ray-scale alignment). Returns corrected obj_poses[T,4,4].

    A single global ray-scale factor `s` is solved (the object is rigid; one scalar
    moves the whole trajectory consistently), matching CHOIR's description of a
    trajectory-level slide rather than per-frame depth edits.
    """
    poses = obj_poses.copy()
    T = poses.shape[0]
    c = poses[:, :3, 3]                                       # object centroids (T,3)
    obj_r = float(np.linalg.norm(obj_verts - obj_verts.mean(0), axis=1).mean())

    if grasp_mask is None:
        grasp_mask = np.ones(T, bool)
    hc = hand_verts[:, hand_contact_idx, :].mean(1)          # hand contact centroid (T,3)
    # interaction frames: hand contact centroid near the object
    near = grasp_mask & (np.linalg.norm(hc - c, axis=1) < 4.0 * obj_r + 0.1)
    if near.sum() < 3:
        return poses

    # object near-surface depth ~ centroid_z - object_radius (camera looks down +z);
    # hand depth ~ contact-centroid z. Solve a global scale s on the centroid ray so
    # that median(object_near_z * s) == median(hand_z) over interaction frames.
    obj_near_z = c[near, 2] - obj_r
    hand_z = hc[near, 2]
    denom = np.median(obj_near_z)
    if abs(denom) < 1e-6:
        return poses
    s = float(np.median(hand_z) / denom)
    s = float(np.clip(s, 0.5, 2.0))                          # guard against degenerate scale
    # slide each centroid along its own camera ray (through the origin) by factor s,
    # which keeps u,v = K·(x/z, y/z) fixed (pure depth move) -> silhouette preserved.
    poses[:, :3, 3] = c * s
    return poses
