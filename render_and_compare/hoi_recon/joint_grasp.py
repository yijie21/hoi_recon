"""Joint hand + object contact optimization (tight-grasp refinement).

Stage 7's default optimizer holds the hand fixed and only slides the object to the
fingertips, which yields a fingertip touch rather than an enveloping grasp. This
module instead jointly optimizes a per-frame rigid hand transform AND the object
6D pose under CHOIR-style energies, so the hand closes around the object:

  E_seat     pull the object centroid into the hand's palm (grasp driver)
  E_contact  pull proximity-recruited hand verts onto the object surface
             (recomputed periodically, so the contact set GROWS as the hand
             closes -> envelopment, not just fingertips)
  E_pen      one-sided non-penetration (hand verts must stay outside the object,
             via nearest object surface point + its normal)
  E_smooth   temporal smoothness on the hand & object motion
  E_prior    stay near the HaMeR hand / stage-6 object (no unrealistic drift)

Torch + Adam on GPU. The hand is moved rigidly (global rotation+translation per
frame) — finger articulation is not re-posed (HaMeR's MANO pose params aren't
carried through the pipeline), but rigidly seating the already-curled hand onto
the object closes the few-cm placement gap that caused the loose grasp.
"""
from __future__ import annotations

import numpy as np

from .logging_utils import log


def _rodrigues(r):
    """Batched axis-angle (T,3) -> rotation matrices (T,3,3)."""
    import torch
    theta = r.norm(dim=1, keepdim=True).clamp(min=1e-8)         # (T,1)
    k = r / theta
    O = torch.zeros_like(theta)
    kx, ky, kz = k[:, 0:1], k[:, 1:2], k[:, 2:3]
    K = torch.cat([O, -kz, ky, kz, O, -kx, -ky, kx, O], dim=1).view(-1, 3, 3)
    I = torch.eye(3, device=r.device, dtype=r.dtype).expand(r.shape[0], 3, 3)
    s = torch.sin(theta)[:, :, None]
    c = (1 - torch.cos(theta))[:, :, None]
    return I + s * K + c * (K @ K)


def _obj_vertex_normals(verts, faces):
    import trimesh
    m = trimesh.Trimesh(np.asarray(verts), np.asarray(faces), process=False)
    return np.asarray(m.vertex_normals, np.float32)


def joint_optimize(hand_verts, hand_joints, contact_idx, obj_verts, obj_faces,
                   obj_poses, obj_radius, o, *, iters=400, cache_period=15,
                   tau=0.02, device=None):
    """Returns (hand_verts*, hand_joints*, obj_poses*, stats).

    o: cfg.optim (uses w_contact/w_pen/w_temporal/w_anchor/lr plus the grasp
    weights w_seat/w_prior_hand if present; sensible defaults otherwise).
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    T, Nh = hand_verts.shape[0], hand_verts.shape[1]

    # Grasp-energy weights are intentionally NOT read from cfg.optim — those are
    # tuned for the mock object-only optimizer (different energies/scales, e.g.
    # w_pen=0.3 for radial penetration). These are tuned for this joint problem;
    # override via cfg.optim["joint"] = {w_seat:..,w_pen:..,..} if needed.
    j = (o.get("joint", {}) if hasattr(o, "get") else {}) or {}
    def w(name, default):
        return float(j.get(name, default)) if name in j else default
    w_seat = w("w_seat", 3.0)        # pull object surface onto the palm
    w_contact = w("w_contact", 1.5)  # recruited hand verts onto object surface
    w_pen = w("w_pen", 60.0)         # non-penetration (hand must stay outside object)
    w_smooth = w("w_smooth", 1.0)    # temporal smoothness of hand+object motion
    w_ph = w("w_prior_hand", 3.0)    # let the hand move to close onto the object
    w_po = w("w_prior_obj", 25.0)    # KEEP the object on its image-grounded track
                                     # (it reprojects onto the real object to a few px;
                                     #  the hand should close the grasp, not the object move)
    lr = w("lr", 0.006)

    Vh0 = torch.tensor(hand_verts, dtype=torch.float32, device=dev)
    Jh0 = torch.tensor(hand_joints, dtype=torch.float32, device=dev)
    Vo0 = torch.tensor(obj_verts, dtype=torch.float32, device=dev)
    Nn0 = torch.tensor(_obj_vertex_normals(obj_verts, obj_faces), dtype=torch.float32, device=dev)
    P0 = torch.tensor(obj_poses, dtype=torch.float32, device=dev)
    R0, t0 = P0[:, :3, :3], P0[:, :3, 3]                        # (T,3,3),(T,3)
    ch0 = Vh0.mean(1)                                           # hand centroid (T,3)
    PALM = [0, 5, 9, 13, 17]
    cpalm0 = Jh0[:, PALM, :].mean(1)                            # palm centre (T,3)

    r = float(obj_radius) if obj_radius and float(obj_radius) > 0 else 0.05
    # seat distance = object cross-section radius (half its smallest extent): the
    # object surface should rest on the palm, so its CENTRE sits ~r_seat from the
    # palm centre (pulling the centre onto the palm would bury the hand inside it).
    ext = Vo0.max(0).values - Vo0.min(0).values
    r_seat = float(0.5 * ext.min())
    # grasp frames: palm near the object centroid (object centroid = t0, verts centered)
    graspf = (torch.norm(cpalm0 - t0, dim=1) < 2.0 * r + 0.06).float()   # (T,)

    th = torch.zeros(T, 3, device=dev, requires_grad=True)
    rh = torch.zeros(T, 3, device=dev, requires_grad=True)
    to = torch.zeros(T, 3, device=dev, requires_grad=True)
    ro = torch.zeros(T, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([th, rh, to, ro], lr=lr)

    def forward():
        Rh = _rodrigues(rh)
        Vh = torch.einsum('tij,tnj->tni', Rh, Vh0 - ch0[:, None]) + ch0[:, None] + th[:, None]
        Jh = torch.einsum('tij,tnj->tni', Rh, Jh0 - ch0[:, None]) + ch0[:, None] + th[:, None]
        dRo = _rodrigues(ro)
        Vo_w = torch.einsum('tij,nj->tni', R0, Vo0) + t0[:, None]
        Vo = torch.einsum('tij,tnj->tni', dRo, Vo_w - t0[:, None]) + t0[:, None] + to[:, None]
        RoW = torch.einsum('tij,tjk->tik', dRo, R0)
        Nw = torch.einsum('tij,nj->tni', RoW, Nn0)
        cpalm = torch.einsum('tij,tj->ti', Rh, cpalm0 - ch0) + ch0 + th
        ocen = t0 + to
        return Vh, Jh, Vo, Nw, cpalm, ocen

    gmask = graspf > 0.5
    cache = {"oidx": None, "cmask": None}

    def recache(Vh, Vo):
        with torch.no_grad():
            oidx = torch.zeros(T, Nh, dtype=torch.long, device=dev)
            cmask = torch.zeros(T, Nh, device=dev)
            for t in range(T):
                if not bool(gmask[t]):
                    continue
                d = torch.cdist(Vh[t], Vo[t])              # (Nh,No)
                dmin, idx = d.min(dim=1)
                oidx[t] = idx
                cmask[t] = (dmin < tau).float()
            cache["oidx"], cache["cmask"] = oidx, cmask

    stats = {}
    for it in range(iters):
        if it % cache_period == 0:
            with torch.no_grad():
                Vh, _, Vo, _, _, _ = forward()
            recache(Vh, Vo)
        Vh, Jh, Vo, Nw, cpalm, ocen = forward()
        oidx, cmask = cache["oidx"], cache["cmask"]
        p = torch.gather(Vo, 1, oidx[:, :, None].expand(-1, -1, 3))   # nearest obj pt
        n = torch.gather(Nw, 1, oidx[:, :, None].expand(-1, -1, 3))   # its normal
        diff = Vh - p
        signed = (diff * n).sum(-1)                                   # >0 outside, <0 inside

        loss = 0.0
        # seat: bring the object's SURFACE to the palm (centre ~r_seat away), so the
        # object is held in the palm without the hand burying into it.
        dpc = torch.norm(ocen - cpalm, dim=1)
        seat = (dpc - r_seat) ** 2
        loss = loss + w_seat * (seat * graspf).sum() / graspf.sum().clamp(min=1)
        # contact: recruited hand verts onto object surface
        cd = (diff ** 2).sum(-1)
        loss = loss + w_contact * (cmask * cd).sum() / cmask.sum().clamp(min=1)
        # penetration: hand verts inside the object (signed<0); quadratic + linear so
        # even shallow penetration is pushed out.
        pen = torch.relu(-signed)
        loss = loss + w_pen * (pen ** 2).mean() + 0.3 * w_pen * pen.mean()
        # temporal smoothness on hand+object motion
        for q in (th, rh, to, ro):
            loss = loss + w_smooth * ((q[1:] - q[:-1]) ** 2).mean()
        # priors (stay near HaMeR / stage6)
        loss = loss + w_ph * (th ** 2).mean() + w_ph * (rh ** 2).mean()
        loss = loss + w_po * (to ** 2).mean() + w_po * (ro ** 2).mean()

        opt.zero_grad(); loss.backward(); opt.step()
        if it % 80 == 0 or it == iters - 1:
            with torch.no_grad():
                log(f"  joint iter {it:3d} loss={float(loss.detach()):.5f} "
                    f"|hand move|max={float(th.abs().max())*100:.1f}cm")

    with torch.no_grad():
        Vh, Jh, Vo, Nw, cpalm, ocen = forward()
        dRo = _rodrigues(ro)
        Pn = torch.eye(4, device=dev).repeat(T, 1, 1)
        Pn[:, :3, :3] = torch.einsum('tij,tjk->tik', dRo, R0)
        Pn[:, :3, 3] = t0 + to
        # envelopment stat: hand verts within 1cm of object
        env = 0
        for t in range(T):
            if bool(gmask[t]):
                env += int((torch.cdist(Vh[t], Vo[t]).min(1).values < 0.01).sum())
        stats["env_within_1cm"] = env / max(int(gmask.sum()), 1)
        stats["grasp_frames"] = int(gmask.sum())
    return (Vh.cpu().numpy(), Jh.cpu().numpy(), Pn.cpu().numpy(), stats)
