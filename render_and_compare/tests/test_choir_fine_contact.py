import numpy as np
import pytest
import trimesh
from hoi_recon.choir_fine import contact


def _quad_mesh():
    """A flat 2x2 quad in the z=0 plane (two triangles), normals +z."""
    v = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]])
    return trimesh.Trimesh(v, f, process=False)


def _sphere_mesh():
    """An icosphere radius 0.1 at the origin, outward normals — a hand point just
    outside along a radius passes the faithful 60° normal gate (curved surface)."""
    return trimesh.creation.icosphere(subdivisions=3, radius=0.1)


def test_near_point_gets_valid_correspondence():
    m = _quad_mesh()
    hand = np.array([[0.0, 0.0, 0.01]])              # 1cm above the surface
    out = contact.build_correspondences(hand, m, dist_thresh=0.02, topk=8, seed=0)
    assert out["valid"][0]
    assert out["weight"][0].sum() == \
        pytest.approx(1.0, abs=1e-5)   # softmax weights normalized
    # anchors lie on the surface (z ~ 0)
    anchors = out["anchor"][0][out["weight"][0] > 0]
    assert np.abs(anchors[:, 2]).max() < 1e-6


def test_far_point_is_invalid():
    m = _quad_mesh()
    hand = np.array([[0.0, 0.0, 0.10]])              # 10cm away > 2cm gate
    out = contact.build_correspondences(hand, m, dist_thresh=0.02, topk=8, seed=0)
    assert not out["valid"][0]


def test_wrong_side_normal_gate_rejects():
    """A point approaching from BEHIND the surface (−z) fails the normal cone."""
    m = _quad_mesh()
    hand = np.array([[0.0, 0.0, -0.01]])             # below the +z surface
    out = contact.build_correspondences(hand, m, dist_thresh=0.02,
                                        normal_deg=60.0, topk=8, seed=0)
    assert not out["valid"][0]


def test_bary_reconstructs_anchor():
    """face_id + barycentric must reconstruct the stored anchor, using the FAITHFUL
    60° normal-gate default (sphere fixture so the gate fires correctly)."""
    m = _sphere_mesh()
    hand = np.array([[0.105, 0.0, 0.0]])             # 5mm outside the sphere along +x
    out = contact.build_correspondences(hand, m, dist_thresh=0.02, topk=4, seed=0)
    assert out["valid"][0]
    k = int(np.argmax(out["weight"][0]))
    fid = out["face_id"][0, k]
    bary = out["bary"][0, k]
    recon = (m.triangles[fid] * bary[:, None]).sum(0)
    assert np.allclose(recon, out["anchor"][0, k], atol=1e-5)


def test_anchor_build_constants_are_valid_kwargs():
    """presets.ANCHOR_BUILD must be directly usable as build_correspondences kwargs."""
    from hoi_recon.choir_fine import presets
    m = _sphere_mesh()
    hand = np.array([[0.105, 0.0, 0.0]])
    out = contact.build_correspondences(hand, m, **presets.ANCHOR_BUILD)  # must not raise
    assert out["valid"][0]
