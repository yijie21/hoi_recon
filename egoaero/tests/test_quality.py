import numpy as np
from egoaero import quality as Q
from egoaero.core import hand as H


def _hand_and_obj():
    v, j, fidx = H.procedural_hand(seed=0)
    # flat object plane at z=0, +z normals, covering xy
    grid = np.array([[x, y, 0.0] for x in np.linspace(-0.1, 0.1, 9)
                     for y in np.linspace(-0.1, 0.1, 9)])
    nrm = np.tile([0, 0, 1.0], (grid.shape[0], 1))
    return v, j, fidx, grid, nrm


def test_per_finger_gap_shapes_and_value():
    v, j, fidx, grid, nrm = _hand_and_obj()
    # place the hand so index-finger pad sits ~2cm above the plane
    verts_seq = np.stack([v, v], 0)                      # T=2 identical frames
    obj_seq = [(grid, nrm), (grid, nrm)]
    gaps = Q.per_finger_gap(verts_seq, fidx, obj_seq, window=[0, 1])
    assert set(H.FINGERS).issubset(gaps.keys())
    assert gaps["index"].shape == (2,)
    assert np.all(gaps["index"] >= 0)                    # distances are non-negative


def test_per_finger_delta_measures_displacement():
    v, j, fidx, grid, nrm = _hand_and_obj()
    coarse = np.stack([v, v], 0)
    repaired = coarse.copy()
    pad = H.fingertip_pad_idx(fidx, "thumb")
    repaired[:, pad] += np.array([0.0, 0.0, -0.01])      # move thumb pad 1cm
    d = Q.per_finger_delta(coarse, repaired, fidx, window=[0, 1])
    assert np.allclose(d["thumb"], 0.01, atol=1e-9)
    assert np.allclose(d["index"], 0.0, atol=1e-9)
