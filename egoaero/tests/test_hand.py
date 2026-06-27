import numpy as np
from egoaero.core import hand as H


def test_structure():
    v, j, fidx = H.procedural_hand(seed=1)
    assert v.shape[1] == 3 and j.shape == (21, 3)
    assert set(H.FINGERS).issubset(fidx.keys())
    for f in H.FINGERS:
        assert len(fidx[f]) > 0


def test_pad_weights_distal_heavy():
    v, j, fidx = H.procedural_hand(seed=1)
    w = H.finger_chain_weights(v, fidx, "index")
    pad = H.fingertip_pad_idx(fidx, "index")
    # pad weights are near 1; palm weights near 0
    assert w[pad].mean() > 0.8
    assert w[fidx["palm"]].mean() < 0.2
    assert w.min() >= 0.0 and w.max() <= 1.0
