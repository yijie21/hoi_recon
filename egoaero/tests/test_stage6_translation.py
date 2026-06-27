import numpy as np
from egoaero.stages import stage6_contact as C


def test_active_window_keeps_manipulation():
    labels = ["pre", "pre", "grasp", "move", "place", "post"]
    win = C.active_window(labels)
    assert 2 in win and 3 in win and 4 in win and 0 not in win


def test_whole_hand_translation_pulls_toward_surface_and_clips():
    # one floating contact point 2cm in +z from a plane at z=0 (normal +z)
    obj_pts = np.array([[0, 0, 0.0], [0.01, 0, 0], [0, 0.01, 0]])
    obj_n = np.tile([0, 0, 1.0], (3, 1))
    pts = np.array([[0, 0, 0.02]])
    s, nn = C.signed_distance(pts, obj_pts, obj_n)
    assert abs(s[0] - 0.02) < 1e-6
    d = C.whole_hand_translation([pts], [s], [nn], gaps=[0.0005],
                                 weights=[1.0], max_trans=0.034)
    assert d[2] < 0                        # pulled toward the surface (-z)
    assert np.linalg.norm(d) <= 0.034 + 1e-9
