import numpy as np
from egoaero.core.mock_scene import generate_ego_hoi

def test_tightness_deepens_contact():
    loose = generate_ego_hoi(num_frames=24, seed=0, tightness=0.0)
    tight = generate_ego_hoi(num_frames=24, seed=0, tightness=1.0)
    # mid-clip the tight hand sits closer to / inside the object than the loose hand
    mid = 12
    obj_c = tight.obj_poses_w[mid, :3, 3]
    d_loose = np.linalg.norm(loose.hand_verts_w[mid].mean(0) - obj_c)
    d_tight = np.linalg.norm(tight.hand_verts_w[mid].mean(0) - obj_c)
    assert d_tight < d_loose

def test_tightness_default_unchanged():
    a = generate_ego_hoi(num_frames=16, seed=1)
    b = generate_ego_hoi(num_frames=16, seed=1, tightness=0.0)
    assert np.allclose(a.hand_verts_w, b.hand_verts_w)   # default == current behavior
