import numpy as np
from egoaero.core.mock_scene import generate_ego_hoi

def test_shapes_and_ego_motion():
    s = generate_ego_hoi(num_frames=24, seed=0)
    assert s.hand_verts_w.shape[0] == 24 and s.obj_poses_w.shape == (24, 4, 4)
    assert s.obj_mask.shape == (24, 480, 640) and s.depth.shape == (24, 480, 640)
    # head moves: camera trajectory is not constant
    assert not np.allclose(s.cam_traj[0], s.cam_traj[-1])
    # depth positive where object is visible
    assert s.depth[s.obj_mask].min() > 0
    assert set(np.unique(s.stage_labels)).issubset({"pre","grasp","move","place","post"})
