import copy
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from leaderboard import gate

BASE = {c: {"chamfer_mm": ch, "rot_traj_med": rt}
        for c, ch, rt in [("vase", 17.5, 19.4), ("mug_white", 60.7, 18.0),
                          ("spatula_red", 158.8, 32.8)]}

def test_worst_chamfer_improves_passes():
    cand = copy.deepcopy(BASE)
    cand["spatula_red"]["chamfer_mm"] = 25.0     # worst clip fixed
    ok, why = gate(cand, BASE)
    assert ok, why

def test_regression_blocks():
    cand = copy.deepcopy(BASE)
    cand["spatula_red"]["chamfer_mm"] = 25.0
    cand["vase"]["rot_traj_med"] = 25.0          # +29% > 20% regression
    ok, why = gate(cand, BASE)
    assert not ok and "vase" in why

def test_tie_needs_rot_improvement():
    cand = copy.deepcopy(BASE)                   # chamfer all tied
    ok, _ = gate(cand, BASE)
    assert not ok                                # nothing improved
    cand["mug_white"]["rot_traj_med"] = 10.0     # mean rot_traj drops
    ok, why = gate(cand, BASE)
    assert ok, why

def test_small_rot_traj_increase_within_noise_floor_passes():
    """rot_traj 19.4->24.0 (+24% but +4.6° < 5° floor) with best-chamfer improvement PASSES."""
    cand = copy.deepcopy(BASE)
    cand["spatula_red"]["chamfer_mm"] = 25.0     # worst clip fixed (improves overall)
    cand["vase"]["rot_traj_med"] = 24.0          # 19.4 -> 24.0 is +4.6°, exceeds 20% relative but < 5° floor
    ok, why = gate(cand, BASE)
    assert ok, why

def test_large_rot_traj_increase_beyond_noise_floor_fails():
    """rot_traj 32.8->40.0 (+22%, +7.2° > 5° floor) still FAILS as regression."""
    cand = copy.deepcopy(BASE)
    cand["spatula_red"]["chamfer_mm"] = 25.0     # worst clip fixed
    cand["spatula_red"]["rot_traj_med"] = 40.0   # 32.8 -> 40.0 is +7.2°, exceeds both 20% relative and 5° absolute floor
    ok, why = gate(cand, BASE)
    assert not ok and "spatula_red" in why
