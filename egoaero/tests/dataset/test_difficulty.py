from egoaero.dataset.difficulty import difficulty_score

W = {"w_occlusion": 1.0, "w_motion": 1.0, "w_residual": 1.0, "w_contact": 1.0}

def _q(R=0.5, U=0.0):
    return {"R_after": R, "U_unresolved": U}

def test_bounds_1_to_5():
    easy = difficulty_score(_q(R=0.0, U=0.0),
                            {"occlusion": 0.0, "obj_motion_m": 0.0, "contact_richness": 1.0}, W)
    hard = difficulty_score(_q(R=3.0, U=1.0),
                            {"occlusion": 1.0, "obj_motion_m": 0.5, "contact_richness": 0.0}, W)
    assert 1 <= easy <= 5 and 1 <= hard <= 5
    assert easy == 1 and hard == 5

def test_monotonic_in_occlusion():
    lo = difficulty_score(_q(), {"occlusion": 0.1, "obj_motion_m": 0.1, "contact_richness": 0.5}, W)
    hi = difficulty_score(_q(), {"occlusion": 0.9, "obj_motion_m": 0.1, "contact_richness": 0.5}, W)
    assert hi >= lo

def test_richer_contact_not_harder():
    poor = difficulty_score(_q(), {"occlusion": 0.5, "obj_motion_m": 0.2, "contact_richness": 0.0}, W)
    rich = difficulty_score(_q(), {"occlusion": 0.5, "obj_motion_m": 0.2, "contact_richness": 1.0}, W)
    assert rich <= poor
