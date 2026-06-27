from egoaero import config

def test_quality_defaults_present():
    q = config.load_config().quality
    assert q.eps_g_m == 0.004 and q.eps_delta_m == 0.012 and q.delta_max_m == 0.015
    assert q.alpha == 1.0 and q.beta == 0.5 and q.gamma == 1.0
    assert q.q_accept == 0.6 and q.q_repairable == 0.3
    assert q.pen_ref_mm == 50000.0 and q.gap_ref_mm == 40.0
    assert q.obj_move_thresh_m_per_frame == 0.01

def test_quality_override_merges():
    q = config.load_config(overrides={"quality": {"q_accept": 0.7}}).quality
    assert q.q_accept == 0.7 and q.q_repairable == 0.3   # other defaults kept
