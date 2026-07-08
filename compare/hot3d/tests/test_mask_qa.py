import numpy as np, os, sys, tempfile
sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.mask_qa import (qa_report, reprompt_point, score_track,
                               select_track)

def _save(dirn, masks):
    ps = []
    for i, m in enumerate(masks):
        p = os.path.join(dirn, f"{i:05d}.npy"); np.save(p, m); ps.append(p)
    return ps

def test_stable_masks_pass():
    with tempfile.TemporaryDirectory() as d:
        m = np.zeros((64, 64), bool); m[20:40, 20:40] = True
        ps = _save(d, [m] * 10)
        r = qa_report(ps, None, None)
        assert not r["bad"] and r["best_frame"] in range(10)

def test_area_explosion_flags_bad():
    with tempfile.TemporaryDirectory() as d:
        small = np.zeros((64, 64), bool); small[20:40, 20:40] = True
        big = np.zeros((64, 64), bool); big[5:60, 5:60] = True   # 7.5x area
        ps = _save(d, [small] * 5 + [big] * 5)
        r = qa_report(ps, None, None)
        assert r["bad"] and r["best_frame"] < 5     # anchor from stable phase

def test_hand_overlap_flags_bad():
    with tempfile.TemporaryDirectory() as d:
        m = np.zeros((64, 64), bool); m[20:50, 20:50] = True
        ps = _save(d, [m] * 6)
        hb = np.tile(np.array([[[18., 18., 52., 52.], [np.nan]*4]]), (6, 1, 1))
        hv = np.tile(np.array([[True, False]]), (6, 1))
        r = qa_report(ps, hb, hv)
        assert r["hand_overlap"].mean() > 0.9 and r["bad"]

def test_reprompt_point_prefers_interior_component():
    # L-shaped forearm blob entering from the border (bigger) + small interior
    # object blob: the re-prompt point must land on the interior blob.
    m = np.zeros((64, 64), bool)
    m[0:50, 0:10] = True      # arm: vertical bar from the top border
    m[40:50, 0:30] = True     # arm: horizontal bar (L shape), ~800 px total
    m[20:30, 45:55] = True    # object: 10x10 interior blob, 100 px
    (x, y), case = reprompt_point(m)
    assert case == "interior"
    assert 45 <= x < 55 and 20 <= y < 30

def test_reprompt_point_falls_back_to_largest():
    # Everything touches the border: fall back to the largest component.
    m = np.zeros((64, 64), bool)
    m[0:30, 0:10] = True      # border blob A (300 px)
    m[60:64, 40:64] = True    # border blob B (96 px)
    (x, y), case = reprompt_point(m)
    assert case == "border-fallback"
    assert x < 10 and y < 30  # centroid of the larger border blob

def test_score_track_prefers_stable_low_overlap():
    T = 12
    stable = score_track([400.0] * T, [0.9] * (T - 1), [0.05] * T, 0.0)
    # (i) dying track: mask vanishes after 4 frames, tIoU collapses
    dying = score_track([400.0] * 4 + [0.0] * (T - 4),
                        [0.9] * 3 + [0.0] * (T - 4), [0.05] * T, 0.0)
    # (ii) border-hugging arm track: temporally stable but high hand overlap
    # and touching the frame border in every frame
    arm = score_track([2000.0] * T, [0.92] * (T - 1), [0.6] * T, 1.0)
    assert stable > dying
    assert stable > arm

def test_score_track_penalizes_area_jump():
    T = 10
    stable = score_track([400.0] * T, [0.9] * (T - 1), [0.05] * T, 0.0)
    # leak onset: area explodes mid-track (tIoU dips only at the jump)
    leaky = score_track([400.0] * 5 + [3000.0] * 5,
                        [0.9] * 4 + [0.12] + [0.9] * 4, [0.05] * T, 0.0)
    assert stable > leaky

def test_select_track_four_bench_cases():
    # The four observed HOT3D bench cases (v3/v3.1/v3.1b screens):
    # masher — original not bad, offset only marginally better (delta 0.134
    # < 0.15): the original-click preference fires and keeps the original.
    idx, fired = select_track([0.145, 0.279], [False, False], original_idx=0)
    assert idx == 0 and fired
    # mug — original click landed on the hand: its track is bad, the offset
    # candidate wins and the tiebreaker must NOT fire.
    idx, fired = select_track([0.10, 0.279], [True, False], original_idx=0)
    assert idx == 1 and not fired
    # spatula — offset clearly better (delta 0.826 >= margin): offset wins.
    idx, fired = select_track([0.10, 0.926], [False, False], original_idx=0)
    assert idx == 1 and not fired
    # cube (puzzle_toy, both hands) — every track bad: no selection, caller
    # falls back to the vanilla single track.
    idx, fired = select_track([0.05, -0.3, -0.5], [True, True, True],
                              original_idx=0)
    assert idx is None and not fired

def test_select_track_original_vetoed():
    # Original click discarded by the hand-mask veto (not among candidates):
    # plain argmax, tiebreaker cannot fire.
    idx, fired = select_track([0.20, 0.25], [False, False], original_idx=None)
    assert idx == 1 and not fired
