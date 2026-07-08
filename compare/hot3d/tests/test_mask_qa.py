import numpy as np, os, sys, tempfile
sys.path.insert(0, "/workspace/code/hoi_recon/render_and_compare")
from hoi_recon.mask_qa import qa_report

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
