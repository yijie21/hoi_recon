import math
from egoaero import quality as Q


def test_quality_score_monotonic_and_bounded():
    base = Q.quality_score(0.2, 0.2, 0.0, 1.0, 0.5, 1.0)
    assert 0 < base <= 1.0
    assert Q.quality_score(0.5, 0.2, 0.0, 1.0, 0.5, 1.0) < base   # higher R -> lower Q
    assert Q.quality_score(0.2, 0.9, 0.0, 1.0, 0.5, 1.0) < base   # higher B -> lower Q
    assert Q.quality_score(0.2, 0.2, 0.5, 1.0, 0.5, 1.0) < base   # higher U -> lower Q
    assert Q.quality_score(0.0, 0.0, 0.0, 1.0, 0.5, 1.0) == 1.0


def test_decision_thresholds():
    rec = {"thumb": 0.9, "index": 0.8, "middle": 0.1, "ring": 0.0, "little": 0.0}
    lab_a, info_a = Q.decision(0.7, rec, q_accept=0.6, q_repairable=0.3)
    lab_r, _ = Q.decision(0.45, rec, q_accept=0.6, q_repairable=0.3)
    lab_x, _ = Q.decision(0.1, rec, q_accept=0.6, q_repairable=0.3)
    assert lab_a == "accept" and lab_r == "repairable_accept" and lab_x == "recapture"
    assert set(info_a["low_recoverability_fingers"]) == {"middle", "ring", "little"}
