import numpy as np
from hoi_recon.choir_fine import phases


def test_all_contact_is_manipulation():
    labels = phases.segment_phases(np.ones(10, bool))
    assert (labels == phases.PHASES.index("manipulation")).all()


def test_no_contact_without_motion_is_approach():
    labels = phases.segment_phases(np.zeros(8, bool))
    assert (labels == phases.PHASES.index("approach")).all()


def test_mid_contact_splits_approach_manip_release():
    cp = np.zeros(10, bool); cp[3:7] = True            # contact on frames 3..6
    labels = phases.segment_phases(cp)
    assert (labels[:3] == phases.PHASES.index("approach")).all()
    assert (labels[3:7] == phases.PHASES.index("manipulation")).all()
    assert (labels[7:] == phases.PHASES.index("release")).all()


def test_static_ends_with_motion_signal():
    cp = np.zeros(10, bool); cp[4:6] = True
    motion = np.full(10, 1.0); motion[:2] = 0.0; motion[8:] = 0.0   # static head + tail
    labels = phases.segment_phases(cp, motion=motion, static_thresh=0.1)
    assert (labels[:2] == phases.PHASES.index("pre_static")).all()
    assert (labels[8:] == phases.PHASES.index("post_static")).all()
    assert labels[2] == phases.PHASES.index("approach")            # moving, pre-contact
