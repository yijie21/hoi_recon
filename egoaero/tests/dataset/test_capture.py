from egoaero.dataset.capture import synthetic_source, clip_overrides


def test_source_yields_n_with_tightness_spread():
    clips = synthetic_source(n=5, seed=0, num_frames=24, tightness_min=0.0, tightness_max=1.0)
    assert len(clips) == 5
    ts = [c["mock_tightness"] for c in clips]
    assert min(ts) == 0.0 and max(ts) == 1.0 and ts == sorted(ts)
    assert all("task_description" in c and "seed" in c for c in clips)


def test_clip_overrides_keys():
    c = synthetic_source(2, 0, 16, 0.0, 1.0)[1]
    ov = clip_overrides(c)
    assert ov["mock_tightness"] == c["mock_tightness"]
    assert ov["num_frames"] == c["num_frames"] and ov["seed"] == c["seed"]
    assert ov["mock"] is True
