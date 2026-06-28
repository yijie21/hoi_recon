import json, os, yaml
from egoaero.dataset import schema
from egoaero.dataset.collect import run_collection


def _ds_cfg():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(here, "egoaero", "configs", "dataset.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["collection"]["num_frames"] = 12
    cfg["collection"]["max_attempts"] = 6
    return cfg


def test_collection_builds_dataset(tmp_path):
    out = str(tmp_path / "egodexr")
    summary = run_collection(out, n_target=2, dataset_cfg=_ds_cfg(), seed=0,
                             work_root=str(tmp_path / "work"))
    assert os.path.exists(os.path.join(out, "summary.json"))
    # decisions counts sum to attempts; accepted <= attempts
    assert sum(summary["decisions"].values()) == summary["n_attempts"]
    assert summary["n_accepted"] <= summary["n_attempts"]
    assert summary["capabilities"]["contact_eval"] is True
    # every written sequence validates
    for sid in os.listdir(out):
        if sid.startswith("seq_"):
            assert schema.validate_sequence(out, sid)
