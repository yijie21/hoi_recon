import numpy as np
from egoaero.bundle import Bundle

def test_save_load_roundtrip(tmp_path):
    b = Bundle(arrays={"x": np.arange(6).reshape(2, 3)}, meta={"n": 2}, assets={"mesh": "m.obj"})
    b.save(str(tmp_path))
    assert Bundle.exists(str(tmp_path))
    b2 = Bundle.load(str(tmp_path))
    assert np.array_equal(b2["x"], b["x"]) and b2.meta["n"] == 2 and b2.assets["mesh"] == "m.obj"
