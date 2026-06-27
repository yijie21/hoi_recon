from egoaero import config

def test_defaults_and_override():
    cfg = config.load_config(overrides={"seed": 7, "contact": {"contact_gap_m": 0.001}})
    assert cfg.mock is True
    assert cfg.seed == 7
    assert cfg.contact.contact_gap_m == 0.001        # override merged
    assert cfg.contact.thenar_gap_m == 0.0018        # App C verbatim default kept

def test_method_root_under_egoaero():
    assert config._METHOD_ROOT.replace("\\", "/").endswith("egoaero")
