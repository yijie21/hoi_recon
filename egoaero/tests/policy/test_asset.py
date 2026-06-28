import os, pytest

def test_shadow_hand_mjcf_loads():
    mujoco = pytest.importorskip("mujoco")
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    xml = os.path.join(here, "assets", "shadow_hand", "right_hand.xml")
    if not os.path.exists(xml):
        pytest.skip("shadow hand not vendored; run scripts/setup_rl.sh")
    model = mujoco.MjModel.from_xml_path(xml)
    assert model.nu > 0          # has actuators
    assert model.nsite > 0       # has sites (fingertips)
