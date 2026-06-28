"""Test HandModel — Shadow Hand MuJoCo wrapper (body-based FK)."""
import os
import numpy as np
import pytest


def _xml():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(here, "assets", "shadow_hand", "right_hand.xml")


def test_hand_model_fk_and_body_ids():
    pytest.importorskip("mujoco")
    if not os.path.exists(_xml()):
        pytest.skip("shadow hand not vendored")
    from egoaero.policy.hand_model import HandModel

    hm = HandModel(_xml())

    # body id dict has exactly the 5 MANO fingers
    assert set(hm.fingertip_body_ids) == {"thumb", "index", "middle", "ring", "little"}

    # all ids are non-negative integers
    for name, bid in hm.fingertip_body_ids.items():
        assert isinstance(bid, int) and bid >= 0, f"bad body id for {name}: {bid}"

    # model attributes
    assert hm.n_act == 20
    assert hm.actuated_joint_qpos_adr.shape == (20,)

    # FK returns (5, 3) array of distinct positions
    tips = hm.fk_fingertips(np.zeros(hm.n_act))
    assert tips.shape == (5, 3), f"expected (5,3), got {tips.shape}"

    # positions are finite
    assert np.all(np.isfinite(tips)), "some fingertip positions are non-finite"

    # all 5 positions are distinct (no two fingertips coincide)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.allclose(tips[i], tips[j]), (
                f"fingertips {i} and {j} coincide: {tips[i]}"
            )
