"""Shadow Hand MuJoCo wrapper: forward kinematics via distal fingertip bodies,
mapped to the MANO finger order.  Documented substitute for the Inspire Hand
(App-D/G).

The vendored ``right_hand.xml`` has only one site (``grasp_site``); there are no
fingertip sites.  We therefore resolve fingertips through the distal **body**
chain and read ``data.xpos`` after ``mj_forward``.

Fingertip body names verified against the vendored asset:
  thumb  → rh_thdistal  (body 25)
  index  → rh_ffdistal  (body  7)
  middle → rh_mfdistal  (body 11)
  ring   → rh_rfdistal  (body 15)
  little → rh_lfdistal  (body 20)
"""
from __future__ import annotations

import numpy as np

_FINGERTIP_BODY_NAMES: dict[str, str] = {
    "thumb": "rh_thdistal",
    "index": "rh_ffdistal",
    "middle": "rh_mfdistal",
    "ring": "rh_rfdistal",
    "little": "rh_lfdistal",
}

_MANO_ORDER = ["thumb", "index", "middle", "ring", "little"]


class HandModel:
    """Load a Shadow Hand MJCF and expose forward kinematics of the 5 fingertips.

    Parameters
    ----------
    xml_path:
        Path to ``right_hand.xml`` (or equivalent MJCF).
    """

    def __init__(self, xml_path: str) -> None:
        import mujoco  # lazy — mujoco is optional at module-import time

        self._mj = mujoco

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Number of actuators (Shadow Hand: 20)
        self.n_act: int = int(self.model.nu)

        # Per-actuator qpos address of its transmission joint
        adr: list[int] = []
        for i in range(self.model.nu):
            j = int(self.model.actuator_trnid[i, 0])
            adr.append(int(self.model.jnt_qposadr[j]))
        self.actuated_joint_qpos_adr: np.ndarray = np.array(adr, dtype=int)

        # Resolve the 5 fingertip body IDs by name
        self.fingertip_body_ids: dict[str, int] = {}
        for finger, body_name in _FINGERTIP_BODY_NAMES.items():
            bid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if bid < 0:
                raise RuntimeError(
                    f"Fingertip body '{body_name}' (finger '{finger}') not found "
                    f"in {xml_path}.  Inspect body names and update "
                    f"_FINGERTIP_BODY_NAMES."
                )
            self.fingertip_body_ids[finger] = int(bid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fk_fingertips(self, q: np.ndarray) -> np.ndarray:
        """Return the 5 fingertip world positions given actuator joint angles.

        Parameters
        ----------
        q:
            Actuator joint angles, shape ``(n_act,)`` or longer (extras ignored).

        Returns
        -------
        np.ndarray of shape ``(5, 3)`` — fingertip XYZ positions in MANO order
        [thumb, index, middle, ring, little].
        """
        q = np.asarray(q, dtype=float)
        self.data.qpos[self.actuated_joint_qpos_adr] = q[: self.n_act]
        self._mj.mj_forward(self.model, self.data)
        return np.stack(
            [self.data.xpos[self.fingertip_body_ids[f]].copy() for f in _MANO_ORDER],
            axis=0,
        )
