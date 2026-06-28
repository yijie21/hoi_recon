"""Gymnasium environments for the two-stage residual RL policy.

StageIEnv : track the reconstructed hand reference (reward r^I).

Design contract
---------------
Action space  : Box(-1, 1, (18,)) — 18 finger actuators only; WRJ excluded.
Observation   : float32 (54,) = concat(
                    fq[18]       — finger joint qpos (one per finger actuator)
                    fqd[18]      — finger joint qvel
                    wrist_pos[3] — reference wrist position at current step
                    tips_ref[15] — reference fingertip positions (5 × 3) flattened
                )
Wrist control : MOCAP-driven.  Each step, data.mocap_pos[task.wrist_mocap_id] is
                set to ref["wrist_pos"][t]; orientation is fixed to [1,0,0,0]
                (identity) because the contract carries no wrist rotation.
                See ASSUMPTIONS.md.

Tendon-type actuators (FFJ0/MFJ0/RFJ0/LFJ0 coupling)
------------------------------------------------------
Four finger actuators are tendon-driven (mjtTrn.mjTRN_TENDON).  Their
`actuator_trnid` is a tendon index, not a joint index.  We resolve the first
joint in the tendon wrap to obtain the qpos/qvel address for the observation.

All heavy imports (mujoco, gymnasium) are lazy-imported inside the factory
function so the module is importable without those packages installed.
"""
from __future__ import annotations

import numpy as np

from . import rewards as RW

_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
_FOREARM_BODY = "rh_forearm"

# obs_dim = 18 (fq) + 18 (fqd) + 3 (wrist_pos) + 15 (fingertips_ref) = 54
_N_FINGER_ACT = 18
_OBS_DIM = _N_FINGER_ACT * 2 + 3 + 5 * 3  # 54


def _resolve_qpos_dof_addrs(model, finger_act_ids, mujoco):
    """Return (qpos_addrs, dof_addrs) int arrays aligned with finger_act_ids.

    Joint-type actuators: trnid is the joint id directly.
    Tendon-type actuators: trnid is tendon id; use first joint in wrap.
    """
    _JOINT = int(mujoco.mjtTrn.mjTRN_JOINT)    # 0
    _TENDON = int(mujoco.mjtTrn.mjTRN_TENDON)  # 3

    qpos_addrs = []
    dof_addrs = []
    for act_id in finger_act_ids:
        trntype = int(model.actuator_trntype[act_id])
        trnid = int(model.actuator_trnid[act_id, 0])
        if trntype == _JOINT:
            jnt = trnid
        elif trntype == _TENDON:
            # first wrap in the tendon gives the first coupled joint
            adr = int(model.tendon_adr[trnid])
            jnt = int(model.wrap_objid[adr])
        else:
            jnt = 0  # safe fallback for unknown transmission types
        qpos_addrs.append(int(model.jnt_qposadr[jnt]))
        dof_addrs.append(int(model.jnt_dofadr[jnt]))
    return np.array(qpos_addrs, dtype=int), np.array(dof_addrs, dtype=int)


def StageIEnv(task):  # noqa: N802  (factory disguised as class)
    """Factory that returns a ``gymnasium.Env`` for Stage-I hand tracking.

    Parameters
    ----------
    task : egoaero.policy.task.Task
        Composed task built by ``build_task``.  Must expose:
        ``.model``, ``.ref``, ``.cfg``, ``.finger_act_ids``,
        ``.fingertip_body_ids``, ``.wrist_mocap_id``, ``.obj_qadr``.

    Returns
    -------
    gymnasium.Env  (instance of the inner ``_StageIEnv`` class)
    """
    import gymnasium as gym
    import mujoco as mj

    model = task.model
    ref = task.ref
    cfg_reward = task.cfg["reward"]

    # Pre-compute qpos/qvel index arrays aligned with finger_act_ids
    qpos_addrs, dof_addrs = _resolve_qpos_dof_addrs(model, task.finger_act_ids, mj)

    # Body ID for the wrist reward (forearm tracks the mocap body)
    forearm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _FOREARM_BODY)
    if forearm_bid < 0:
        raise RuntimeError(f"Body '{_FOREARM_BODY}' not found in composed model.")

    n_act = len(task.finger_act_ids)  # should be 18

    class _StageIEnv(gym.Env):
        """Stage-I tracking environment (inner, not meant to be imported directly)."""

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.data = mj.MjData(model)
            self.t = 0
            self._prev_a = np.zeros(n_act, dtype=np.float64)

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(_OBS_DIM,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
            )

        # ------------------------------------------------------------------
        # Private helpers
        # ------------------------------------------------------------------

        def _set_finger_ctrl(self, a: np.ndarray) -> None:
            """Map normalised actions a ∈ [-1,1] → ctrl ranges; write data.ctrl."""
            for i, act_id in enumerate(task.finger_act_ids):
                lo = float(model.actuator_ctrlrange[act_id, 0])
                hi = float(model.actuator_ctrlrange[act_id, 1])
                self.data.ctrl[act_id] = lo + 0.5 * (float(a[i]) + 1.0) * (hi - lo)

        def _set_mocap(self, t_idx: int) -> None:
            """Set mocap wrist pose for frame t_idx (identity orientation)."""
            self.data.mocap_pos[task.wrist_mocap_id] = ref["wrist_pos"][t_idx]
            self.data.mocap_quat[task.wrist_mocap_id] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        def _fq_fqd(self):
            """Return (fq, fqd) arrays of shape (18,) from finger joint addresses."""
            fq = self.data.qpos[qpos_addrs].astype(np.float64)
            fqd = self.data.qvel[dof_addrs].astype(np.float64)
            return fq, fqd

        def _fingertips(self) -> np.ndarray:
            """Return [5,3] fingertip body positions after forward/step."""
            return np.stack(
                [self.data.xpos[task.fingertip_body_ids[f]].copy()
                 for f in _FINGER_ORDER],
                axis=0,
            )

        def _obs(self) -> np.ndarray:
            """Build the 54-dim float32 observation."""
            t = min(self.t, ref["T"] - 1)
            fq, fqd = self._fq_fqd()
            obs = np.concatenate([
                fq,
                fqd,
                ref["wrist_pos"][t],
                ref["fingertips_h"][t].reshape(-1),
            ]).astype(np.float32)
            return obs

        # ------------------------------------------------------------------
        # gymnasium API
        # ------------------------------------------------------------------

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            mj.mj_resetData(model, self.data)

            # Place object at reference initial pose
            qadr = task.obj_qadr
            self.data.qpos[qadr:qadr + 3] = ref["obj_pos"][0]
            obj_quat = np.zeros(4)
            mj.mju_mat2Quat(obj_quat, ref["obj_R"][0].flatten())
            self.data.qpos[qadr + 3:qadr + 7] = obj_quat

            # Set wrist mocap to frame 0 and run kinematics
            self._set_mocap(0)
            mj.mj_forward(model, self.data)

            self.t = 0
            self._prev_a = np.zeros(n_act, dtype=np.float64)
            return self._obs(), {}

        def step(self, action):
            a = np.asarray(action, dtype=np.float64)
            t_idx = min(self.t, ref["T"] - 1)

            # 1. Set mocap BEFORE mj_step (weld constraint picks it up each step)
            self._set_mocap(t_idx)

            # 2. Map action → actuator ctrl
            self._set_finger_ctrl(a)

            # 3. Advance physics
            mj.mj_step(model, self.data)

            # 4. Collect observation components
            fq, fqd = self._fq_fqd()
            tips = self._fingertips()

            # 5. Compute reward terms
            wrist_robot = self.data.xpos[forearm_bid].copy()
            rw = RW.r_wrist(
                wrist_robot, np.eye(3), np.zeros(3),
                ref["wrist_pos"][t_idx], np.eye(3), np.zeros(3),
                float(cfg_reward["lam_p"]),
                float(cfg_reward["lam_R"]),
                float(cfg_reward["lam_v"]),
            )
            rf = RW.r_finger(
                tips, ref["fingertips_h"][t_idx],
                float(cfg_reward["lam_k"]),
            )
            rs = RW.r_smooth(
                a, self._prev_a,
                np.zeros(n_act),   # torque not available without inv-dynamics
                fqd,
                float(cfg_reward["lam_a"]),
                float(cfg_reward["lam_tau"]),
            )
            reward = float(RW.r_stage1(
                rw, rf, rs,
                float(cfg_reward["w_w"]),
                float(cfg_reward["w_f"]),
                float(cfg_reward["w_s"]),
            ))

            self._prev_a = a.copy()
            self.t += 1
            terminated = False
            truncated = bool(self.t >= ref["T"])
            obs = self._obs()
            return obs, reward, terminated, truncated, {}

    return _StageIEnv()
