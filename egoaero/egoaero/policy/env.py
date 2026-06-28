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


def StageIIEnv(task, pi_I):  # noqa: N802  (factory disguised as class)
    """Factory returning a ``gymnasium.Env`` for Stage-II residual hand+object control.

    Parameters
    ----------
    task : egoaero.policy.task.Task
        Composed task built by ``build_task``.
    pi_I : callable or SB3-style policy
        Frozen Stage-I policy.  Either a callable ``obs54 -> action18`` (array-like)
        or an object with ``.predict(obs, deterministic=True) -> (action, state)``.

    Returns
    -------
    gymnasium.Env  (instance of the inner ``_StageIIEnv`` class)

    Observation (77-dim float32)
    ----------------------------
    stage1_obs[54] : finger qpos[18] + qvel[18] + ref_wrist_pos[3] + ref_fingertips[15]
    obj_pos[3]     : object centroid position
    obj_quat[4]    : object orientation quaternion (w, x, y, z)
    obj_vel[6]     : object linear + angular velocity (free joint dof)
    hod[5]         : per-finger fingertip-to-object-centre distance
    f_contact[5]   : per-finger contact force magnitude (via cfrc_ext[3:6] linear component)

    Action (18-dim float32)
    -----------------------
    Residual Δa ∈ [-1, 1]^18.  Applied finger ctrl = clip(pi_I(obs54) + Δa, -1, 1).

    Termination / Truncation
    ------------------------
    terminated : object position error vs reference > cfg["term"]["obj_pos_err_m"]
    truncated  : t >= T
    """
    import gymnasium as gym
    import mujoco as mj

    model = task.model
    ref = task.ref
    cfg_reward = task.cfg["reward"]
    cfg_term = task.cfg["term"]

    # Pre-compute qpos/qvel index arrays for the 18 finger actuators
    qpos_addrs, dof_addrs = _resolve_qpos_dof_addrs(model, task.finger_act_ids, mj)

    n_act = len(task.finger_act_ids)  # 18

    # Resolve the object free-joint dof address once
    _obj_jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "obj_free")
    if _obj_jid < 0:
        raise RuntimeError("Joint 'obj_free' not found in composed model.")
    _obj_dadr = int(model.jnt_dofadr[_obj_jid])

    # Body ID for the wrist reward (forearm body)
    forearm_bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, _FOREARM_BODY)
    if forearm_bid < 0:
        raise RuntimeError(f"Body '{_FOREARM_BODY}' not found in composed model.")

    # Stage-I obs dim = 18 qpos + 18 qvel + 3 wrist + 15 fingertips
    _S1_DIM = _OBS_DIM   # 54
    # Stage-II obs = stage1(54) + obj_pos(3) + obj_quat(4) + obj_vel(6) + hod(5) + f(5)
    _S2_DIM = _S1_DIM + 7 + 6 + 5 + 5  # 77

    def _base_action(pi, obs):
        """Query pi_I, handling callable or SB3-style policy."""
        if hasattr(pi, "predict"):
            act, _ = pi.predict(obs, deterministic=True)
        else:
            act = pi(obs)
        return np.asarray(act, dtype=np.float64)

    class _StageIIEnv(gym.Env):
        """Stage-II residual environment (inner class)."""

        metadata = {"render_modes": []}

        def __init__(self):
            super().__init__()
            self.data = mj.MjData(model)
            self.t = 0
            self._prev_a = np.zeros(n_act, dtype=np.float64)

            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(_S2_DIM,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32
            )

        # ------------------------------------------------------------------
        # Private helpers
        # ------------------------------------------------------------------

        def _set_finger_ctrl(self, a: np.ndarray) -> None:
            """Map normalised finger actions a ∈ [-1,1]^18 → ctrl ranges."""
            for i, act_id in enumerate(task.finger_act_ids):
                lo = float(model.actuator_ctrlrange[act_id, 0])
                hi = float(model.actuator_ctrlrange[act_id, 1])
                self.data.ctrl[act_id] = lo + 0.5 * (float(a[i]) + 1.0) * (hi - lo)

        def _set_mocap(self, t_idx: int) -> None:
            self.data.mocap_pos[task.wrist_mocap_id] = ref["wrist_pos"][t_idx]
            self.data.mocap_quat[task.wrist_mocap_id] = np.array(
                [1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )

        def _fq_fqd(self):
            fq = self.data.qpos[qpos_addrs].astype(np.float64)
            fqd = self.data.qvel[dof_addrs].astype(np.float64)
            return fq, fqd

        def _fingertips(self) -> np.ndarray:
            return np.stack(
                [self.data.xpos[task.fingertip_body_ids[f]].copy()
                 for f in _FINGER_ORDER],
                axis=0,
            )

        def _stage1_obs(self) -> np.ndarray:
            """Build the 54-dim Stage-I observation (same contract as StageIEnv)."""
            t = min(self.t, ref["T"] - 1)
            fq, fqd = self._fq_fqd()
            return np.concatenate([
                fq,
                fqd,
                ref["wrist_pos"][t],
                ref["fingertips_h"][t].reshape(-1),
            ]).astype(np.float32)

        def _obj_pose(self):
            """Return (pos[3], quat[4]) of the object from qpos."""
            qadr = task.obj_qadr
            pos = self.data.qpos[qadr: qadr + 3].copy()
            quat = self.data.qpos[qadr + 3: qadr + 7].copy()
            return pos, quat

        def _obj_vel(self) -> np.ndarray:
            """Return 6-dim free-joint velocity [lin_vel(3), ang_vel(3)]."""
            return self.data.qvel[_obj_dadr: _obj_dadr + 6].copy()

        def _contact_forces(self) -> np.ndarray:
            """Per-finger contact force magnitude via body cfrc_ext[3:6] (linear force).

            NOTE: cfrc_ext[bid] is a 6-vector [torque(3), force(3)] in world frame.
            We use the linear force half [3:6] as the per-fingertip contact magnitude.
            See ASSUMPTIONS.md.
            """
            f = np.zeros(5, dtype=np.float64)
            for i, fg in enumerate(_FINGER_ORDER):
                bid = task.fingertip_body_ids[fg]
                f[i] = float(np.linalg.norm(self.data.cfrc_ext[bid][3:6]))
            return f

        def _obs(self) -> np.ndarray:
            """Build the full 77-dim float32 Stage-II observation."""
            t = min(self.t, ref["T"] - 1)
            s1 = self._stage1_obs()
            pos, quat = self._obj_pose()
            obj_vel = self._obj_vel()
            tips = self._fingertips()
            hod = np.linalg.norm(tips - pos[None, :], axis=1)
            f_contact = self._contact_forces()
            return np.concatenate([
                s1, pos, quat, obj_vel, hod, f_contact
            ]).astype(np.float32)

        # ------------------------------------------------------------------
        # gymnasium API
        # ------------------------------------------------------------------

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            mj.mj_resetData(model, self.data)

            # Place object at reference initial pose
            qadr = task.obj_qadr
            self.data.qpos[qadr: qadr + 3] = ref["obj_pos"][0]
            obj_quat = np.zeros(4)
            mj.mju_mat2Quat(obj_quat, ref["obj_R"][0].flatten())
            self.data.qpos[qadr + 3: qadr + 7] = obj_quat

            # Set wrist mocap and run forward kinematics
            self._set_mocap(0)
            mj.mj_forward(model, self.data)

            self.t = 0
            self._prev_a = np.zeros(n_act, dtype=np.float64)
            return self._obs(), {}

        def step(self, action):
            dadelta = np.asarray(action, dtype=np.float64)
            t_idx = min(self.t, ref["T"] - 1)

            # 1. Query Stage-I policy and compute combined action
            s1_obs = self._stage1_obs()
            base = _base_action(pi_I, s1_obs)
            a = np.clip(base + dadelta, -1.0, 1.0)

            # 2. Set mocap BEFORE mj_step
            self._set_mocap(t_idx)

            # 3. Map action → actuator ctrl and advance physics
            self._set_finger_ctrl(a)
            mj.mj_step(model, self.data)

            # 4. Collect post-step quantities
            fq, fqd = self._fq_fqd()
            tips = self._fingertips()
            pos, quat = self._obj_pose()

            # Rotation matrix from quaternion for r_obj
            Rm = np.zeros(9)
            mj.mju_quat2Mat(Rm, quat)
            obj_R_sim = Rm.reshape(3, 3)

            obj_vel6 = self._obj_vel()
            obj_lin_vel = obj_vel6[:3]
            f_contact = self._contact_forces()
            hod = np.linalg.norm(tips - pos[None, :], axis=1)

            # 5. Compute Stage-I sub-reward (same calc as StageIEnv)
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
                np.zeros(n_act),
                fqd,
                float(cfg_reward["lam_a"]),
                float(cfg_reward["lam_tau"]),
            )
            r1 = RW.r_stage1(
                rw, rf, rs,
                float(cfg_reward["w_w"]),
                float(cfg_reward["w_f"]),
                float(cfg_reward["w_s"]),
            )

            # 6. Object pose reward
            r_obj = RW.r_obj(
                pos, obj_R_sim, obj_lin_vel,
                ref["obj_pos"][t_idx], ref["obj_R"][t_idx], np.zeros(3),
                float(cfg_reward["mu_p"]),
                float(cfg_reward["mu_R"]),
                float(cfg_reward["mu_v"]),
            )

            # 7. Contact reward
            r_contact = RW.r_contact(
                hod, f_contact,
                ref["contact_active"][t_idx],
                float(cfg_reward["mu_d"]),
                float(cfg_reward["mu_F"]),
            )

            # 8. Residual penalty
            r_res = RW.r_res(dadelta, float(cfg_reward["mu_delta"]))

            # 9. Combined Stage-II reward
            reward = float(RW.r_stage2(
                r1, r_obj, r_contact, r_res,
                float(cfg_reward["eta_I"]),
                float(cfg_reward["eta_o"]),
                float(cfg_reward["eta_c"]),
                float(cfg_reward["eta_delta"]),
            ))

            self._prev_a = a.copy()
            self.t += 1

            # 10. Termination / truncation
            obj_pos_err = float(np.linalg.norm(pos - ref["obj_pos"][t_idx]))
            terminated = bool(obj_pos_err > float(cfg_term["obj_pos_err_m"]))
            truncated = bool(self.t >= ref["T"])

            obs = self._obs()
            return obs, reward, terminated, truncated, {}

    return _StageIIEnv()
