"""Two-stage residual policy training with stable-baselines3 PPO.

train_two_stage(task, policy_cfg, budget, out_dir, seed)
---------------------------------------------------------
Stage I  : PPO on StageIEnv  for ``policy_cfg["budget"][budget]`` total steps.
Stage II : freeze pi_I → PPO on StageIIEnv for the same number of steps.
           pi_I is wrapped as a callable ``obs -> action`` so StageIIEnv can use
           it regardless of whether it is an SB3 model or a plain function.

n_steps / batch_size guards
----------------------------
SB3 PPO requires  batch_size <= n_steps * n_envs  (n_envs=1 here).
At smoke budget (steps < default n_steps) we cap n_steps to the step budget
and clamp batch_size ≤ n_steps so SB3 does not raise.
"""
from __future__ import annotations
import os


def train_two_stage(task, policy_cfg, budget="smoke", out_dir=None, seed=0):
    """Run two-stage PPO and return (pi_I, pi_R).

    Parameters
    ----------
    task       : egoaero.policy.task.Task — will have task.cfg overwritten with policy_cfg
    policy_cfg : dict — loaded policy.yaml; must contain ``budget``, ``ppo``, ``reward``,
                 ``term`` sections.
    budget     : str  — key into ``policy_cfg["budget"]`` (e.g. ``"smoke"`` or ``"real"``).
    out_dir    : str | None — if given, save pi_I.zip and pi_R.zip there.
    seed       : int — random seed passed to SB3 PPO.

    Returns
    -------
    (pi_I, pi_R)  — two stable_baselines3.PPO instances.
    """
    from stable_baselines3 import PPO
    from .env import StageIEnv, StageIIEnv

    # Overwrite task config so both env factories see reward / term from policy_cfg
    task.cfg = policy_cfg

    steps = int(policy_cfg["budget"][budget])
    ppo_cfg = policy_cfg["ppo"]

    # Cap n_steps so smoke budget (steps < default 2048) still works
    n_steps = min(int(ppo_cfg["n_steps"]), max(64, steps))
    # batch_size must be <= n_steps (SB3 assertion: batch_size <= n_steps * n_envs)
    batch_size = min(int(ppo_cfg["batch_size"]), n_steps)

    common_kw = dict(
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=float(ppo_cfg["learning_rate"]),
        seed=seed,
        policy_kwargs={"net_arch": list(ppo_cfg["net_arch"])},
        verbose=0,
    )

    # ------------------------------------------------------------------
    # Stage I — track the reference hand motion
    # ------------------------------------------------------------------
    env1 = StageIEnv(task)
    pi_I = PPO(ppo_cfg["policy"], env1, **common_kw)
    pi_I.learn(total_timesteps=steps)

    # ------------------------------------------------------------------
    # Stage II — residual correction with frozen pi_I
    # ------------------------------------------------------------------
    # Wrap pi_I as a callable so StageIIEnv's _base_action helper (which
    # checks for a .predict attribute) routes to the SB3 model.
    frozen = lambda obs: pi_I.predict(obs, deterministic=True)[0]  # noqa: E731

    env2 = StageIIEnv(task, frozen)
    pi_R = PPO(ppo_cfg["policy"], env2, **common_kw)
    pi_R.learn(total_timesteps=steps)

    # ------------------------------------------------------------------
    # Persist
    # ------------------------------------------------------------------
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pi_I.save(os.path.join(out_dir, "pi_I"))
        pi_R.save(os.path.join(out_dir, "pi_R"))

    return pi_I, pi_R
