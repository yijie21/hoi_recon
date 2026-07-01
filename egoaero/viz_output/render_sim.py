"""Render the SP2 simulation: the reconstructed hand-object trajectory retargeted to a
simulated Shadow Hand in MuJoCo. Rolls out StageIIEnv (mocap wrist follows the reconstructed
human-hand trajectory; fingers driven by the two-stage policy or zero) and renders each step
offscreen (OSMesa) into a keyframe montage + animated GIF. This is the right panel of Fig. 1."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import numpy as np
import mujoco as mj
import imageio.v2 as imageio

sys.path.insert(0, "/workspace/code/hoi_recon/egoaero")
from egoaero.policy.task import build_task
from egoaero.policy.env import StageIIEnv

import yaml


def load_policy(path):
    if path and os.path.exists(path):
        from stable_baselines3 import PPO
        return PPO.load(path, device="cpu")
    return None


def main(run_dir, out_dir, policy_dir=None, W=480, H=480):
    os.makedirs(out_dir, exist_ok=True)
    with open("/workspace/code/hoi_recon/egoaero/egoaero/configs/policy.yaml") as f:
        cfg = yaml.safe_load(f)
    hand_xml = "/workspace/code/hoi_recon/egoaero/assets/shadow_hand/right_hand.xml"
    task = build_task(run_dir, hand_xml, cfg)
    model = task.model

    pi_I = load_policy(os.path.join(policy_dir, "pi_I.zip")) if policy_dir else None
    pi_R = load_policy(os.path.join(policy_dir, "pi_R.zip")) if policy_dir else None
    tag = "trained-policy" if pi_R is not None else "zero-policy"
    print(f"[sim] run={os.path.basename(run_dir.rstrip('/'))}  policy={tag}")

    if pi_I is not None:
        def _pi_I(obs):
            return pi_I.predict(obs, deterministic=True)[0]
    else:
        def _pi_I(obs):
            return np.zeros(len(task.finger_act_ids), dtype=np.float32)

    env = StageIIEnv(task, _pi_I)
    obs, _ = env.reset(seed=0)

    # camera framed on the object's initial position
    ref = task.ref
    lookat = ref["obj_pos"][0].astype(np.float64)
    cam = mj.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance = 0.45
    cam.azimuth = 140.0
    cam.elevation = -20.0

    renderer = mj.Renderer(model, H, W)
    opt = mj.MjvOption()
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    frames = []
    done = False
    while not done:
        a = pi_R.predict(obs, deterministic=True)[0] if pi_R is not None \
            else np.zeros(env.action_space.shape, dtype=np.float32)
        obs, _, term, trunc, _ = env.step(a)
        renderer.update_scene(env.data, camera=cam, scene_option=opt)
        frames.append(renderer.render().copy())
        done = bool(term or trunc)

    T = len(frames)
    print(f"[sim] rolled {T} steps, {W}x{H}")

    # GIF
    gif = os.path.join(out_dir, "sim_rollout.gif")
    imageio.mimsave(gif, frames, fps=7)
    print("wrote", gif)

    # keyframe montage
    ks = np.linspace(0, T - 1, min(6, T)).astype(int)
    strip = np.concatenate([frames[k] for k in ks], axis=1)
    png = os.path.join(out_dir, "sim_keyframes.png")
    imageio.imwrite(png, strip)
    print("wrote", png, "frames", list(ks))


if __name__ == "__main__":
    run_dir = sys.argv[1]
    out_dir = sys.argv[2]
    policy_dir = sys.argv[3] if len(sys.argv) > 3 else None
    main(run_dir, out_dir, policy_dir)
