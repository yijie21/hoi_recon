"""Phase 2 (GL via OSMesa, NO torch): replay a recorded qpos/mocap trajectory through the
MuJoCo Shadow-Hand model and render each step offscreen into a keyframe montage + GIF."""
import os, sys, numpy as np, yaml
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
import mujoco as mj
import imageio.v2 as imageio
sys.path.insert(0, "/workspace/code/hoi_recon/egoaero")
from egoaero.policy.task import build_task

run_dir, rec_npz, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
label = sys.argv[4] if len(sys.argv) > 4 else "sim"
os.makedirs(out_dir, exist_ok=True)
cfg = yaml.safe_load(open("/workspace/code/hoi_recon/egoaero/egoaero/configs/policy.yaml"))
task = build_task(run_dir, "/workspace/code/hoi_recon/egoaero/assets/shadow_hand/right_hand.xml", cfg)
model = task.model
data = mj.MjData(model)

rec = np.load(rec_npz)
qpos, mpos, mquat = rec["qpos"], rec["mocap_pos"], rec["mocap_quat"]
T = len(qpos)

# brighten the scene: bump the headlight (model default is dim) + brighten hand materials
model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]
model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]
for i in range(model.nmat):
    rgba = model.mat_rgba[i]
    if rgba[:3].max() < 0.35:  # lift very dark Shadow-Hand materials toward mid-grey
        model.mat_rgba[i, :3] = np.clip(rgba[:3] + 0.45, 0, 1)

cam = mj.MjvCamera()
cam.distance, cam.azimuth, cam.elevation = 0.5, 144.0, -15.0
opt = mj.MjvOption()
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
W = H = 540
model.vis.global_.offheight = max(int(model.vis.global_.offheight), H)
model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), W)
renderer = mj.Renderer(model, H, W)

tip_ids = list(task.fingertip_body_ids.values())
obj_qadr = task.obj_qadr

frames = []
for t in range(T):
    data.qpos[:] = qpos[t]
    data.qvel[:] = 0
    data.mocap_pos[:] = mpos[t]
    data.mocap_quat[:] = mquat[t]
    mj.mj_forward(model, data)
    # per-frame lookat = midpoint of fingertip centroid and object → keeps both framed
    tip_c = np.mean([data.xpos[b] for b in tip_ids], axis=0)
    obj_c = data.qpos[obj_qadr:obj_qadr + 3]
    cam.lookat[:] = 0.5 * (tip_c + obj_c)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    frames.append(renderer.render().copy())

imageio.mimsave(os.path.join(out_dir, "sim_rollout.gif"), frames, fps=7)
ks = np.linspace(0, T - 1, min(6, T)).astype(int)
strip = np.concatenate([frames[k] for k in ks], axis=1)
imageio.imwrite(os.path.join(out_dir, "sim_keyframes.png"), strip)
print(f"[{label}] {T} steps rendered -> {out_dir}/sim_rollout.gif + sim_keyframes.png  keyframes={list(map(int,ks))}")
