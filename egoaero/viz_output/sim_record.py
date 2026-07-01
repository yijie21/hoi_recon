"""Phase 1 (torch, NO GL): roll out StageIIEnv with the trained two-stage policy and
record the full simulator state (qpos + mocap) per step. Decoupled from rendering because
torch + OSMesa segfault when loaded in the same process."""
import os, sys, numpy as np, yaml
sys.path.insert(0, "/workspace/code/hoi_recon/egoaero")
from stable_baselines3 import PPO
from egoaero.policy.task import build_task
from egoaero.policy.env import StageIIEnv

run_dir, out_npz, policy_dir = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
cfg = yaml.safe_load(open("/workspace/code/hoi_recon/egoaero/egoaero/configs/policy.yaml"))
hand_xml = "/workspace/code/hoi_recon/egoaero/assets/shadow_hand/right_hand.xml"
task = build_task(run_dir, hand_xml, cfg)

pi_I = PPO.load(os.path.join(policy_dir, "pi_I.zip"), device="cpu") if policy_dir else None
pi_R = PPO.load(os.path.join(policy_dir, "pi_R.zip"), device="cpu") if policy_dir else None

def _pi_I(obs):
    if pi_I is not None:
        return pi_I.predict(obs, deterministic=True)[0]
    return np.zeros(len(task.finger_act_ids), dtype=np.float32)

env = StageIIEnv(task, _pi_I)
obs, _ = env.reset(seed=0)
qpos, mpos, mquat = [], [], []
done = False
while not done:
    a = pi_R.predict(obs, deterministic=True)[0] if pi_R is not None \
        else np.zeros(env.action_space.shape, dtype=np.float32)
    obs, _, term, trunc, _ = env.step(a)
    qpos.append(env.data.qpos.copy())
    mpos.append(env.data.mocap_pos.copy())
    mquat.append(env.data.mocap_quat.copy())
    done = bool(term or trunc)

np.savez(out_npz,
         qpos=np.array(qpos), mocap_pos=np.array(mpos), mocap_quat=np.array(mquat),
         obj_pos0=task.ref["obj_pos"][0])
print(f"recorded {len(qpos)} steps -> {out_npz}  (policy={'trained' if pi_R is not None else 'zero'})")
