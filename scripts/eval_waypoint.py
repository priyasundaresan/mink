import argparse
import copy
import numpy as np
import torch
import os
import pyrallis
import mujoco as mj

from envs.mj_env import MujocoEnvConfig, MujocoEnv
from envs.mj_utils.camera_utils import pcl_from_obs
from models.waypoint_transformer import WaypointTransformer
from scipy.spatial.transform import Rotation as R
import common_utils
import open3d as o3d
from contextlib import nullcontext

def eval_waypoint(
    policy: WaypointTransformer,
    env_cfg: MujocoEnvConfig,
    seed: int,
    num_pass: int,
    save_dir,
    record: bool,
    headless: bool,
):
    assert not policy.training
    env = MujocoEnv(env_cfg)
    np.random.seed(seed)
    env.reset()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    reached = True
    terminate = False

    if headless:
        context = nullcontext()
        viewer = None
        os.environ["MUJOCO_GL"] = "egl"
    else:
        viewer = context = mj.viewer.launch_passive(
         model=env.model,
         data=env.data,
         show_left_ui=False,
         show_right_ui=False,
        )
        mj.mjv_defaultFreeCamera(env.model, viewer.cam)

    with context:
        while not terminate and env.num_step < env.max_num_step:
            obs = env.observe()
            recorder.add_numpy(obs, ["viewer_image"])

            points, colors = pcl_from_obs(obs)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            proprio = obs["proprio"]

            if reached:
                with torch.no_grad():
                    _, pos_cmd, rot_cmd, gripper_cmd, mode = policy.inference(
                        torch.from_numpy(points).float(),
                        torch.from_numpy(colors).float(),
                        torch.from_numpy(proprio).float(),
                        num_pass=num_pass,
                    )
                    if policy.cfg.use_euler:
                        euler_cmd = rot_cmd
                    else:
                        euler_cmd = R.from_quat(rot_cmd).as_euler('xyz')

            reached, terminate = env.move_to(pos_cmd, euler_cmd, float(gripper_cmd), viewer, recorder)

    if recorder is not None:
        recorder.save(f"s{seed}", fps=150)

    return env.reward, env.num_step

def _eval_waypoint_multi_episode(
    policy: WaypointTransformer,
    num_pass: int,
    env_cfg: MujocoEnvConfig,
    seed: int,
    num_episode: int,
    save_dir,
    record: int,
):
    scores = []
    num_steps = []
    for seed in range(seed, seed + num_episode):
        score, num_step = eval_waypoint(
            policy,
            env_cfg,
            seed=seed,
            num_pass=num_pass,
            save_dir=save_dir,
            record=record,
        )
        scores.append(score)
        num_steps.append(num_step)
    return np.mean(scores), np.mean(num_steps)


def eval_waypoint_policy(
    policy: WaypointTransformer,
    env_cfg_path: str,
    num_pass: int,
    num_eval_episode: int,
    stat: common_utils.MultiCounter,
    prefix: str = "",
    save_dir = None,
    record: int = 0
):
    assert os.path.exists(env_cfg_path), f"cannot locate env config {env_cfg_path}"
    env_cfg = pyrallis.load(MujocoEnvConfig, open(env_cfg, "r"))
    score, num_step = _eval_waypoint_multi_episode(
        policy, num_pass, env_cfg, 99999, num_eval_episode, save_dir=save_dir, record=record
    )
    stat[f"eval/{prefix}score"].append(score)
    stat[f"eval/{prefix}num_step"].append(num_step)
    return score

def main():
    import os
    import sys
    from scripts.train_waypoint import load_waypoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=20)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='rollouts')
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/cube.yaml")
    parser.add_argument("--headless", action='store_true')
    args = parser.parse_args()

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    print(f">>>>>>>>>>{args.model}<<<<<<<<<<")
    policy = load_waypoint(args.model, device='cuda')
    policy.train(False)
    policy = policy.cuda()

    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    if args.topk > 0:
        print(f"Overriding topk_eval to be {args.topk}")
        policy.cfg.topk_eval = args.topk
    else:
        print(f"Eval with original topk_eval {policy.cfg.topk_eval}")

    scores = []
    for idx, seed in enumerate(range(args.seed, args.seed + args.num_episode)):
        score, num_step = eval_waypoint(
            policy,
            env_cfg,
            seed=seed,
            num_pass=args.num_pass,
            save_dir=args.save_dir,
            record=args.record,
            headless=args.headless,
        )

        scores.append(score)
        print(f"[{idx+1}/{args.num_episode}] avg. score: {np.mean(scores):.4f}, num_steps: {num_step}")
        print(common_utils.wrap_ruler("", max_len=80))

if __name__ == "__main__":
    # python scripts/eval_waypoint.py --model exps/waypoint/cube/ema.pt --env_cfg envs/cfgs/cube.yaml --headless
    main()
