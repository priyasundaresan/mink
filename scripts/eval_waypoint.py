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

def eval_waypoint(
    policy: WaypointTransformer,
    env_cfg: str,
    seed: int,
    num_pass: int,
    save_dir,
    record: bool,
):
    assert not policy.training
    env_cfg = pyrallis.load(MujocoEnvConfig, open(env_cfg, "r"))
    env = MujocoEnv(env_cfg)
    np.random.seed(seed)
    env.reset()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    freeze_counter = 0


    with mj.viewer.launch_passive(
        model=env.model,
        data=env.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
    
        reached = True
        terminate = False
        mj.mjv_defaultFreeCamera(env.model, viewer.cam)
    
        while viewer.is_running() and not terminate:
            obs = env.observe()
            recorder.add_numpy(obs, ["viewer_image"])

            points, colors = pcl_from_obs(obs)
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(points)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud('test.pcd', point_cloud)
            proprio = obs["proprio"]

            if reached:
                with torch.no_grad():
                    _, pos_cmd, euler_cmd, gripper_cmd, _ = policy.inference(
                        torch.from_numpy(points).float(),
                        torch.from_numpy(colors).float(),
                        torch.from_numpy(proprio).float(),
                        num_pass=num_pass,
                    )

            reached, terminate = env.move_to(pos_cmd, euler_cmd, float(gripper_cmd), viewer, recorder)
            
    if recorder is not None:
        recorder.save(f"s{seed}", fps=150)
    return 

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
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/mj_env.yaml")
    args = parser.parse_args()

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    print(f">>>>>>>>>>{args.model}<<<<<<<<<<")
    policy = load_waypoint(args.model, device='cuda')
    policy.train(False)
    policy = policy.cuda()

    scores = []
    for idx, seed in enumerate(range(args.seed, args.seed + args.num_episode)):
        eval_waypoint(
            policy,
            args.env_cfg,
            seed=seed,
            num_pass=args.num_pass,
            save_dir=args.save_dir,
            record=args.record,
        )
        print(common_utils.wrap_ruler("", max_len=80))

if __name__ == "__main__":
    # python scripts/eval_waypoint.py --model exps/waypoint/run1/ema.pt --record 1 --save_dir rollouts
    main()
