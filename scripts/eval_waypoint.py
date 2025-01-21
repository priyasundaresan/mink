import argparse
import copy
import numpy as np
import torch
import os
import pyrallis

from envs.mj_env import MujocoEnvConfig, MujocoEnv
from envs.mj_utils.camera_utils import pcl_from_obs
from models.waypoint_transformer import WaypointTransformer
from scipy.spatial.transform import Rotation as R
import common_utils

@torch.no_grad
def eval_waypoint(
    policy: WaypointTransformer,
    seed: int,
    num_pass: int,
    save_dir,
    record: bool,
):
    assert not policy.training

    env = MujocoEnv(env_cfg)
    np.random.seed(seed)
    env.reset()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    freeze_counter = 0
    while True:
        obs = env.observe()

        points, colors = pcl_from_obs(obs)
        proprio = obs["proprio"]

        _, pos_cmd, euler_cmd, gripper_cmd, _ = policy.inference(
            torch.from_numpy(points).float(),
            torch.from_numpy(colors).float(),
            torch.from_numpy(proprio).float(),
            num_pass=num_pass,
        )

        action = {'base_pose': np.zeros(3), \
                  'arm_pos': pos_cmd, \
                  'arm_quat': R.from_euler('xyz', euler_cmd).as_quat()}
        gripper_action = float(gripper_cmd)

        env.step(action, gripper_state)

    return 

def main():
    import os
    import sys
    from scripts.train_waypoint import load_waypoint

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=10)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--record", type=int, default=0)
    args = parser.parse_args()

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    print(f">>>>>>>>>>{model}<<<<<<<<<<")
    policy = load_waypoint(args.model)
    policy.train(False)
    policy = policy.cuda()

    scores = []
    for seed in range(args.seed, args.seed + args.num_episode):
        eval_waypoint(
            policy,
            seed=seed,
            num_pass=args.num_pass,
            save_dir=args.save_dir,
            record=args.record,
        )

        print(f"{model}")
        print(common_utils.wrap_ruler("", max_len=80))

if __name__ == "__main__":
    main()
