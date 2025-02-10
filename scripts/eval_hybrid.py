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
from interactive_scripts.dataset_recorder import ActMode

import os
import sys
from scripts.train_waypoint import load_waypoint
from scripts.train_dense import load_model
from common_utils.eval_utils import (
    check_for_interrupt,
)

def eval_hybrid(
    waypoint_policy,
    dense_policy,
    dense_dataset,
    env_cfg,
    seed,
    num_pass,
    save_dir,
    record,
):
    assert not waypoint_policy.training
    assert not dense_policy.training
    env_cfg = pyrallis.load(MujocoEnvConfig, open(env_cfg, "r"))
    env = MujocoEnv(env_cfg)
    np.random.seed(seed)
    env.reset()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    with mj.viewer.launch_passive(
        model=env.model,
        data=env.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
    
        reached = True
        mode = ActMode.Waypoint.value
        mj.mjv_defaultFreeCamera(env.model, viewer.cam)

        mode = ActMode.Waypoint.value
        while viewer.is_running():
            if mode == ActMode.Waypoint.value: 
                mode = run_waypoint_mode(env, waypoint_policy, num_pass, viewer, recorder)
            elif mode == ActMode.Dense.value:
                mode = run_dense_mode(env, dense_dataset, dense_policy, viewer, recorder)
            elif mode == ActMode.Terminate.value:
                print('Terminate')
                if recorder is not None:
                    recorder.save(f"s{seed}", fps=150)
                return
    return 

def run_dense_mode(env, dense_dataset, dense_policy, viewer, recorder):
    cached_actions = []
    mode = ActMode.Dense.value
    image_capture_interval = int(env.frequency / 10)  # Capture images every 10Hz
    step_counter = 0  # Counter to track simulation steps
    action = None
    gripper_state = 1 #TODO

    consecutive_modes_required = 5 
    WAYPOINT_THRESH = 0.5
    TERMINATE_THRESH = 1.3
    mode_history = []

    while mode == ActMode.Dense.value:
        if step_counter % image_capture_interval == 0:
            obs = env.observe()
    
            dense_obs = dense_dataset.process_observation(obs)
            for k, v in dense_obs.items():
                dense_obs[k] = v.cuda()
    
            if len(cached_actions) == 0:
                action_seq = dense_policy.act(dense_obs)
    
                for action in action_seq.split(1, dim=0):
                    cached_actions.append(action.squeeze(0))

            action = cached_actions.pop(0)

            ee_pos, ee_quat, gripper_open, raw_mode = action.split([3, 4, 1, 1])
            ee_quat = ee_quat.detach().cpu().numpy()
            ee_quat /= np.linalg.norm(ee_quat)
            print(ee_quat)
            ee_pos = ee_pos.detach().cpu().numpy()

            if len(mode_history) == consecutive_modes_required:
                if np.all(np.array(mode_history) < WAYPOINT_THRESH):
                    mode = ActMode.Waypoint.value
                elif np.all(np.array(mode_history) > TERMINATE_THRESH):
                    mode = ActMode.Terminate.value
                else:
                    mode = ActMode.Dense.value
                mode_history = []
            mode_history.append(raw_mode.item())

            action = {
                'base_pose': np.zeros(3),
                'arm_pos': ee_pos * [1,-1,1],
                'arm_quat': ee_quat,
                'gripper_pos': np.array(gripper_open.item()),
                'base_image': np.zeros((640, 360, 3)),
                'wrist_image': np.zeros((640, 480, 3)),
            }

        if recorder is not None and step_counter % 4 == 0:
            recorder.add_numpy(obs, ["viewer_image"], color=(255, 140, 0))

        gripper_state = gripper_state if not action else action['gripper_pos'] 
        env.step(action, gripper_state)

        if viewer is not None:
            viewer.sync()
        env.rate_limiter.sleep()

        step_counter += 1

        if check_for_interrupt():
            mode = ActMode.Terminate.value
    return mode

def run_waypoint_mode(env, waypoint_policy, num_pass, viewer, recorder):
    num_waypoint_inferences = 0
    mode = ActMode.Waypoint.value
    reached = True
    while mode == ActMode.Waypoint.value:
        obs = env.observe()
        recorder.add_numpy(obs, ["viewer_image"])

        points, colors = pcl_from_obs(obs)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        proprio = obs["proprio"]

        if reached:
            with torch.no_grad():
                _, pos_cmd, rot_cmd, gripper_cmd, _ = waypoint_policy.inference(
                    torch.from_numpy(points).float(),
                    torch.from_numpy(colors).float(),
                    torch.from_numpy(proprio).float(),
                    num_pass=num_pass,
                )

                if waypoint_policy.cfg.use_euler:
                    euler_cmd = rot_cmd
                else:
                    euler_cmd = R.from_quat(rot_cmd).as_euler('xyz')

                num_waypoint_inferences += 1 # HACK, mode pred not yet working
                if num_waypoint_inferences == 2:
                    mode = ActMode.Dense.value

        reached, terminate = env.move_to(pos_cmd, euler_cmd, float(gripper_cmd), viewer, recorder)
        if terminate:
            mode = ActMode.Dense.value
    return mode

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--waypoint_model", type=str, required=True, help="waypoint model path")
    parser.add_argument("-d", "--dense_model", type=str, required=True, help="dense model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=20)
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='rollouts')
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/mj_env.yaml")
    args = parser.parse_args()

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    print(f">>>>>>>>>>{args.waypoint_model, args.dense_model}<<<<<<<<<<")
    waypoint_policy = load_waypoint(args.waypoint_model, device='cuda')
    waypoint_policy.train(False)
    waypoint_policy = waypoint_policy.cuda()

    dense_policy, dense_dataset, _ = load_model(args.dense_model, "cuda", load_only_one=True)
    dense_policy.eval()

    if dense_policy.cfg.use_ddpm:
        common_utils.cprint(f"Warning: override to use ddim with step 10")
        dense_policy.cfg.use_ddpm = 0
        dense_policy.cfg.ddim.num_inference_timesteps = 10

    scores = []
    for idx, seed in enumerate(range(args.seed, args.seed + args.num_episode)):
        eval_hybrid(
            waypoint_policy,
            dense_policy,
            dense_dataset,
            args.env_cfg,
            seed=seed,
            num_pass=args.num_pass,
            save_dir=args.save_dir,
            record=args.record,
        )
        print(common_utils.wrap_ruler("", max_len=80))

if __name__ == "__main__":
    # python scripts/eval_hybrid.py -w exps/waypoint/cabinet/ema.pt -d exps/dense/cabinet/latest.pt
    main()
