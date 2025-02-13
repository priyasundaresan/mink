import argparse
import copy
import numpy as np
import torch
import os
import pyrallis
import mujoco as mj

from envs.mj_env import MujocoEnvConfig, MujocoEnv
from envs.mj_utils.camera_utils import pcl_from_obs
from scipy.spatial.transform import Rotation as R
import common_utils

import os
import sys
from scripts.train_dense import load_model
from common_utils.eval_utils import (
    check_for_interrupt,
)
from contextlib import nullcontext

def eval_dense(
    dense_policy,
    dense_dataset,
    env_cfg,
    seed,
    save_dir,
    record,
    headless, # Whether to render the rollout onscreen (headless=False) or offscreen (headless=True)
):
    assert not dense_policy.training
    env = MujocoEnv(env_cfg)
    np.random.seed(seed)
    env.reset()

    recorder = None
    if record:
        assert save_dir is not None
        recorder = common_utils.Recorder(save_dir)

    cached_actions = []
    image_capture_interval = int(env.frequency / 10)  # Capture images every 10Hz
    step_counter = 0  # Counter to track simulation steps
    action = None
    gripper_state = 0 #TODO
    viewer = None

    if headless:
        context = nullcontext()
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

        while env.num_step < env.max_num_step:
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
                ee_pos = ee_pos.detach().cpu().numpy()

                if env.is_success() or check_for_interrupt():
                    break

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

    if recorder is not None:
        recorder.save(f"s{seed}", fps=150)

    return env.reward, env.num_step

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dense_model", type=str, required=True, help="dense model path")
    parser.add_argument("--seed", type=int, default=99999)
    parser.add_argument("--num_episode", type=int, default=20)
    parser.add_argument("--num_pass", type=int, default=3)
    parser.add_argument("--save_dir", type=str, default='rollouts')
    parser.add_argument("--record", type=int, default=1)
    parser.add_argument("--headless", action='store_true')
    parser.add_argument("-e", "--env_cfg", type=str, default="envs/cfgs/open.yaml")
    args = parser.parse_args()

    if args.save_dir is not None:
        log_path = os.path.join(args.save_dir, "eval.log")
        sys.stdout = common_utils.Logger(log_path, print_to_stdout=True)

    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    print(f">>>>>>>>>>{args.dense_model}<<<<<<<<<<")
    dense_policy, dense_dataset, _ = load_model(args.dense_model, "cuda", load_only_one=True)
    dense_policy.eval()

    if dense_policy.cfg.use_ddpm:
        common_utils.cprint(f"Warning: override to use ddim with step 10")
        dense_policy.cfg.use_ddpm = 0
        dense_policy.cfg.ddim.num_inference_timesteps = 10

    scores = []
    for idx, seed in enumerate(range(args.seed, args.seed + args.num_episode)):
        score, num_step = eval_dense(
            dense_policy,
            dense_dataset,
            env_cfg,
            seed=seed,
            save_dir=args.save_dir,
            record=args.record,
            headless=args.headless
        )
        scores.append(score)
        print(f"[{idx+1}/{args.num_episode}] avg. score: {np.mean(scores):.4f}, num_steps: {num_step}")
        print(common_utils.wrap_ruler("", max_len=80))

if __name__ == "__main__":
    ### Example Usage
    # python scripts/eval_dense.py -d exps/dense/cabinet_interpolate/latest.pt -e envs/cfgs/open.yaml
    # python scripts/eval_dense.py -d exps/dense/cube_interpolate/latest.pt -e envs/cfgs/cube.yaml 
    # python scripts/eval_dense.py -d exps/dense/cube_interpolate/latest.pt -e envs/cfgs/cube.yaml --headless
    ###
    main()
