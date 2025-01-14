import mujoco as mj
import mujoco.viewer
import numpy as np
from pathlib import Path
from loop_rate_limiters import RateLimiter
import queue
import mink
from teleop.policies import TeleopPolicy
from scipy.spatial.transform import Rotation as R
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode
from envs.mj_utils.camera import Camera
from envs.robot_utils import Proprio, LinearWaypointReach, quaternion_to_euler_diff
from dataclasses import dataclass, field
from common_utils import Stopwatch
import pyrallis
import argparse
from dm_control.viewer import user_input
import os

@dataclass
class MujocoEnvConfig:
    cameras: list[str]
    xml_file: str
    data_folder: str
    image_size: int
    crop_floor: int = 1

class MujocoEnv:
    def __init__(self, cfg: MujocoEnvConfig):
        self.cfg = cfg
        self.model = mj.MjModel.from_xml_path(cfg.xml_file)
        self.data = mj.MjData(self.model)
        self.data_folder = cfg.data_folder

        self.recorder = DatasetRecorder(self.data_folder)
        self.stopwatch = Stopwatch()

        # Camera setup
        self.cameras = {camera_name: Camera(self.model, self.data, camera_name) for camera_name in cfg.cameras}

        # Tasks and solver setup
        self.configuration = mink.Configuration(self.model)

        self.end_effector_task = mink.FrameTask(
            frame_name="pinch_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_cost = np.zeros((self.model.nv,))
        self.posture_cost[3:] = 1e-3
        self.posture_task = mink.PostureTask(self.model, cost=self.posture_cost)

        self.tasks = [self.end_effector_task, self.posture_task]
        self.solver = "quadprog"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 20

        # Actuator IDs
        self.joint_names = [
            "joint_x", "joint_y", "joint_th",
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"
        ]

        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.joint_names])
        self.fingers_actuator_id = self.model.actuator("fingers_actuator").id
        self.tendon_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_TENDON, "split")
        self.eef_index = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base")
        self.pinch_site_index = mj.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch_site")

        self.frequency = 200.0
        self.rate_limiter = RateLimiter(frequency=self.frequency, warn=False)

        self.teleop_policy = None

    def reset(self):
        next_idx = self.recorder.get_next_idx()
        mj.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        mj.mj_forward(self.model, self.data)

        # Initialize the mocap target at the end-effector site
        mink.move_mocap_to_frame(
            self.model, self.data, "pinch_site_target", "pinch_site", "site"
        )

    def step(self, action, gripper_closed, is_delta=False):

        # Update mocap target from action
        if isinstance(action, dict):
            self.data.mocap_pos[0] = action['arm_pos'][[0, 1, 2]] * [1, -1, 1]
            self.data.mocap_quat[0] = action['arm_quat'][[3, 1, 0, 2]] * [1, -1, 1, 1]

        # Update target from mocap
        T_wt = mink.SE3.from_mocap_name(self.model, self.data, "pinch_site_target")

        self.end_effector_task.set_target(T_wt)

        # IK solving
        for _ in range(self.max_iters):
            vel = mink.solve_ik(self.configuration, self.tasks, 1 / self.frequency, self.solver, 1e-3)
            self.configuration.integrate_inplace(vel, 1 / self.frequency)
            err = self.end_effector_task.compute_error(self.configuration)
            if (
                np.linalg.norm(err[:3]) <= self.pos_threshold
                and np.linalg.norm(err[3:]) <= self.ori_threshold
            ):
                break

        # Apply controls
        self.data.ctrl[self.actuator_ids] = self.configuration.q[self.dof_ids]
        self.data.ctrl[self.fingers_actuator_id] = gripper_closed * 255
        mj.mj_step(self.model, self.data)

    def observe_camera(self, channel_first=False) -> dict[str, np.ndarray]:
        obs = {}

        for name in self.cameras:
            camera = self.cameras[name]

            rgb_image, depth_image = camera.rgbd_obs

            if channel_first:
                rgb_image = rgb_image.transpose(2, 0, 1)
                depth_image = depth_image.transpose(2, 0, 1)

            obs["%s_image"%name] = rgb_image
            obs["%s_depth"%name] = depth_image
            obs["%s_K"%name] = camera._K
            obs["%s_T"%name] = camera.T_world_cam
    
        return obs

    def observe_proprio(self) -> Proprio:
        configuration = self.configuration.q[self.dof_ids]
        gripper_width = 1.0 - self.data.ten_length[self.tendon_index]

        #eef_pos = self.data.xpos[self.eef_index]
        #eef_quat = self.data.xquat[self.eef_index]

        #TODO: bit of a hack
        eef_pos = self.data.site_xpos[self.pinch_site_index]
        eef_quat= R.from_matrix(self.data.site_xmat[self.pinch_site_index].reshape(3, 3)).as_quat()
        eef_quat = eef_quat[[1, 0, 2, 3]] * [1, -1, 1, 1]

        proprio = Proprio(
            base_xy_th=configuration[:3],
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            joint_pos=configuration[3:],
            gripper_width=gripper_width
        )
        return proprio

    def observe(self) -> dict[str, np.ndarray]:
        with self.stopwatch.time("observe_camera"):
            obs = self.observe_camera()

        with self.stopwatch.time("observe_proprio"):
            proprio = self.observe_proprio()

        obs["base_xy_th"] = proprio.base_xy_th
        obs["eef_pos"] = proprio.eef_pos
        obs["eef_euler"] = proprio.eef_euler
        obs["eef_quat"] = proprio.eef_quat
        obs["joint_pos"] = proprio.joint_pos
        obs["gripper_width"] = proprio.gripper_width_np
        obs["proprio"] = proprio.eef_pos_euler_grip

        return obs

    def keyboard_callback(self, key):
        """Handle keyboard input."""
        if key == user_input.KEY_SPACE:
            return True
        return False

    def collect_episode(self):
        """Run the simulation with rendering and camera functionality."""
    
        # Reset
        if self.teleop_policy is None:
            self.teleop_policy = TeleopPolicy()
        self.teleop_policy.reset()

        self.reset()
    
        image_capture_interval = int(self.frequency / 10)  # Capture images every 10Hz
        step_counter = 0  # Counter to track simulation steps
        action = None
        gripper_state = 0
        
        with mj.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
    
            mj.mjv_defaultFreeCamera(self.model, viewer.cam)
    
            while viewer.is_running():
                # Collect and process key inputs
    
                # Update simulation observations
                obs = {
                    'base_pose': np.zeros(3),
                    'arm_pos': self.data.mocap_pos[0],
                    'arm_quat': self.data.mocap_quat[0][[3, 1, 0, 2]] * [1, -1, 1, 1],
                    'gripper_pos': np.zeros(1),
                }
    
                # Capture images at 10Hz
                if action and step_counter % image_capture_interval == 0:

                    with self.stopwatch.time("observe"):
                        record_obs = self.observe()  # Only capture observations at 10Hz

                    action['arm_euler'] = R.from_quat(action['arm_quat']).as_euler('xyz')

                    # Absolute action to record
                    record_action = np.concatenate([action['arm_pos'], \
					              action['arm_euler'], \
						      action['gripper_pos']])

                    delta_pos = action['arm_pos'] - record_obs['eef_pos']
                    delta_euler = quaternion_to_euler_diff(action['arm_quat'], \
		        				   record_obs['eef_quat'])

                    # Delta action to record
                    record_delta_action = np.concatenate([delta_pos, \
					              delta_euler, \
						      action['gripper_pos']])

                    self.recorder.record(ActMode.Dense, record_obs, action=record_action, delta_action=record_delta_action)
    
                # Compute action and step the simulation
                action = self.teleop_policy.step(obs)

                if action == "end_episode":
                    break
               
                gripper_state = gripper_state if not action else action['gripper_pos'] 

                with self.stopwatch.time("step"):
                    self.step(action, gripper_state)
    
                # Sync the viewer and sleep to maintain frequency
                viewer.sync()
                self.rate_limiter.sleep()
    
                # Increment the step counter
                step_counter += 1

        self.recorder.end_episode(save=True)
        self.stopwatch.summary()
        print('Done saving')

    def replay_episode(self, episode_fn):
        demo = np.load(episode_fn, allow_pickle=True)['arr_0']
        # Reset
        self.reset()


        episode = []
        for t, step in enumerate(list(demo)):
            episode.append(step)

        image_capture_interval = int(self.frequency / 10)  # Capture images every 10Hz
        step_counter = 0  # Counter to track simulation steps
        action = None
        gripper_state = 0
        
        with mj.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
    
            mj.mjv_defaultFreeCamera(self.model, viewer.cam)
    
            while viewer.is_running() and len(episode):
                # Capture images at 10Hz
                if step_counter % image_capture_interval == 0:

                    step = episode.pop(0)

                    recorded_obs = step['obs']
                    recorded_action = step['action']
                    recorded_delta_action = step['delta_action']

                    ## Absolute Replay
                    action = {
                        'base_pose': np.zeros(3),
                        'arm_pos': recorded_action[:3],
                        'arm_quat': R.from_euler('xyz', recorded_action[3:6]).as_quat(),
                        'gripper_pos': np.array(recorded_action[-1]),
                        'base_image': np.zeros((640, 360, 3)),
                        'wrist_image': np.zeros((640, 480, 3)),
                    }

                    #### Delta Replay
                    #obs_eef_pos = recorded_obs['eef_pos']
                    #obs_eef_euler = recorded_obs['eef_euler']

                    #action = {
                    #    'base_pose': np.zeros(3),
                    #    'arm_pos': recorded_delta_action[:3] + obs_eef_pos,
                    #    'arm_quat': R.from_euler('xyz', recorded_delta_action[3:6] + obs_eef_euler).as_quat(),
                    #    'gripper_pos': np.array(recorded_delta_action[-1]),
                    #    'base_image': np.zeros((640, 360, 3)),
                    #    'wrist_image': np.zeros((640, 480, 3)),
                    #}
                    
                gripper_state = gripper_state if not action else action['gripper_pos'] 
                self.step(action, gripper_state)
    
                # Sync the viewer and sleep to maintain frequency
                viewer.sync()
                self.rate_limiter.sleep()
    
                # Increment the step counter
                step_counter += 1

    def relabel_episode(self, episode_fn):
        demo = np.load(episode_fn, allow_pickle=True)['arr_0']
        key_queue = queue.Queue()

        # Reset
        self.reset()

        traj = []
        for t, step in enumerate(list(demo)):
            action = step["action"]
            traj.append(action)

        episode_counter = 0
        waypoint_idxs = []

        image_capture_interval = int(self.frequency / 10)  # Capture images every 10Hz
        step_counter = 0  # Counter to track simulation steps
        action = None
        gripper_state = 0
        
        with mj.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=lambda key: key_queue.put(key)
        ) as viewer:
    
            mj.mjv_defaultFreeCamera(self.model, viewer.cam)
    
            while viewer.is_running() and len(traj):

                # Capture images at 10Hz
                if step_counter % image_capture_interval == 0:
                    recorded_action = traj.pop(0)

                    # Collect and process key inputs
                    while not key_queue.empty():
                        if self.keyboard_callback(key_queue.get()):
                            print('Labeled waypoint')
                            waypoint_idxs.append(episode_counter)

                    action = {
                        'base_pose': np.zeros(3),
                        'arm_pos': recorded_action[:3],
                        'arm_quat': R.from_euler('xyz', recorded_action[3:6]).as_quat(),
                        'gripper_pos': np.array(recorded_action[-1]),
                        'base_image': np.zeros((640, 360, 3)),
                        'wrist_image': np.zeros((640, 480, 3)),
                    }

                    episode_counter += 1
                    
                gripper_state = gripper_state if not action else action['gripper_pos'] 
                self.step(action, gripper_state)
    
                # Sync the viewer and sleep to maintain frequency
                viewer.sync()
                self.rate_limiter.sleep()
    
                # Increment the step counter
                step_counter += 1

        waypoint_idx = -1
        curr_waypoint_step = 0

        for t, step in enumerate(list(demo)):

            if t == curr_waypoint_step and len(waypoint_idxs):

                waypoint_action = list(demo)[waypoint_idxs[0]]['action'] 

                step['action'] = waypoint_action
                step['mode'] = ActMode.Waypoint
                step['waypoint_idx'] = waypoint_idx

                curr_waypoint_step = waypoint_idxs.pop(0)
                waypoint_idx += 1

            step['waypoint_idx'] = waypoint_idx

        for t, step in enumerate(list(demo)):
            print(step['mode'], step['waypoint_idx'], step['action'])

        if not os.path.exists('devrelabel'):
            os.mkdir('devrelabel')

        np.savez(episode_fn.replace('dev1', 'devrelabel'), demo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/mj_env.yaml")
    args = parser.parse_args()
    
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    # Create and run the environment
    env = MujocoEnv(env_cfg)

    #env.replay_episode("dev1/demo00000.npz")
    env.relabel_episode("dev1/demo00000.npz")
