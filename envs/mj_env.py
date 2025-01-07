import mujoco as mj
import mujoco.viewer
import numpy as np
from pathlib import Path
from loop_rate_limiters import RateLimiter
import mink
from teleop.policies import TeleopPolicy
from interactive_scripts.dataset_recorder import DatasetRecorder, ActMode
from envs.mj_utils.camera import Camera
from envs.robot_utils import Proprio
from dataclasses import dataclass, field
from common_utils import Stopwatch
import pyrallis
import argparse

@dataclass
class MujocoEnvConfig:
    cameras: list[str]
    xml_file: str
    data_folder: str
    image_size: int
    record_sim_state: int = 0
    crop_floor: int = 1

class MujocoEnv:
    def __init__(self, cfg: MujocoEnvConfig):
        self.cfg = cfg
        self.model = mj.MjModel.from_xml_path(cfg.xml_file)
        self.data = mj.MjData(self.model)
        self.data_folder = cfg.data_folder

        self.recorder = DatasetRecorder(self.data_folder)

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

    def step(self, action, gripper_closed):

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
            obs["%s_T"%name] = camera._T
    
        return obs

    def observe_proprio(self) -> Proprio:
        configuration = self.configuration.q[self.dof_ids]
        gripper_width = 1.0 - self.data.ten_length[self.tendon_index]
        eef_pos = self.data.xpos[self.eef_index]
        eef_quat = self.data.xquat[self.eef_index]
        proprio = Proprio(
            base_xy_th=configuration[:3],
            eef_pos=eef_pos,
            eef_quat=eef_quat,
            joint_pos=configuration[3:],
            gripper_width=gripper_width
        )
        return proprio

    def observe(self) -> dict[str, np.ndarray]:
        obs = self.observe_camera()

        proprio = self.observe_proprio()
        obs["base_xy_th"] = proprio.base_xy_th
        obs["eef_euler"] = proprio.eef_euler
        obs["eef_quat"] = proprio.eef_quat
        obs["joint_pos"] = proprio.joint_pos
        obs["gripper_width"] = proprio.gripper_width_np
        obs["proprio"] = proprio.eef_pos_euler_grip

        #if self.cfg.record_sim_state:
        #    # obs["model"] = self.env.sim.model.get_xml()
        #    obs["sim_state"] = self.env.sim.get_state().flatten()

        return obs

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
                    record_obs = self.observe()  # Only capture observations at 10Hz
                    record_action = np.concatenate([action['arm_pos'], \
					              action['arm_quat'], \
						      action['gripper_pos']])
                    self.recorder.record(ActMode.Dense, record_obs, action=record_action)
    
                # Compute action and step the simulation
                action = self.teleop_policy.step(obs)

                if action == "end_episode":
                    break
               
                gripper_state = gripper_state if not action else action['gripper_pos'] 
                self.step(action, gripper_state)
    
                # Sync the viewer and sleep to maintain frequency
                viewer.sync()
                self.rate_limiter.sleep()
    
                # Increment the step counter
                step_counter += 1

        self.recorder.end_episode(save=True)
        print('Done saving')

    def replay_episode(self, demo):
        # Reset
        self.reset()

        traj = []
        for t, step in enumerate(list(demo)):
            action = step["action"]
            traj.append(action)

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
    
            while viewer.is_running() and len(traj):
                # Capture images at 10Hz
                if step_counter % image_capture_interval == 0:
                    recorded_action = traj.pop(0)
                    action = {
                        'base_pose': np.zeros(3),
                        'arm_pos': recorded_action[:3],
                        'arm_quat': recorded_action[3:7],
                        'gripper_pos': np.array(recorded_action[-1]),
                        'base_image': np.zeros((640, 360, 3)),
                        'wrist_image': np.zeros((640, 480, 3)),
                    }
                    
                gripper_state = gripper_state if not action else action['gripper_pos'] 
                self.step(action, gripper_state)
    
                # Sync the viewer and sleep to maintain frequency
                viewer.sync()
                self.rate_limiter.sleep()
    
                # Increment the step counter
                step_counter += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/mj_env.yaml")
    args = parser.parse_args()
    
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    # Create and run the environment
    env = MujocoEnv(env_cfg)

    #env.reset()
    #stopwatch = Stopwatch()
    #for i in range(30):
    #    with stopwatch.time("observe_camera"):
    #        env.observe_camera()
    #stopwatch.summary()

    demo = np.load("dev1/demo00000.npz", allow_pickle=True)['arr_0']
    env.replay_episode(demo)

