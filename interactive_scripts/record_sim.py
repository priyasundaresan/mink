import mujoco as mj
import time
import mujoco.viewer
import numpy as np
import os
from pathlib import Path
from loop_rate_limiters import RateLimiter
import queue
import mink
from teleop.policies import TeleopPolicy
from dm_control.viewer import user_input
from mj_utils.camera import Camera

class MujocoEnv:
    def __init__(self, xml_file, camera_names=["base", "wrist"]):
        self.model = mj.MjModel.from_xml_path(xml_file)
        self.data = mj.MjData(self.model)

        self.gripper_closed = False
 
        # Camera setup
        self.cameras = {camera_name: Camera(self.model, self.data, camera_name) for camera_name in camera_names}

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
        self.frequency = 200.0
        self.rate_limiter = RateLimiter(frequency=self.frequency, warn=False)

        self.teleop_policy = TeleopPolicy()

    def reset(self):
        mj.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        mj.mj_forward(self.model, self.data)

        # Initialize the mocap target at the end-effector site
        mink.move_mocap_to_frame(
            self.model, self.data, "pinch_site_target", "pinch_site", "site"
        )
        self.gripper_closed = False

    def step(self, action, gripper_closed):
        # Update mocap target from action
        if isinstance(action, dict):
            self.data.mocap_pos[0] = action['arm_pos'][[1, 0, 2]] * [-1, 1, 1]
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

    def keyboard_callback(self, key):
        """Handle keyboard input."""
        if key == user_input.KEY_P:
            self.gripper_closed = not self.gripper_closed

    def observe_camera(self, channel_first=False) -> dict[str, np.ndarray]:
        obs = {}
        for name in self.cameras:
            camera = self.cameras[name]

            rgb_image, depth_image = camera.rgbd_image

            if channel_first:
                rgb_image = rgb_image.transpose(2, 0, 1)
                depth_image = depth_image.transpose(2, 0, 1)

            obs["%s_image"%name] = rgb_image
            obs["%s_depth"%name] = depth_image
    
        return obs

    def run_one_episode(self):
        """Run the simulation with rendering and camera functionality."""
        key_queue = queue.Queue()
    
        # Reset
        self.teleop_policy.reset()
        self.reset()
    
        image_capture_interval = int(self.frequency / 10)  # Capture images every 10Hz
        step_counter = 0  # Counter to track simulation steps
    
        with mj.viewer.launch_passive(
            model=self.model,
            data=self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=lambda key: key_queue.put(key)
        ) as viewer:
    
            mj.mjv_defaultFreeCamera(self.model, viewer.cam)
    
            while viewer.is_running():
                # Collect and process key inputs
                while not key_queue.empty():
                    self.keyboard_callback(key_queue.get())
    
                # Update simulation observations
                obs = {
                    'base_pose': np.zeros(3),
                    'arm_pos': self.data.mocap_pos[0],
                    'arm_quat': self.data.mocap_quat[0][[1, 2, 3, 0]],
                    'gripper_pos': np.zeros(1),
                }
    
                # Capture images at 10Hz
                if step_counter % image_capture_interval == 0:
                    camera_obs = self.observe_camera()  # Only capture images at 10Hz
    
                # Compute action and step the simulation
                action = self.teleop_policy.step(obs)

                if action == "end_episode":
                    break

                self.step(action, self.gripper_closed)
    
                # Sync the viewer and sleep to maintain frequency
                viewer.sync()
                self.rate_limiter.sleep()
    
                # Increment the step counter
                step_counter += 1

if __name__ == "__main__":
    _HERE = Path(__file__).parent
    _XML = _HERE / "stanford_tidybot" / "scene.xml"

    # Create and run the environment
    env = MujocoEnv(_XML.as_posix())
    env.reset()

    while True:
        env.run_one_episode()
