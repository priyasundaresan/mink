import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from loop_rate_limiters import RateLimiter
import mink
from teleop.policies import TeleopPolicy
from dm_control.viewer import user_input

class MujocoEnv:
    def __init__(self, xml_file, frame_name="pinch_site", rgb_size=(640, 480)):
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)

        # Tasks and solver setup
        self.configuration = mink.Configuration(self.model)

        self.end_effector_task = mink.FrameTask(
            frame_name=frame_name,
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

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        self.configuration.update(self.data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        mujoco.mj_forward(self.model, self.data)

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
            vel = mink.solve_ik(self.configuration, self.tasks, 1 / 200.0, self.solver, 1e-3)
            self.configuration.integrate_inplace(vel, 1 / 200.0)
            err = self.end_effector_task.compute_error(self.configuration)
            if (
                np.linalg.norm(err[:3]) <= self.pos_threshold
                and np.linalg.norm(err[3:]) <= self.ori_threshold
            ):
                break

        # Apply controls
        self.data.ctrl[self.actuator_ids] = self.configuration.q[self.dof_ids]
        self.data.ctrl[self.fingers_actuator_id] = gripper_closed * 255
        mujoco.mj_step(self.model, self.data)



if __name__ == "__main__":
    _HERE = Path(__file__).parent
    _XML = _HERE / "stanford_tidybot" / "scene.xml"

    env = MujocoEnv(_XML.as_posix())
    policy = TeleopPolicy()
    policy.reset()

    env.reset()
    
    gripper_state = 0

    with mujoco.viewer.launch_passive(
        model=env.model,
        data=env.data,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:

        rate = RateLimiter(frequency=200.0, warn=False)
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        while viewer.is_running():
            obs = {
                'base_pose': np.zeros(3),
                'arm_pos': env.data.mocap_pos[0],
                'arm_quat': env.data.mocap_quat[0][[3, 1, 0, 2]] * [1, -1, 1, 1],
                'gripper_pos': np.zeros(1),
            }
            action = policy.step(obs)
            gripper_state = gripper_state if (not action or type(action) == str) else action['gripper_pos'] 
            env.step(action, gripper_state)

            viewer.sync()
            rate.sleep()
