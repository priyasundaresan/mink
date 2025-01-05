import mujoco
import mujoco.viewer
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from dm_control.viewer import user_input
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R

import mink
from teleop.policies import TeleopPolicy

_HERE = Path(__file__).parent
_XML = _HERE / "stanford_tidybot" / "scene.xml"

@dataclass
class KeyCallback:
    gripper_closed: bool = False
    fix_base: bool = False
    pause: bool = False

    def __call__(self, key: int) -> None:
        if key == user_input.KEY_ENTER:
            self.fix_base = not self.fix_base
        elif key == user_input.KEY_SPACE:
            self.pause = not self.pause
        elif key == user_input.KEY_P:  # Toggle gripper state with "P" key
            self.gripper_closed = not self.gripper_closed


if __name__ == "__main__":
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    # Joints to control
    joint_names = [
        "joint_x", "joint_y", "joint_th",
        "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7"
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    fingers_actuator_id = model.actuator("fingers_actuator").id

    # Configuration and tasks
    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="pinch_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )
    posture_cost = np.zeros((model.nv,))
    posture_cost[3:] = 1e-3
    posture_task = mink.PostureTask(model, cost=posture_cost)

    immobile_base_cost = np.zeros((model.nv,))
    immobile_base_cost[:3] = 100
    damping_task = mink.DampingTask(model, immobile_base_cost)

    tasks = [end_effector_task, posture_task]
    limits = [
        mink.ConfigurationLimit(model),
    ]

    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Teleoperation setup
    policy = TeleopPolicy()
    policy.reset()  # Wait for user to press "Start episode"

    key_callback = KeyCallback()

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        show_left_ui=False,
        show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)
        mujoco.mj_forward(model, data)

        # Initialize the mocap target at the end-effector site
        mink.move_mocap_to_frame(model, data, "pinch_site_target", "pinch_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)

        while viewer.is_running():
            # Fetch iPhone teleop data
            obs = {
                'base_pose': np.zeros(3),
                'arm_pos': data.mocap_pos[0],  # Initial position of mocap
                'arm_quat': data.mocap_quat[0][[1, 2, 3, 0]],  # Reformat (w, x, y, z)
                'gripper_pos': np.zeros(1),
            }
            action = policy.step(obs)

            # Update mocap target from iPhone data
            if isinstance(action, dict):
                data.mocap_pos[0] = action['arm_pos'][[1, 0, 2]] * [-1, 1, 1]
                data.mocap_quat[0] = action['arm_quat'][[3, 1, 0, 2]] * [1, -1, 1, 1]

            # Update target from mocap
            T_wt = mink.SE3.from_mocap_name(model, data, "pinch_site_target")
            end_effector_task.set_target(T_wt)

            # IK solving
            for _ in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                if (
                    np.linalg.norm(err[:3]) <= pos_threshold
                    and np.linalg.norm(err[3:]) <= ori_threshold
                ):
                    break

            # Apply controls
            if not key_callback.pause:
                data.ctrl[actuator_ids] = configuration.q[dof_ids]
                data.ctrl[fingers_actuator_id] = key_callback.gripper_closed * 255
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)

            # Sync and maintain rate
            viewer.sync()
            rate.sleep()
