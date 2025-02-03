from dataclasses import dataclass
from scipy.spatial.transform import Rotation, Slerp
import numpy as np

def quaternion_to_euler_diff(quat1, quat2):
    """
    Calculate the Euler angle difference between two quaternions.
    
    Args:
        quat1: First quaternion as a list or numpy array [w, x, y, z].
        quat2: Second quaternion as a list or numpy array [w, x, y, z].
    
    Returns:
        euler_diff: Difference in Euler angles (roll, pitch, yaw) as a numpy array [roll_diff, pitch_diff, yaw_diff].
    """
    # Convert quaternions to scipy Rotation objects
    rot1 = Rotation.from_quat([quat1[1], quat1[2], quat1[3], quat1[0]])  # [x, y, z, w] format for scipy
    rot2 = Rotation.from_quat([quat2[1], quat2[2], quat2[3], quat2[0]])  # [x, y, z, w] format for scipy
    
    # Compute the relative rotation
    relative_rotation = rot1.inv() * rot2
    
    # Convert the relative rotation to Euler angles (in radians)
    euler_diff = relative_rotation.as_euler('xyz', degrees=False)  # Change 'xyz' to desired order
    
    return euler_diff

@dataclass
class Proprio:
    # supplied as arguments
    base_xy_th: np.ndarray
    eef_pos: np.ndarray
    eef_quat: np.ndarray
    joint_pos: np.ndarray
    gripper_width: float  

    # computed in __init__
    gripper_width_np: np.ndarray  # gripper_width converted to array
    eef_euler: np.ndarray  # rotation in euler
    eef_pos_euler_grip: np.ndarray

    def __init__(
        self,
        base_xy_th: list[float],
        eef_pos: list[float],
        eef_quat: list[float],
        joint_pos: list[float],
        gripper_width: float,
    ):
        self.base_xy_th= np.array(base_xy_th)  # , dtype=np.float32)
        self.eef_pos = np.array(eef_pos)  # , dtype=np.float32)
        self.eef_quat = np.array(eef_quat)  # , dtype=np.float32)
        self.joint_pos= np.array(joint_pos)  # , dtype=np.float32)
        self.gripper_width = gripper_width

        self.gripper_width_np = np.array([self.gripper_width])  # , dtype=np.float32)
        self.eef_euler = Rotation.from_quat(self.eef_quat).as_euler("xyz")  # .astype(np.float32)

        self.eef_pos_euler_grip = np.concatenate([self.eef_pos, self.eef_euler, self.gripper_width_np])


def position_action_to_delta_action(
    curr_pos: np.ndarray, curr_euler: np.ndarray, new_pos: np.ndarray, new_euler: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    delta_pos = new_pos - curr_pos
    curr_rot = Rotation.from_euler("xyz", curr_euler)
    target_rot = Rotation.from_euler("xyz", new_euler)
    delta_rot = target_rot * curr_rot.inv()
    delta_euler = delta_rot.as_euler("xyz")
    return delta_pos, delta_euler


# positional interpolation
def get_waypoint(start_pt, target_pt, max_delta):
    total_delta = target_pt - start_pt
    num_steps = (np.linalg.norm(total_delta) // max_delta) + 1
    remainder = np.linalg.norm(total_delta) % max_delta
    if remainder > 1e-3:
        num_steps += 1
    delta = total_delta / num_steps

    def gen_waypoint(i):
        return start_pt + delta * min(i, num_steps)

    return gen_waypoint, int(num_steps)


# rotation interpolation
def get_ori(initial_euler, final_euler, num_steps):
    diff = np.linalg.norm(final_euler - initial_euler)
    ori_chg = Rotation.from_euler("xyz", [initial_euler.copy(), final_euler.copy()], degrees=False)
    if diff < 0.02 or num_steps < 2:

        def gen_ori(i):
            return initial_euler

    else:
        slerp = Slerp([1, num_steps], ori_chg)

        def gen_ori(i):
            interp_euler = slerp(i).as_euler("xyz")
            return interp_euler

    return gen_ori


def _sgd_style_step(step_size, max_norm, delta):
    delta = step_size * delta
    delta_norm = np.linalg.norm(delta)
    delta = delta / delta_norm * min(delta_norm, max_norm)
    return delta

@dataclass
class LinearWaypointReachConfig:
    pos_threshold: float = 0.01
    pos_step_size: float = 0.1
    rot_threshold: float = 0.02
    rot_step_size: float = 0.1

class LinearWaypointReach:
    def __init__(
        self,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        cfg: LinearWaypointReachConfig,
    ):
        self.target_pos = target_pos
        self.target_rot = Rotation.from_euler("xyz", target_euler)
        self.cfg = cfg

    def step(self, curr_pos: np.ndarray, curr_euler: np.ndarray):
        # Compute position delta
        delta_pos = self.target_pos - curr_pos
        pos_reached = np.linalg.norm(delta_pos) < self.cfg.pos_threshold

        if pos_reached:
            abs_pos = self.target_pos
        else:
            # Take a linear step toward the target
            step_size = min(self.cfg.pos_step_size, np.linalg.norm(delta_pos))
            delta_pos = (delta_pos / np.linalg.norm(delta_pos)) * step_size
            abs_pos = curr_pos + delta_pos

        # Compute rotation delta using Slerp
        curr_rot = Rotation.from_euler("xyz", curr_euler)
        key_times = [0, 1]  # Define interpolation times
        key_rots = Rotation.concatenate([curr_rot, self.target_rot])
        slerp = Slerp(key_times, key_rots)

        # Interpolate to the next step (e.g., 0.5 for halfway)
        interpolated_rot = slerp(0.5)  # Adjust step size as needed
        abs_rot = interpolated_rot.as_euler("xyz")

        # Check if rotation has reached the target
        rot_reached = np.linalg.norm((self.target_rot * curr_rot.inv()).as_euler("xyz")) < self.cfg.rot_threshold

        # Check if both position and rotation have reached the target
        reached = pos_reached and rot_reached

        return abs_pos, abs_rot, reached



@dataclass
class WaypointReachConfig:
    pos_threshold: float = 0.01
    pos_step_size: float = 0.5
    pos_max_norm: float = 0.1
    rot_threshold: float = 0.02
    rot_step_size: float = 0.5
    rot_max_norm: float = 0.2

class WaypointReach:
    def __init__(
        self,
        max_delta_action: np.ndarray,
        target_pos: np.ndarray,
        target_euler: np.ndarray,
        cfg: WaypointReachConfig,
    ):
        assert max_delta_action.shape == (6,)
        self.max_delta_pos = max_delta_action[:3]
        self.max_delta_euler = max_delta_action[3:]
        self.target_pos = target_pos
        self.target_euler = target_euler
        self.cfg = cfg

    def step(self, curr_pos: np.ndarray, curr_euler: np.ndarray):
        delta_pos = self.target_pos - curr_pos
        pos_reached = np.linalg.norm(delta_pos) < self.cfg.pos_threshold
        if pos_reached:
            # print(f"Pos reached, err: {np.linalg.norm(delta_pos)}")
            delta_pos_action = np.zeros_like(curr_pos)
        else:
            # print("delta pos norm:", np.linalg.norm(delta_pos))
            delta_pos = _sgd_style_step(self.cfg.pos_step_size, self.cfg.pos_max_norm, delta_pos)
            delta_pos_action = (delta_pos / self.max_delta_pos).clip(min=-1, max=1)

        # next, process rot
        curr_rot = Rotation.from_euler("xyz", curr_euler)
        target_rot = Rotation.from_euler("xyz", self.target_euler)
        delta_euler = (target_rot * curr_rot.inv()).as_euler("xyz")

        rot_reached = np.linalg.norm(delta_euler) < self.cfg.rot_threshold
        if rot_reached:
            # print(f"Rot reached, err: {np.linalg.norm(delta_euler)}")
            delta_euler_action = np.zeros_like(delta_euler)
        else:
            # print("delta euler norm:", np.linalg.norm(delta_euler))
            delta_euler = _sgd_style_step(
                self.cfg.rot_step_size, self.cfg.rot_max_norm, delta_euler
            )
            delta_euler_action = (delta_euler / self.max_delta_euler).clip(min=-1, max=1)

        reached = rot_reached and pos_reached
        return delta_pos_action, delta_euler_action, reached

    def step_(self, curr_pos: np.ndarray, curr_euler: np.ndarray):
        delta_pos = self.target_pos - curr_pos
        delta_pos = delta_pos.clip(min=-self.max_delta_pos, max=self.max_delta_pos)

        pos_err = np.linalg.norm(delta_pos)
        pos_reached = pos_err < 0.01
        if pos_reached:
            delta_pos = np.zeros_like(delta_pos)
        else:
            delta_pos = delta_pos / self.max_delta_pos

        # next, process rot
        curr_rot = Rotation.from_euler("xyz", curr_euler)
        target_rot = Rotation.from_euler("xyz", self.target_euler)
        delta_euler = (target_rot * curr_rot.inv()).as_euler("xyz")

        if np.linalg.norm(delta_euler) < 0.02:
            delta_euler = np.zeros_like(delta_euler)
            rot_reached = True
        else:
            delta_euler = delta_euler.clip(min=-self.max_delta_euler, max=self.max_delta_euler)
            delta_euler /= self.max_delta_euler
            rot_reached = False

        reached = rot_reached and pos_reached
        return delta_pos, delta_euler, reached

@dataclass
class LinearWaypointReachConfig:
    pos_threshold: float = 0.01
    pos_step_size: float = 0.08
    rot_threshold: float = 0.05
    rot_step_size: float = 0.3


