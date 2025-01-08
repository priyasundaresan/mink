from dataclasses import dataclass
from scipy.spatial.transform import Rotation
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
    rot1 = R.from_quat([quat1[1], quat1[2], quat1[3], quat1[0]])  # [x, y, z, w] format for scipy
    rot2 = R.from_quat([quat2[1], quat2[2], quat2[3], quat2[0]])  # [x, y, z, w] format for scipy
    
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
