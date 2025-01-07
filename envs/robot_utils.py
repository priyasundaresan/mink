from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import numpy as np

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
