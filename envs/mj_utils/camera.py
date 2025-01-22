import mujoco as mj
import os
import numpy as np
from datetime import datetime
import open3d as o3d
import cv2
from envs.mj_utils.camera_utils import *

class Camera:
    def __init__(self, model, data, cam_name: str = "", save_dir="data/img/"):
        """Initialize Camera instance.

        Args:
        - model: Mujoco model.
        - data: Mujoco data.
        - cam_name: Name of the camera.
        - save_dir: Directory to save captured images.
        """
        self._cam_name = cam_name
        self._model = model
        self._data = data
        self._save_dir = save_dir + self._cam_name + "/"

        # Retrieve camera ID
        self._cam_id = self._data.cam(self._cam_name).id
        
        # Retrieve camera resolution directly from the model
        self._width, self._height = self._model.cam_resolution[self._cam_id]

        self._renderer = mj.Renderer(self._model, self._height, self._width)
        self._camera = mj.MjvCamera()
        self._scene = mj.MjvScene(self._model, maxgeom=10_000)

        self._K = self.K

        self._image = np.zeros((self._height, self._width, 3), dtype=np.uint8)
        self._depth_image = np.zeros((self._height, self._width, 1), dtype=np.float32)
        self._point_cloud = np.zeros((self._height, self._width, 1), dtype=np.float32)

        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)

    @property
    def height(self) -> int:
        """
        Get the height of the camera.

        Returns:
                int: The height of the camera.
        """
        return self._height

    @property
    def width(self) -> int:
        """
        Get the width of the camera.

        Returns:
                int: The width of the camera.
        """
        return self._width

    @property
    def save_dir(self) -> str:
        """
        Get the directory where images captured by the camera are saved.

        Returns:
                str: The directory where images captured by the camera are saved.
        """
        return self._save_dir

    @property
    def name(self) -> str:
        """
        Get the name of the camera.

        Returns:
                str: The name of the camera.s
        """
        return self._cam_name

    @property
    def K(self) -> np.ndarray:
        fy = 0.5 * self._height / np.tan(self.fov * np.pi / 360)  # Vertical focal length
        fx = fy * (self._width / self._height)  # Horizontal focal length based on aspect ratio
        K = np.array([
            [fx, 0, self._width / 2],  # Intrinsics for x-axis
            [0, fy, self._height / 2],  # Intrinsics for y-axis
            [0, 0, 1]  # Homogeneous coordinates
        ])
        return K
    

    @property
    def T_world_cam(self) -> np.ndarray:
        """
        Compute the homogeneous transformation matrix for the camera, taking into
        account the base's transformation in the world frame.
    
        Returns:
        np.ndarray: The 4x4 homogeneous transformation matrix representing the camera's pose in the world frame.
        """
        # Transformation from the camera to the base frame
        cam_pos = self._data.cam(self._cam_id).xpos
        cam_rot = self._data.cam(self._cam_id).xmat.reshape(3, 3)
        T_world_cam = make_tf(pos=cam_pos, ori=cam_rot).A
        camera_axis_correction = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        T_world_cam = T_world_cam @ camera_axis_correction
    
        return T_world_cam

    @property
    def P(self) -> np.ndarray:
        """
        Compute the projection matrix for the camera.

        The projection matrix is computed as the product of the camera's intrinsic matrix (K)
        and the homogeneous transformation matrix (T_world_cam).

        Returns:
        np.ndarray: The 3x4 projection matrix.
        """
        return self.K @ self.T_world_cam

    @property
    def rgbd_obs(self) -> np.ndarray:
        self._renderer.update_scene(self._data, camera=self.name)
    
        # Capture RGB image
        rgb_image = self._renderer.render()
        self._image = rgb_image
    
        # Enable depth rendering, capture depth, and then disable it
        self._renderer.enable_depth_rendering()
        depth_image = self._renderer.render()
        self._depth_image = depth_image
        self._renderer.disable_depth_rendering()
    
        depth_image = np.expand_dims(depth_image, axis=-1)

        return self._image, self._depth_image

    @property
    def point_cloud_cam(self) -> np.ndarray:
        """Return the captured point cloud."""
        self._point_cloud_cam = self._depth_to_point_cloud(self.depth_image)
        return self._point_cloud_cam

    @property
    def fov(self) -> float:
        """Get the field of view (FOV) of the camera.

        Returns:
        - float: The field of view angle in degrees.
        """
        return self._model.cam(self._cam_id).fovy[0]

    @property
    def id(self) -> int:
        """Get the identifier of the camera.

        Returns:
        - int: The identifier of the camera.
        """
        return self._cam_id

    def _depth_to_point_cloud(self, depth_image: np.ndarray) -> np.ndarray:
        """
        Method to convert depth image to a point cloud in camera coordinates.

        Args:
        - depth_image: The depth image we want to convert to a point cloud.

        Returns:
        - np.ndarray: 3D points in camera coordinates.
        """
        # Get image dimensions
        dimg_shape = depth_image.shape
        height = dimg_shape[0]
        width = dimg_shape[1]

        # Create pixel grid
        y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

        # Flatten arrays for vectorized computation
        x_flat = x.flatten()
        y_flat = y.flatten()
        depth_flat = depth_image.flatten()

        # Negate depth values because z-axis goes into the camera
        depth_flat = -depth_flat

        # Stack flattened arrays to form homogeneous coordinates
        homogeneous_coords = np.vstack((x_flat, y_flat, np.ones_like(x_flat)))

        # Compute inverse of the intrinsic matrix K
        K_inv = np.linalg.inv(self.K)

        # Calculate 3D points in camera coordinates
        points_camera = np.dot(K_inv, homogeneous_coords) * depth_flat

        # Homogeneous coordinates to 3D points
        points_camera = np.vstack((points_camera, np.ones_like(x_flat)))

        points_camera = points_camera.T

        # dehomogenize
        points_camera = points_camera[:, :3] / points_camera[:, 3][:, np.newaxis]

        return points_camera

    def shoot(self, autosave: bool = True) -> None:
        """
        Captures a new rgb image, depth image and point cloud from the camera.
        Args:
        - autosave: If the camera rgb image, depth image and point cloud should be saved.

        Returns:
        - None.
        """
        self._image = self.image
        self._depth_image = self.depth_image
        self._point_cloud_cam = self.point_cloud_cam
        self._point_cloud = self.point_cloud
        self._seg_image = self.seg_image
        if autosave:
            self.save()

    def save(self, img_name: str = "") -> None:
        """Saves the captured image and depth information.

        Args:
        - img_name: Name for the saved image file.
        """
        print(f"saving rgb image, depth image and point cloud to {self.save_dir}")

        if img_name == "":
            timestamp = datetime.now()
            cv2.imwrite(
                self._save_dir + f"{timestamp}_rgb.png",
                cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(self._save_dir + f"{timestamp}_seg.png", self.seg_image)
            np.save(self._save_dir + f"{timestamp}_depth.npy", self.depth_image)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
            o3d.io.write_point_cloud(self._save_dir + f"{timestamp}.pcd", pcd)

        else:
            cv2.imwrite(
                self._save_dir + f"{img_name}_rgb.png",
                cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(self._save_dir + f"{img_name}_seg.png", self.seg_image)
            np.save(self._save_dir + f"{img_name}_depth.npy", self.depth_image)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.point_cloud)
            o3d.io.write_point_cloud(self._save_dir + f"{image_name}.pcd", pcd)
