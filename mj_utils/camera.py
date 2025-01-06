import mujoco as mj
import os
import numpy as np
from datetime import datetime
import open3d as o3d
import cv2

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
        """
        Compute the intrinsic camera matrix (K) based on the camera's field of view (fov),
        width (_width), and height (_height) parameters, following the pinhole camera model.

        Returns:
        np.ndarray: The intrinsic camera matrix (K), a 3x3 array representing the camera's intrinsic parameters.
        """
        # Convert the field of view from degrees to radians
        theta = np.deg2rad(self.fov)

        # Focal length calculation (f in terms of sensor width and height)
        f_x = (self._width / 2) / np.tan(theta / 2)
        f_y = (self._height / 2) / np.tan(theta / 2)

        # Pixel resolution (assumed to be focal length per pixel unit)
        alpha_u = f_x  # focal length in terms of pixel width
        alpha_v = f_y  # focal length in terms of pixel height

        # Optical center offsets (assuming they are at the center of the sensor)
        u_0 = (self._width - 1) / 2.0
        v_0 = (self._height - 1) / 2.0

        # Intrinsic camera matrix K
        K = np.array([[alpha_u, 0, u_0], [0, alpha_v, v_0], [0, 0, 1]])

        return K

    @property
    def T_world_cam(self) -> np.ndarray:
        """
        Compute the homogeneous transformation matrix for the camera.

        The transformation matrix is computed from the camera's position and orientation.
        The position and orientation are retrieved from the camera data.

        Returns:
        np.ndarray: The 4x4 homogeneous transformation matrix representing the camera's pose.
        """
        pos = self._data.cam(self._cam_id).xpos
        rot = self._data.cam(self._cam_id).xmat.reshape(3, 3).T
        T = make_tf(pos=pos, ori=rot).A
        return T

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
    def rgbd_image(self) -> np.ndarray:
        """
        Capture and return a synchronized RGB-D image.
    
        Returns:
        - np.ndarray: A single RGB-D image where the last dimension is 4 (RGB + Depth).
        """
        # Update the scene for both RGB and Depth rendering
        self._renderer.update_scene(self._data, camera=self.name)
    
        # Capture RGB image
        rgb_image = self._renderer.render()
        self._image = rgb_image
    
        # Enable depth rendering, capture depth, and then disable it
        self._renderer.enable_depth_rendering()
        depth_image = self._renderer.render()
        self._depth_image = depth_image
        self._renderer.disable_depth_rendering()
    
        # Combine RGB and Depth into a single RGB-D image
        # Normalize depth to match RGB dimensions if needed
        depth_image = np.expand_dims(depth_image, axis=-1)  # Add a channel dimension
        return self._image, self._depth_image


    @property
    def point_cloud(self) -> np.ndarray:
        """Return the captured point cloud."""
        self._point_cloud = self._depth_to_point_cloud(self.depth_image)
        return self._point_cloud

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
