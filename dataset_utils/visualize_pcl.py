import open3d as o3d
import numpy as np
from envs.mj_utils.camera_utils import pcl_from_obs
from scipy.spatial.transform import Rotation as R

def label_salient_points(episode_fn):
    demo = np.load(episode_fn, allow_pickle=True)['arr_0']
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Loop through each timestep
    for t, step in enumerate(demo):
        obs = step["obs"]
        merged_points, merged_colors = pcl_from_obs(obs, crop=True)

        # Check for valid point cloud data
        if merged_points is None or merged_colors is None or len(merged_points) == 0:
            print(f"Step {t}: Invalid or empty point cloud data.")
            continue

        merged_points = (R.from_euler('x', -55, degrees=True).as_matrix() @ merged_points.T).T

        # Create a new point cloud for each step
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.vstack(merged_points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.vstack(merged_colors))
        print(f"Step {t}: Displaying {len(merged_points)} points.")

        # Add the point cloud to the visualizer
        vis.add_geometry(point_cloud)

        # Update the geometry in the visualizer
        vis.poll_events()
        vis.update_renderer()

        # Remove the current point cloud to prepare for the next frame
        vis.remove_geometry(point_cloud)

    vis.destroy_window()

if __name__ == "__main__":
    label_salient_points('dev1/demo00000.npz')

