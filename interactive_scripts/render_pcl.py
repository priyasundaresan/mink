import pyrallis
import argparse
from envs.mj_env import MujocoEnvConfig, MujocoEnv
from envs.mj_utils.camera_utils import pcl_from_obs
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/mj_env_only_basecams.yaml")
    args = parser.parse_args()
    
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    env = MujocoEnv(env_cfg)
    env.reset()

    obs = env.observe()
    merged_points, merged_colors = pcl_from_obs(obs, crop=False)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(merged_points)
    point_cloud.colors = o3d.utility.Vector3dVector(merged_colors)
    o3d.visualization.draw_geometries([point_cloud], window_name="Point Cloud Viewer")
