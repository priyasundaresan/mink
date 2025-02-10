import pyrallis
import os
import argparse
from envs.mj_env import MujocoEnvConfig, MujocoEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/mj_env.yaml")
    args = parser.parse_args()
    
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    env = MujocoEnv(env_cfg)
    data_dir = 'dev1'
    for fn in os.listdir(data_dir):
        if 'npz' in fn:
            env.replay_episode(os.path.join(data_dir, fn))
