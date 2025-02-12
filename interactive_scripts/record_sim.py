import pyrallis
import argparse
from envs.mj_env import MujocoEnvConfig, MujocoEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_cfg", type=str, default="envs/cfgs/open.yaml")
    args = parser.parse_args()
    
    env_cfg = pyrallis.load(MujocoEnvConfig, open(args.env_cfg, "r"))

    env = MujocoEnv(env_cfg)
    env.reset()

    while True:
        env.collect_episode()
