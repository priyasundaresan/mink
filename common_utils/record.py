import os
from collections import defaultdict
import imageio
import torch
import numpy as np


class Recorder:
    def __init__(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        self.combined_frames = []
        self.tensors = defaultdict(list)

    def add(self, camera_obses: dict[str, torch.Tensor]):
        combined = []
        for camera, obs in camera_obses.items():
            assert obs.dim() == 3 and obs.size(0) == 3
            assert obs.dtype == torch.uint8

            tensor = obs.cpu()
            self.tensors[camera].append(tensor)
            frame = tensor.permute([1, 2, 0]).numpy()
            combined.append(frame)

        combined = np.concatenate(combined, axis=1)
        self.combined_frames.append(combined)

    def add_numpy(self, obs: dict, cameras: list[str], color=None):
        combined = []
        for cam in cameras:
            image = obs[cam]
            assert len(image.shape) == 3 and image.shape[-1] == 3

            combined.append(image)

        combined = np.concatenate(combined, axis=1)
        if color is not None:
            combined[:20, :, :] = color

        self.combined_frames.append(combined)

    def save(self, name, fps=10):
        path = os.path.join(self.save_dir, f"{name}.mp4")
        # print(f"saving video to {path}")
        # control freq defaults to 0
        imageio.mimsave(path, self.combined_frames, fps=fps)
        self.combined_frames.clear()
        return path

    def save_images(self, name):
        path = os.path.join(self.save_dir, f"{name}.jpg")
        image = np.concatenate(self.combined_frames, axis=0)
        imageio.imwrite(path, image)
        print(f"saving image to {path}")
        self.combined_frames.clear()
        return path
