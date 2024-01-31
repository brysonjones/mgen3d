import os
import torch
from mgen3d.data.data_utils import read_image
from mgen3d.data.nerf_dataset import NeRFDataset
from mgen3d.pose_generator_utils import SphericalPoseSampler


class SamplingDataset(NeRFDataset):
    def __init__(
        self,
        root_dir: str,
        data_split: str,
        transform=None,
        white_background: bool = False,
        radius_range: tuple = (1.0, 1.5),
        theta_range: tuple = (0, 100),
        phi_range: tuple = (0, 360),
        epsilon: float = 0.25,
    ):
        super.__init__(root_dir, data_split, transform, white_background)
        self.radius_range = radius_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.epsilon = epsilon
        self.sampler = SphericalPoseSampler(
            radius_range, theta_range, phi_range, epsilon
        )
        self.H = None
        self.W = None
        self.K = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        pose, is_front = self.sampler.sample_pose()
        sample = {
            "poses": pose,
            "is_front": is_front,
            "H": self.H,
            "W": self.W,
            "K": self.K,
        }
        if is_front:
            sample_data = NeRFDataset.__getitem__(idx)
            sample["imgs"] = sample_data["imgs"]
        else:
            sample_image = None
            sample["imgs"] = sample_image

        return sample
