import numpy as np
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
        fov_ref: float = 20.0,
        fov_range: tuple = (15.0, 25.0),
        epsilon: float = 0.25,
    ):
        super.__init__(root_dir, data_split, transform, white_background)
        assert self.imgs.shape[0] == 1, "Only one reference view should be provided in the dataset"
        self.radius_range = radius_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.epsilon = epsilon
        self.sampler = SphericalPoseSampler(
            radius_range, theta_range, phi_range, epsilon
        )
        self.fov_ref = fov_ref
        self.fov_range = fov_range
        self.H = self.imgs.shape[1]
        self.W = self.imgs.shape[2]

    def __len__(self):
        return 1

    def get_intrinsics_(self, epsilon):
        focal_y, focal_x = None, None
        if epsilon < self.epsilon:
            focal_y, focal_x = self.calculate_focal_length(self.H, self.W, fov_y=self.fov_ref, fov_x=self.fov_ref)
        else:
            fov_y = np.random.uniform(self.fov_range[0], self.fov_range[1])
            fov_x = np.random.uniform(self.fov_range[0], self.fov_range[1])
            focal_y, focal_x = self.calculate_focal_length(self.H, self.W, fov_y, fov_x)
        assert focal_y is not None and focal_x is not None, "Focal length not calculated"
        H, W, focal_y, focal_x, _ = self.resize_image(
            self.imgs[0].numpy(), H, W, fov_y, fov_x
        )
        K = np.array([[focal_x, 0, 0.5 * W], [0, focal_y, 0.5 * H], [0, 0, 1]])
        return K

    def __getitem__(self, idx):
        pose, is_reference_view = self.sampler.sample_pose()
        sample = {
            "poses": pose,
            "is_reference_view": is_reference_view,
            "H": self.H,
            "W": self.W,
            "K": self.get_intrinsics_(self.epsilon)
        }
        if is_reference_view:
            sample_data = self.imgs[0]
            sample["imgs"] = sample_data["imgs"]
        else:
            sample_image = None
            sample["imgs"] = sample_image

        return sample
