import cv2
import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset

from mgen3d.data.data_utils import read_image


class NeRFDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        data_split: str,
        transform=None,
        white_background: bool = False,
        image_shape=None,
    ):
        self.root_dir = root_dir
        self.data_split = data_split
        self.transform = transform
        self.white_background = white_background
        self.image_shape = image_shape

        with open(
            os.path.join(root_dir, "transforms_{}.json".format(self.data_split)), "r"
        ) as fp:
            self.metadata = json.load(fp)

        imgs = []
        poses = []

        for frame in self.metadata["frames"][::1]:
            if frame["file_path"].endswith("png"):
                fname = os.path.join(self.root_dir, frame["file_path"])
            else:
                fname = os.path.join(self.root_dir, frame["file_path"] + ".png")
            imgs.append(read_image(fname, self.white_background))
            poses.append(np.array(frame["transform_matrix"]))
        self.imgs = torch.tensor(np.array(imgs))  # keep all 4 channels (RGBA)
        self.poses = torch.tensor(np.array(poses), dtype=torch.float)

    def calculate_focal_length(self, H, W, fov_y, fov_x):
        focal_y = 0.5 * H / np.tan(0.5 * fov_y)
        focal_x = 0.5 * W / np.tan(0.5 * fov_x)
        return focal_y, focal_x

    def resize_image(self, image, H, W, fov_y, fov_x):
        if self.image_shape is None:
            focal_y, focal_x = self.calculate_focal_length(H, W, fov_y, fov_x)
            return int(H), int(W), focal_y, focal_x, image
        else:
            new_H = self.image_shape[1]
            new_W = self.image_shape[0]
            focal_y, focal_x = self.calculate_focal_length(new_H, new_W, fov_y, fov_x)
            return (
                int(new_H),
                int(new_W),
                focal_y,
                focal_x,
                cv2.resize(image, self.image_shape, interpolation=cv2.INTER_AREA),
            )

    def __len__(self):
        return len(self.metadata["frames"])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        H, W = self.imgs[idx].shape[:2]
        fov_x = float(self.metadata["camera_angle_x"])
        fov_y = fov_x  # TODO: see if in the datasets we have this is true or not

        H, W, focal_y, focal_x, image = self.resize_image(
            self.imgs[idx].numpy(), H, W, fov_y, fov_x
        )
        K = np.array([[focal_x, 0, 0.5 * W], [0, focal_y, 0.5 * H], [0, 0, 1]])

        # store everything into a dict
        sample = {
            "imgs": image,
            "poses": self.poses[idx],
            "H": H,
            "W": W,
            "K": K,
        }

        return sample
