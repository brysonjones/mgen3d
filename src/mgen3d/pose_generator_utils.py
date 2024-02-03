import numpy as np
import torch
import trimesh
from typing import *

trans_t = lambda t: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]])

rot_phi = lambda phi: np.array(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
)

rot_theta = lambda th: np.array(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
)

def create_spherical_pose(theta: float, phi: float, radius: float):
    """

    Args:
        theta (float): angle about y-axis
        phi (float): angle about x-axis
        radius (float): distance from origin

    Returns:
        np.ndarray: camera-to-world transformation matrix
    """
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return c2w


def create_view_data(H, W, K, theta: float, phi: float, radius: float):
    pose = create_spherical_pose(theta, phi, radius)
    data = {
        "H": torch.tensor(H),
        "W": torch.tensor(W),
        "K": torch.tensor(K),
        "poses": torch.tensor(pose).float(),
    }
    return data


class SphericalPoseSampler:
    def __init__(
        self,
        radius_range: Tuple[float] = (1, 1.5),
        theta_range: Tuple[float] = (0, 100),
        phi_range: Tuple[float] = (0, 360),
        sample_epsilon: float = 0.25,
    ):
        self.radius_range = radius_range
        self.theta_range = theta_range
        self.phi_range = phi_range
        self.sample_epsilon = sample_epsilon

    def sample_pose(self):
        sample_value = np.random.uniform(0.0, 1.0)
        if sample_value < self.sample_epsilon:
            radius = (
                self.radius_range[1] - self.radius_range[0]
            ) / 2 + self.radius_range[0]
            theta = (self.theta_range[1] - self.theta_range[0]) / 2 + self.theta_range[
                0
            ]
            phi = (self.phi_range[1] - self.phi_range[0]) / 2 + self.phi_range[0]
            is_reference_view = True
        else:
            radius = np.random.uniform(self.radius_range[0], self.radius_range[1])
            theta = np.random.uniform(self.theta_range[0], self.theta_range[1])
            phi = np.random.uniform(self.phi_range[0], self.phi_range[1])
            is_reference_view = False
        pose = create_spherical_pose(theta, phi, radius)

        return pose, is_reference_view

    def sample_poses(self, num_poses: int = 1):
        """Generate a list of poses on a sphere

        Args:
            num_poses (int): number of poses to generate

        Returns:
            poses: list of camera-to-world transformation matrices
            is_front_list: list of booleans indicating whether the pose is
            on the front hemisphere
        """
        poses = []
        is_front_list = []
        for _ in range(num_poses):
            pose, is_front = self.sample_pose()
            poses.append(pose)
            is_front_list.append(is_front)
        return poses, is_front_list


def compute_frustum_corners(fov, aspect_ratio, near, far):
    tangent = np.tan(np.radians(fov) / 2)
    height_near = 2 * near * tangent
    width_near = height_near * aspect_ratio

    height_far = 2 * far * tangent
    width_far = height_far * aspect_ratio

    # Near plane corners
    near_top_left = np.array([-width_near / 2, height_near / 2, -near, 1])
    near_top_right = np.array([width_near / 2, height_near / 2, -near, 1])
    near_bottom_left = np.array([-width_near / 2, -height_near / 2, -near, 1])
    near_bottom_right = np.array([width_near / 2, -height_near / 2, -near, 1])

    # Far plane corners
    far_top_left = np.array([-width_far / 2, height_far / 2, -far, 1])
    far_top_right = np.array([width_far / 2, height_far / 2, -far, 1])
    far_bottom_left = np.array([-width_far / 2, -height_far / 2, -far, 1])
    far_bottom_right = np.array([width_far / 2, -height_far / 2, -far, 1])

    return np.array(
        [
            near_top_left,
            near_top_right,
            near_bottom_left,
            near_bottom_right,
            far_top_left,
            far_top_right,
            far_bottom_left,
            far_bottom_right,
        ]
    )


def visualize_poses(poses, size=0.1):
    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=0.5)
    objects = [axes, sphere]

    fov = 60  # field of view in degrees
    aspect_ratio = 16 / 9  # width/height
    near = 0.02  # near plane distance
    far = 0.1  # far plane distance
    frustrum = compute_frustum_corners(fov, aspect_ratio, near, far)

    for pose in poses:
        # the camera frustrum is visualized with 12 line segments.
        frustrum_in_world_frame = [pose @ corner for corner in frustrum]
        segs = np.array(
            [  # near plane
                [frustrum_in_world_frame[0][:3], frustrum_in_world_frame[1][:3]],
                [frustrum_in_world_frame[1][:3], frustrum_in_world_frame[3][:3]],
                [frustrum_in_world_frame[3][:3], frustrum_in_world_frame[2][:3]],
                [frustrum_in_world_frame[2][:3], frustrum_in_world_frame[0][:3]],
                # far plane
                [frustrum_in_world_frame[4][:3], frustrum_in_world_frame[5][:3]],
                [frustrum_in_world_frame[5][:3], frustrum_in_world_frame[7][:3]],
                [frustrum_in_world_frame[7][:3], frustrum_in_world_frame[6][:3]],
                [frustrum_in_world_frame[6][:3], frustrum_in_world_frame[4][:3]],
                # sides
                [frustrum_in_world_frame[0][:3], frustrum_in_world_frame[4][:3]],
                [frustrum_in_world_frame[1][:3], frustrum_in_world_frame[5][:3]],
                [frustrum_in_world_frame[2][:3], frustrum_in_world_frame[6][:3]],
                [frustrum_in_world_frame[3][:3], frustrum_in_world_frame[7][:3]],
            ]
        )
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def main():
    poses = gen_spherical_poses()
    visualize_poses(poses)
