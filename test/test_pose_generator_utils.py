import numpy as np
from mgen3d.pose_generator_utils import *


def test_create_spherical_pose():
    radius = 1
    theta = 30
    phi = 45
    c2w = create_spherical_pose(theta, phi, radius)
    assert c2w.shape[0] == 4
    assert c2w.shape[1] == 4

    c2w_inv = c2w.copy()
    c2w_inv[0:3, 0:3] = c2w[0:3, 0:3].T
    c2w_inv[0:3, 3] = -c2w[0:3, 0:3].T @ c2w[0:3, 3]
    identity = c2w @ c2w_inv
    assert np.allclose(identity, np.eye(4), rtol=1e-5, atol=1e-8)


def test_spherical_pose_generator():
    # Test case 1
    radius_range = (1, 1.5)
    theta_range = (0, 360)
    phi_range = (0, 90)
    sample_epsilon = 0.0
    pose_generator = SphericalPoseSampler(
        radius_range, theta_range, phi_range, sample_epsilon
    )

    num_poses = 100

    poses, is_front_list = pose_generator.sample_poses(num_poses)

    assert len(poses) == num_poses

    for pose in poses:
        assert pose.shape == (4, 4)
        assert (
            np.linalg.norm(pose[0:3, 3]) >= radius_range[0]
            and np.linalg.norm(pose[0:3, 3]) <= radius_range[1]
        )
