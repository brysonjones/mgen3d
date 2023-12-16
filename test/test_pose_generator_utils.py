
import numpy as np
from  mgen3d.pose_generator_utils import *

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
    
def test_gen_spherical_poses():
    # Test case 1
    radius_range = (1, 1.5)
    theta_range = (0, 360)
    phi_range = (0, 90)
    num_poses = 100

    poses = gen_spherical_poses(radius_range, theta_range, phi_range, num_poses)

    assert len(poses) == num_poses

    for pose in poses:
        assert pose.shape == (4, 4)
        assert np.linalg.norm(pose[0:3, 3]) >= radius_range[0] and np.linalg.norm(pose[0:3, 3]) <= radius_range[1]

    # Test case 2
    radius_range = (2, 3)
    theta_range = (45, 135)
    phi_range = (30, 60)
    num_poses = 50

    poses = gen_spherical_poses(radius_range, theta_range, phi_range, num_poses)

    assert len(poses) == num_poses

    for pose in poses:
        assert pose.shape == (4, 4)
        assert np.linalg.norm(pose[0:3, 3]) >= radius_range[0] and np.linalg.norm(pose[0:3, 3]) <= radius_range[1]

    # Test case 3
    radius_range = (0.5, 1)
    theta_range = (180, 270)
    phi_range = (45, 75)
    num_poses = 20

    poses = gen_spherical_poses(radius_range, theta_range, phi_range, num_poses)

    assert len(poses) == num_poses

    for pose in poses:
        assert pose.shape == (4, 4)
        assert np.linalg.norm(pose[0:3, 3]) >= radius_range[0] and np.linalg.norm(pose[0:3, 3]) <= radius_range[1]

def test_visualize_poses():
    poses = gen_spherical_poses(num_poses=30)
    visualize_poses(poses)