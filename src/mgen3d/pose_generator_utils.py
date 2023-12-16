
import numpy as np
import trimesh
from typing import *

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]])

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]])

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]])

def create_spherical_pose(theta:float, phi:float, radius:float):
    """

    Args:
        theta (float): angle about y-axis
        phi (float): angle about x-axis
        radius (float): distance from origin

    Returns:
        np.ndarray: camera-to-world transformation matrix
    """    
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w  

def gen_spherical_poses(radius_range:Tuple[float]=(1, 1.5), theta_range:Tuple[float]=(0, 360), 
                        phi_range:Tuple[float]=(0, 90), num_poses:int=100):
    """Generate a list of poses on a sphere

    Args:
        radius_range (float): range of radius values
        theta_range (float): range of theta values
        phi_range (float): range of phi values
        num_poses (int): number of poses to generate

    Returns:
        list: list of camera-to-world transformation matrices
    """    
    poses = []
    radius_list = np.random.uniform(radius_range[0], radius_range[1], num_poses)
    theta_list = np.random.uniform(theta_range[0], theta_range[1], num_poses)
    phi_list = np.random.uniform(phi_range[0], phi_range[1], num_poses)
    for radius, theta, phi in zip(radius_list, theta_list, phi_list):
        poses.append(create_spherical_pose(theta, phi, radius))
    return poses

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

    return np.array([
        near_top_left, near_top_right, near_bottom_left, near_bottom_right,
        far_top_left, far_top_right, far_bottom_left, far_bottom_right
    ])


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
        segs = np.array([ # near plane
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
                         [frustrum_in_world_frame[3][:3], frustrum_in_world_frame[7][:3]]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def main():
    poses = gen_spherical_poses()
    visualize_poses(poses)
                    