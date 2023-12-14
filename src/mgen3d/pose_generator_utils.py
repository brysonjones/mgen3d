
import numpy as np
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
        radius (float): radius of sphere
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

