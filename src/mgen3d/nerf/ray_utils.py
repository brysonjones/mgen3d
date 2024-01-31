import torch
import torch.nn.functional as F
import numpy as np

def generate_rays(input_dict:dict, device:str="cpu") -> tuple:
    '''
    inputs:
        H: image height
        W: image width
        K: camera intrinsic matrix
        c2w: camera-to-world transformation matrix
    outputs:
        rays_o: (H, W, 3) origin of rays
        rays_d: (H, W, 3) direction of rays
    '''
    H = int(input_dict["H"])
    W = int(input_dict["W"])
    K = input_dict["K"].squeeze()
    c2w = input_dict["poses"].squeeze()
    
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), 
                          torch.linspace(0, H-1, H),
                          indexing='ij')
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2]+0.5)/K[0][0], -(j-K[1][2]+0.5)/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # equiv to dot product
    rays_d = F.normalize(rays_d, dim=-1)  # (height, width, 3)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o.to(device), rays_d.to(device)


def get_chunks(input:torch.Tensor, chunk_size:int) -> list:
    """
    Inputs:
        input: torch.Tensor((N, feature_num)D)
        chunk_size: the size of each chunk
    Return:
        chunks: list((chunk_size)D), each as torch.Tensor((chunk_size, feature_num)D)
    """
    return [input[i:i+chunk_size] for i in range(0, input.shape[0], chunk_size)]


