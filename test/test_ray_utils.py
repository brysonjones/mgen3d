
from mgen3d.nerf.ray_utils import *
import torch

def test_generate_rays():
    H = 32
    W = 32
    K = torch.eye(3, 3)
    K[0, 2] = 0.5
    K[1, 2] = 0.5
    c2w = torch.eye(4, 4)
    input_dict = {"H": H, "W": W, "K": K, "poses": c2w}
    rays_o, rays_d = generate_rays(input_dict)
    assert rays_o.shape == (H, W, 3)
    assert rays_d.shape == (H, W, 3)

    # check that all rays_d are unit vectors
    assert torch.allclose(torch.norm(rays_d, dim=-1), torch.ones((H, W)))
    