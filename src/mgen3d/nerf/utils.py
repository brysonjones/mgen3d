import torch
import torch.nn as nn
import torch.optim as optim
from nerfacc import OccGridEstimator
from yaml import YAMLObject
from mgen3d.nerf.networks import TinyCudaNetwork

def construct_estimator(config: YAMLObject, device: str = "cpu"):
    # aabb values correspond to the original Instant-NGP paper
    aabb_scale = config["estimator"]["aabb_scale"]
    aabb = torch.tensor(
        [-aabb_scale, -aabb_scale, -aabb_scale, aabb_scale, aabb_scale, aabb_scale],
        device=device,
    )
    estimator = OccGridEstimator(
        roi_aabb=aabb,
        resolution=config["estimator"]["grid_resolution"],
        levels=config["estimator"]["grid_nlvl"],
    ).to(device)
    return estimator


def construct_network(config: YAMLObject, device: str = "cpu"):
    network = TinyCudaNetwork(config["network"],
                              config["estimator"]["aabb_scale"]).to(device)
    return network


def construct_optimizer(config: YAMLObject, network: nn.Module):
    model_params = list(network.parameters())
    optimizer = optim.AdamW(
        model_params,
        lr=config["optimizer"]["learning_rate"],
        betas=(config["optimizer"]["beta"][0], config["optimizer"]["beta"][1]),
        eps=config["optimizer"]["epsilon"],
    )
    return optimizer


def construct_scheduler(config: YAMLObject, optimizer: optim.Optimizer):
    scheduler = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, total_iters=100
            ),
            torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    config["optimizer"]["max_steps"] // 2,
                    config["optimizer"]["max_steps"] * 3 // 4,
                    config["optimizer"]["max_steps"] * 9 // 10,
                ],
                gamma=config["optimizer"]["gamma"],
            ),
        ]
    )
    return scheduler
