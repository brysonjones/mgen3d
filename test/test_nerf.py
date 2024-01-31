import yaml
from mgen3d.nerf.nerf import NeRF
from mgen3d.nerf.utils import *
import nerfacc
import pytest
import torch


def test_init_nerf_model():
    # simply test that constructor works and does not throw any errors
    with open("./config/default.yaml", "r") as file:
        config = yaml.safe_load(file)
    estimator = construct_estimator(config)
    network = construct_network(config)
    optimizer = construct_optimizer(config, network)
    scheduler = construct_scheduler(config, optimizer)
    nerf = NeRF(config, estimator, network, optimizer, scheduler)