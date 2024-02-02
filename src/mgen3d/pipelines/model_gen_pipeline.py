
import sys
import absl.flags as flags
import torch
import torch.nn.functional as F
import yaml

from mgen3d.nerf.nerf import NeRF
from mgen3d.nerf.utils import *
from mgen3d.diffusion import StableDiffusion
from mgen3d.depth import DPT

FLAGS = flags.FLAGS
flags.DEFINE_string('config_path', "./config/default.yaml", 'Path to the config file')

flags.mark_flag_as_required('config_path')

class Pipeline:
    def __init__(self, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nerf_estimator = construct_estimator(config, device=device)
        nerf_network = construct_network(config, device=device)
        nerf_optimizer = construct_optimizer(config, nerf_network)
        nerf_scheduler = construct_scheduler(config, nerf_optimizer)
        self.nerf = NeRF(config,
                         estimator=nerf_estimator,
                         network=nerf_network,
                         optimizer=nerf_optimizer,
                         scheduler=nerf_scheduler
                        )
        self.stable_diffusion = StableDiffusion(config, device=device)
        self.depth_model = DPT(config, device=device)
        
    def train(self):
        pass
    
    def train_epoch(self):
        pass
        
    def train_single_step(self):
        pass
    
    def pixel_wise_loss(self, rendered_image, target_image, mask=None):
        if mask is None:
            mask = torch.ones_like(target_image)
        loss = F.l1_loss(rendered_image * mask, target_image, reduction='mean')
        return loss
    
    def depth_loss(self):
        pass

def main():
    flags.FLAGS(sys.argv)
    with open(FLAGS.config_path, "r") as file:
        config = yaml.safe_load(file)
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    main()
