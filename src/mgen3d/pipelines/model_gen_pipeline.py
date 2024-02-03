
import sys
import absl.flags as flags
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import PearsonCorrCoef
import tqdm
import yaml

from mgen3d.nerf.nerf import NeRF
from mgen3d.nerf.utils import *
from mgen3d.diffusion import StableDiffusion
from mgen3d.depth import DPT
from mgen3d.data.sampling_dataset import SamplingDataset

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
        
        
    def train(self,
        num_epochs: int,
        train_dataset: Dataset):  
        loss_list = []  # training along iterations
        train_dataloader = DataLoader(train_dataset, 1, shuffle=True, num_workers=0)
        for epoch in tqdm.trange(0, num_epochs):
            train_loss = self.train_epoch(train_dataset, epoch)
            loss_list.append(train_loss)
    
    def train_epoch(self):
        # if reference view
            # generate rays at reference view
            # sample rays
            # sample points along rays
            # render with nerf 
            # take pixel-wise loss
            # take depth loss
        # if sampled view
            # generate rays at sampled view
            # sample rays
            # sample points along rays
            # render with nerf
            # generate detailed description of the reference view with image captioning model (this can likely be done once ahead of time)
            # encode the rendered image into latents with the diffusion model
            # perform noise prediction with the diffusion model
                # if t step of diffusion scheduler is below threshold:
                    # perform clip loss
                # else:
                    # perform score distillaltion sampling loss
        pass
        
    def train_single_step(self):
        pass
    
    def pixel_wise_loss(self, rendered_image, target_image, mask=None):
        if mask is None:
            mask = torch.ones_like(target_image)
        loss = F.l1_loss(rendered_image * mask, target_image, reduction='mean')
        return loss
    
    def depth_loss(self, depth_pred, depth_ref, mask): 
        # use negative pearson correlation coefficient as loss
        pearson = PearsonCorrCoef()
        loss = -pearson(depth_pred[mask], depth_ref[mask])
        return loss

def main():
    flags.FLAGS(sys.argv)
    with open(FLAGS.config_path, "r") as file:
        config = yaml.safe_load(file)
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    flags.FLAGS(sys.argv)
    main()
