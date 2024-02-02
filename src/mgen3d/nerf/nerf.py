import nerfacc
from nerfacc import OccGridEstimator
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import tqdm
import yaml

from mgen3d.nerf.ray_utils import generate_rays, get_chunks
from mgen3d.metrics import calc_psnr_from_mse
from mgen3d.pose_generator_utils import create_view_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeRF:
    def __init__(
        self,
        config: yaml.YAMLObject,
        estimator: OccGridEstimator,
        network: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
    ):
        self._estimator = estimator
        self._network = network
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._grad_scaler = torch.cuda.amp.GradScaler(
            config["optimizer"]["grad_scaler"]
        )
        self.grad_norm_clip = config["optimizer"]["grad_norm_clip"]
        # TODO: look into implementing log-linear annealing, as described in Mip-NeRF 360 paper
        self.warm_up_steps = config["optimizer"]["warm_up_steps"]
        self.scheduler_step_period = config["optimizer"]["scheduler_step_period"]
        self.loss_fn = nn.functional.smooth_l1_loss
        self.checkpoint_path = config["paths"]["checkpoint_path"]
        self.output_image_path = config["paths"]["output_image_path"]

        # store scene parameters
        self.white_background = config["scene"]["white_background"]

        # ray parameters
        self.chunk_size = config["rays"]["chunk_size"]
        self.num_rays = config["rays"]["num_rays"]

        # sampling parameters
        self.near_plane_ = config["sampling"]["near_plane"]
        self.far_plane_ = config["sampling"]["far_plane"]
        self.early_stop_eps_ = config["sampling"]["early_stop_eps"]
        self.alpha_thresh_ = config["sampling"]["alpha_thresh"]
        self.render_step_size_ = config["sampling"]["render_step_size"]
        self.stratified_sampling_ = config["sampling"]["stratified_sampling"]
        self.sampling_cone_angle_ = config["sampling"]["sampling_cone_angle"]
        
        # eval parameters
        self.eval_period = config["evaluation"]["period"]

        self._step = 0

    def forward_pass(self, input_dict: dict, ray_coords: torch.Tensor = None):
        """Forward pass through the pipeline.

        Args:
            input_dict (torch.Tensor): dictionary that contains: transform from camera to world (pose),
                                        image height, image width, camera intrinsic matrix
            ray_coords (torch.Tensor): random ray sample coordinates. Defaults to None.

        Returns:
            rgb_map (torch.Tensor): rendered rgb image
        """
        rays_o, rays_d = generate_rays(input_dict, device=device)
        if ray_coords is not None:
            rays_o = rays_o[ray_coords[:, 0], ray_coords[:, 1]]  # (num_rays, 3)
            rays_d = rays_d[ray_coords[:, 0], ray_coords[:, 1]]  # (num_rays, 3)
            output_shape = (rays_d.shape[0], 3)
        else:
            output_shape = (input_dict["H"], input_dict["W"], 3)

        rgb_map = self.render_image(rays_o, rays_d, output_shape, self.chunk_size)

        return rgb_map

    def train(
        self,
        num_epochs: int,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        checkpoint_to_load: str = None,
    ):
        """Train the NeRF model.
        Args:
            num_epochs (int): number of epochs to train for
            images (list): a list of images to train on
            poses (list): a list of the corresponding poses for the images
            checkpoint_to_load (str, optional): A checkpoint to load during training. Defaults to None
        """
        print("start training")
        if checkpoint_to_load is None:
            starting_epoch = 0
        else:
            print("loading checkpoint: {}".format(checkpoint_to_load))
            starting_epoch = self.load_checkpoint(checkpoint_to_load)

        loss_list = []  # training along iterations
        psnr_list = []  # validation along epochs
        for epoch in tqdm.trange(starting_epoch, num_epochs):
            train_loss = self.train_epoch(train_dataset, epoch)
            loss_list.append(train_loss)

            if epoch % self.eval_period == 0 and epoch != 0:
                # run eval loop
                eval_psnr = self.eval(eval_dataset)
                psnr_list.append(eval_psnr)
                print(
                    "epoch: {}, loss: {}, psnr: {}".format(epoch, train_loss, eval_psnr)
                )
                test_sample = next(iter(eval_dataset))
                self.create_gif(
                    [
                        create_view_data(
                            test_sample["H"],
                            test_sample["W"],
                            test_sample["K"],
                            angle,
                            -30.0,
                            4.0,
                        )
                        for angle in np.linspace(-180, 180, 40 + 1)[:-1]
                    ]
                )

    def train_epoch(self, dataset: Dataset, epoch_num: int):
        """Train the NeRF model for one epoch.

        Args:
            images (list): a list of images to train on
            poses (list): a list of the corresponding poses for the images
            epoch_num (int): the current epoch number
        """
        self._network.train()
        self._estimator.train()
        train_loss = 0.0

        # create index list and shuffle before iteratinng through images
        dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=0)
        for sample in tqdm.tqdm(dataloader):
            train_loss += self.train_one_step(sample, epoch_num)
        train_loss /= len(dataset)
        self.save_checkpoint(epoch_num, train_loss)

        return train_loss
    
    def train_one_step(self, sample: dict, epoch_num: int):
        # update occupancy grid
        def occ_eval_fn(x):
            density = self._network(x)
            occ = density * self.render_step_size_
            return occ

        self._estimator.update_every_n_steps(
            step=self._step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=1e-2,
        )
        self._step = self._step + 1

        # get image
        image = sample["imgs"].squeeze().to(device)

        # generate random ray sample coordinates
        if self.num_rays is None:
            ray_coords = None
            target = image
        else:
            H = sample["H"].item()
            W = sample["W"].item()
            coords = torch.stack(
                torch.meshgrid(
                    torch.linspace(0, H - 1, H),
                    torch.linspace(0, W - 1, W),
                    indexing="ij",
                ),
                -1,
            )  # (H, W, 2)
            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
            select_inds = np.random.choice(
                coords.shape[0], size=[self.num_rays], replace=False
            )  # (num_rays,)
            ray_coords = coords[select_inds].long()  # (num_rays, 2)
            target = image[ray_coords[:, 0], ray_coords[:, 1]]  # (num_rays, 3)

        with torch.cuda.amp.autocast():
            rgb_map = self.forward_pass(sample, ray_coords)
            loss = self.loss_fn(rgb_map, target)
        self.optimize(loss, epoch_num)
        return loss.item()

    def optimize(self, loss: torch.Tensor, epoch: int):
        self._optimizer.zero_grad()
        self._grad_scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(self._network.parameters(), self.grad_norm_clip)
        self._optimizer.step()
        self._scheduler.step()

    def eval(self, dataset: Dataset):
        """Evaluate the NeRF model, calculating the PSNR for the given images and poses.

        Args:
            images (list): a list of images to validate on
            poses (list): a list of the corresponding poses for the images

        Returns:
            psnr: the average PSNR for the given images and poses
        """

        self._network.eval()
        self._estimator.eval()
        psnr = 0.0
        with torch.no_grad():
            dataloader = DataLoader(dataset, 1, shuffle=True, num_workers=0)
            for sample in tqdm.tqdm(dataloader):
                image = sample["imgs"].squeeze().to(device)
                rgb_map = self.forward_pass(sample, None)
                mse_loss = self.loss_fn(rgb_map, image)
                psnr += calc_psnr_from_mse(mse_loss)
        psnr /= len(dataset)

        return psnr

    def render_image(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        output_shape: tuple,
        chunk_size: int,
    ) -> torch.Tensor:
        def sigma_fn(
            t_starts: torch.Tensor, t_ends: torch.Tensor, ray_indices: torch.Tensor
        ) -> torch.Tensor:
            """Define how to query density for the estimator."""
            t_origins = ray_o_chunk[ray_indices]  # (n_samples, 3)
            t_dirs = ray_d_chunk[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self._network(positions)
            return sigmas  # (n_samples,)

        def rgb_sigma_fn(
            t_starts: torch.Tensor, t_ends: torch.Tensor, ray_indices: torch.Tensor
        ):
            """Query rgb and density values from a user-defined radiance field."""
            t_origins = ray_o_chunk[ray_indices]  # (n_samples, 3)
            t_dirs = ray_d_chunk[ray_indices]  # (n_samples, 3)
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgb_sigmas = self._network(positions, t_dirs)
            rgbs, sigmas = torch.split(rgb_sigmas, [3, 1], dim=-1)
            return rgbs, sigmas.squeeze(-1)  # (n_samples, 3), (n_samples,)

        # chunking
        ray_o_chunks = get_chunks(rays_o, chunk_size)
        ray_d_chunks = get_chunks(rays_d, chunk_size)

        # neural radiance field querying
        chunked_outputs = []
        for ray_o_chunk, ray_d_chunk in zip(ray_o_chunks, ray_d_chunks):
            ray_o_chunk = ray_o_chunk.reshape(-1, ray_o_chunk.shape[-1])
            ray_d_chunk = ray_d_chunk.reshape(-1, ray_d_chunk.shape[-1])
            ray_indices, t_starts, t_ends = self._estimator.sampling(
                ray_o_chunk,
                ray_d_chunk,
                near_plane=self.near_plane_,
                far_plane=self.far_plane_,
                sigma_fn=sigma_fn,
                early_stop_eps=self.early_stop_eps_,
                alpha_thre=self.alpha_thresh_,
                render_step_size=self.render_step_size_,
                stratified=self.stratified_sampling_,
                cone_angle=self.sampling_cone_angle_,
            )

            # Differentiable Volumetric Rendering.
            # colors: (n_rays, 3). opacity: (n_rays, 1). depth: (n_rays, 1).
            color, opacity, depth, extras = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=ray_o_chunk.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=torch.tensor([1.0, 1.0, 1.0]).to(device),
            )
            chunked_outputs.append(color)
        outputs = torch.cat(chunked_outputs, dim=0).reshape(output_shape)

        return outputs

    def render_view(self, view_dict: np.ndarray):
        """Render a view from a given pose.

        Args:
            pose (np.ndarray): transform from camera to world (pose) of the view to be rendered

        Returns:
            rgb_map (np.ndarray): rendered rgb image as a numpy array
        """
        self._network.eval()
        with torch.no_grad():
            rgb_map = self.forward_pass(view_dict, None)
            rgb_map = rgb_map.cpu().numpy()
            rgb_map = np.maximum(
                np.minimum(rgb_map, np.ones_like(rgb_map)), np.zeros_like(rgb_map)
            )
            return rgb_map

    def save_checkpoint(self, epoch: int, loss: float):
        """Save a checkpoint of the current model.

        Args:
            epoch (int): the current epoch number
            loss (float): the last calculated loss
        """
        os.makedirs(self.checkpoint_path, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self._network.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
            },
            self.checkpoint_path + "epoch_{:03d}_checkpoint.pth".format(epoch),
        )

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint of the model

        Args:
            checkpoint_path (str): a path to the checkpoint to load

        Returns:
            epoch (int): the epoch number of the loaded checkpoint
        """
        checkpoint = torch.load(self.checkpoint_path + checkpoint_path)
        self._network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        return epoch

    def create_gif(self, poses: list):
        """Create a gif from a list of poses.

        Args:
            poses (list): a list of poses to render views from
        """
        os.makedirs("./outputs/gifs/", exist_ok=True)
        frames = []
        for pose in tqdm.tqdm(poses):
            frames.append(
                Image.fromarray((self.render_view(pose) * 255).astype(np.uint8))
            )

        frame_one = frames[0]
        frame_one.save(
            "./outputs/gifs/output.gif",
            format="GIF",
            append_images=frames,
            save_all=True,
            duration=10,
            loop=0,
        )
        print("\nGIF Created at Checkpoint!")
