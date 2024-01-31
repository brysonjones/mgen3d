import sys

import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
from yaml import YAMLObject
import yaml
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class _TruncExp(Function):
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = torch.exp(x)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        # TODO: clean up magic numbers (make this tunable?)
        return g * torch.exp(torch.clamp(x, min=-15, max=15))

trunc_exp = _TruncExp.apply

class TinyCudaNetwork(nn.Module):
    def __init__(
        self,
        config: YAMLObject,
        aabb_scale,
        input_channels=3,
        input_channels_views=3,
    ) -> None:
        super().__init__()
        self.x_input_size = input_channels
        self.d_input_size = input_channels_views
        self.aabb_scale = aabb_scale
        per_level_scale = torch.exp2(
            torch.log2(torch.tensor(2048 * self.aabb_scale / 16)) / (16 - 1)
        )  # from Instant-NGP

        config["pos_encoding"]["per_level_scale"] = per_level_scale.tolist()

        NUM_DENSITY_NETWORK_OUTPUTS = 16
        self.density_layers_with_encoding = tcnn.NetworkWithInputEncoding(
            self.x_input_size,
            NUM_DENSITY_NETWORK_OUTPUTS,
            config["pos_encoding"],
            config["density_network"],
        )
        self.dir_encoding = tcnn.Encoding(self.d_input_size, config["dir_encoding"])

        NUM_RGB_NETWORK_OUTPUTS = 3
        self.rgb_layers = tcnn.Network(
            self.density_layers_with_encoding.n_output_dims
            + self.dir_encoding.n_output_dims,
            NUM_RGB_NETWORK_OUTPUTS,
            config["rgb_network"],
        )
        self.density_activation = lambda x: trunc_exp(x)
        self.rgb_activation = F.sigmoid

    def forward(
        self, position: torch.Tensor, direction: torch.Tensor = None
    ) -> torch.Tensor:
        rescaled_position = (position + self.aabb_scale) / ( 
            2 * self.aabb_scale
        )  # to [0, 1]
        density_output = self.density_layers_with_encoding(rescaled_position)
        if direction is None:
            density = self.density_activation(density_output[:, 0]) * 2.0
            return density

        # tcnn requires directions in the range [0, 1]
        rescaled_direction = (direction + 1.0) / 2.0
        dir_encoding_output = self.dir_encoding(rescaled_direction)
        rgb_input = torch.cat([dir_encoding_output, density_output], dim=-1)
        rgb_output = self.rgb_layers(rgb_input)
        outputs = torch.cat(
            [
                self.rgb_activation(rgb_output),
                self.density_activation(density_output[:, 0]).unsqueeze(-1),
            ],
            -1,
        )

        return outputs

    
class VanillaNetwork(nn.Module):
    def __init__(
        self, config: YAMLObject, pos_encoder: nn.Module, dir_encoder: nn.Module
    ) -> None:
        super().__init__()
        self.pos_encoder = pos_encoder
        self.dir_encoder = dir_encoder
        self.x_input_size = pos_encoder.output_size
        self.d_input_size = dir_encoder.output_size
        self.skip_conn_layers = config["network"]["skip_connection_layers"]
        self.hidden_size = config["network"]["hidden_size"]
        self.num_layers = config["network"]["num_layers"]

        # TODO clean this definition up -- this is messy
        self.pos_linears = nn.ModuleList(
            [nn.Linear(self.x_input_size, self.hidden_size)]
            + [
                nn.Linear(self.hidden_size, self.hidden_size)
                if i not in self.skip_conn_layers
                else nn.Linear(
                    self.hidden_size + self.x_input_size, self.hidden_size
                )
                for i in range(self.num_layers - 1)
            ]
        )
        self.viewing_dir_linears = nn.ModuleList(
            [nn.Linear(self.d_input_size + self.hidden_size, self.hidden_size // 2)]
        )

        self.feature_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.sigma_linear = nn.Linear(self.hidden_size, 1)  # for density estimation
        self.rgb_linear = nn.Linear(
            self.hidden_size // 2, 3
        )  # for color estimation

        self.sigma_activation = nn.ReLU(True)
        self.rgb_activation = nn.Sigmoid()

    def forward(
        self, position: torch.Tensor, direction: torch.Tensor = None
    ) -> torch.Tensor:
        # perform encoding
        x_encoded = self.pos_encoder(position)

        h = x_encoded
        for i, l in enumerate(self.pos_linears):
            h = self.pos_linears[i](h)
            h = F.relu(h, True)
            if i in self.skip_conn_layers:
                h = torch.cat([x_encoded, h], -1)

        sigma = self.sigma_activation(self.sigma_linear(h))
        if direction is None:
            return sigma.squeeze(-1)

        d_encoded = self.dir_encoder(direction)

        feature = self.feature_linear(h)
        h = torch.cat([feature, d_encoded], -1)

        for i, l in enumerate(self.viewing_dir_linears):
            h = self.viewing_dir_linears[i](h)
            h = F.relu(h, True)

        rgb = self.rgb_activation(self.rgb_linear(h))
        outputs = torch.cat([rgb, sigma], -1)

        return outputs
