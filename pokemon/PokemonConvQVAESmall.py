import math
import torch
import numpy as np
import torch.nn as nn
from random import sample
from QConvLayers import SimpleQConv2dLayer
import pennylane as qml

conv_out_channels = [4,
                     32,
                     64,
                     96,
                     128]

conv_kernel_sizes = [(2, 2),
                     (2, 2),
                     (5, 5),
                     (3, 3),
                     (3, 3)]

pool_kernel_sizes = [(2, 2),
                     (2, 2),
                     (2, 2)]

layer_norm_sizes = [[19, 19],
                    [15, 15],
                    [5, 5],
                    [3, 3]]

layer_norm_sizes_de = [[92, 92],
                       [42, 42],
                       [18, 18]]

channels_in_image = 1

input_size = 40
size_before_flatten = (3, 3)
flat_size = size_before_flatten[0] * size_before_flatten[1] * conv_out_channels[-1]


class PokemonConvVAEEncoder(nn.Module):
    def __init__(self, latent_size, noise_std, q_dev):
        super(PokemonConvVAEEncoder, self).__init__()

        self.noise_std = noise_std

        self.conv_encoder = nn.Sequential(
            SimpleQConv2dLayer(channels_in_image, conv_out_channels[0], kernel_size=conv_kernel_sizes[0],
                               quantum_device=q_dev, random_rotations=4, stride=2),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Conv2d(conv_out_channels[0], conv_out_channels[1], kernel_size=conv_kernel_sizes[1]),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(layer_norm_sizes[0]),

            nn.Conv2d(conv_out_channels[1], conv_out_channels[2], kernel_size=conv_kernel_sizes[2]),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(layer_norm_sizes[1]),

            nn.AvgPool2d(pool_kernel_sizes[1]),
            nn.Conv2d(conv_out_channels[2], conv_out_channels[3], kernel_size=conv_kernel_sizes[3]),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(layer_norm_sizes[2]),

            nn.Conv2d(conv_out_channels[3], conv_out_channels[4], kernel_size=conv_kernel_sizes[4]),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.LayerNorm(layer_norm_sizes[3]),

            nn.Flatten()
        )

        self.mean_layer = nn.Linear(flat_size, latent_size)
        self.logvar_layer = nn.Linear(flat_size, latent_size)

    def reparam(self, mean, logvar):
        std = torch.exp(logvar * 0.5)
        epsilon = torch.randn_like(logvar)
        return mean + std * epsilon

    def forward(self, x):
        #input_noise = torch.normal(mean=0, std=self.noise_std, size=x.size(), device='cuda')
        #x += input_noise
        x = self.conv_encoder(x)
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparam(mean, logvar)

        return z, mean, logvar


class PokemonConvVAEDecoder(nn.Module):
    def __init__(self, latent_size, q_dev):
        super(PokemonConvVAEDecoder, self).__init__()

        self.latent_size = latent_size
        self.latent_to_flat = nn.Linear(latent_size, flat_size)
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(conv_out_channels[4], conv_out_channels[3], kernel_size=conv_kernel_sizes[4]),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),

            nn.ConvTranspose2d(conv_out_channels[3], conv_out_channels[2], kernel_size=conv_kernel_sizes[3]),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Upsample(size=tuple(layer_norm_sizes[1]), mode='bicubic'),

            nn.ConvTranspose2d(conv_out_channels[2], conv_out_channels[1], kernel_size=conv_kernel_sizes[2]),
            nn.LeakyReLU(negative_slope=0.05, inplace=True),
            nn.Upsample(size=38, mode='bicubic'),

            nn.ConvTranspose2d(conv_out_channels[1], conv_out_channels[0], kernel_size=conv_kernel_sizes[1]),
            nn.ConvTranspose2d(conv_out_channels[0], channels_in_image, kernel_size=conv_kernel_sizes[0]),

            nn.Sigmoid()
        )

    def forward(self, z):
        y = self.latent_to_flat(z)
        y = y.view((y.size(0), conv_out_channels[-1], size_before_flatten[0], size_before_flatten[1]))
        y = self.conv_decoder(y)
        return y

    def generate_from_z(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(dim=0)
        if z.dim() != 2:
            raise Exception("Invalid 'z' tensor size!")
        y = self.forward(z.cuda()).squeeze().cpu()
        return y

    def generate(self, number_of_examples):
        z = torch.randn(size=(number_of_examples, self.latent_size))
        return self.generate_from_z(z)


class PokemonConvVAE(nn.Module):
    def __init__(self, latent_size, noise_std):
        super(PokemonConvVAE, self).__init__()

        q_dev = qml.device('default.qubit',
                           wires=conv_kernel_sizes[0][0] * conv_kernel_sizes[0][1],
                           c_dtype=np.complex64,
                           r_dtype=np.float32)

        self.encoder = PokemonConvVAEEncoder(latent_size, noise_std, q_dev)
        self.decoder = PokemonConvVAEDecoder(latent_size, q_dev)

    def forward(self, x):
        z, mean, logvar = self.encoder(x)
        y = self.decoder(z)
        return y, mean, logvar

    def generate_from_z(self, z):
        return self.decoder.generate_from_z(z)

    def generate(self, number_of_examples):
        return self.decoder.generate(number_of_examples)
