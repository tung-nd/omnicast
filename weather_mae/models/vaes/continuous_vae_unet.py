import torch
import torch.nn as nn
import numpy as np
from .enc_dec_cnn import Encoder, Decoder


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, dtype=self.parameters.dtype, device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3])
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean


class KLVAEUnet(nn.Module):
    def __init__(
        self,
        img_size= (128, 256),
        in_channels=69,
        out_ch=69,
        ch=64,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        z_channels=256,
        double_z=True,
        resolution=256,
        attn_resolutions=[],
        dropout=0.0,
    ):
        super().__init__()
        
        if img_size[0] == 721 and img_size[1] == 1440:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(2, 1), stride=1, padding=0)
            self.deconv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=(2, 1), stride=1, padding=0)
        else:
            self.conv = nn.Identity()
            self.deconv = nn.Identity()
        
        self.encoder = Encoder(
            ch=ch,
            img_size=img_size,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
        )
        self.decoder = Decoder(
            ch=ch,
            img_size=img_size,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolutions,
            dropout=dropout,
            in_channels=in_channels,
            resolution=resolution,
            z_channels=z_channels,
            double_z=double_z,
            out_ch=out_ch,
        )

    def forward(self, x):
        x = self.conv(x)
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        z = posterior.sample()
        x_hat = self.decoder(z)
        x_hat = self.deconv(x_hat)
        return x_hat, posterior

    def encode(self, x, sample=False):
        x = self.conv(x)
        moments = self.encoder(x)
        posterior = DiagonalGaussianDistribution(moments)
        if sample:
            return posterior.sample()
        return posterior.mean

    def encode_posterior(self, x):
        x = self.conv(x)
        moments = self.encoder(x)
        return moments
    
    def decode(self, z):
        x_hat = self.decoder(z)
        x_hat = self.deconv(x_hat)
        return x_hat

# import torch
# model = KLVAEUnet().cuda()
# x = torch.randn(2, 69, 128, 256).cuda()
# out, posterior = model(x)
# print (out.shape, posterior.mean.shape, posterior.logvar.shape)