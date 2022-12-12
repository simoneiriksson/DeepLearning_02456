import numpy as np
from torch import nn, Tensor
import torch
from torch.distributions import Distribution, Exponential, Cauchy, HalfCauchy, Normal
from gmfpp.models.PrintSize import *
from typing import List, Set, Dict, Tuple, Optional, Any


class ConvVariationalAutoencoder(nn.Module):
    def update_(self):
        pass
    
   
    def __init__(self, input_shape, latent_features:int) -> None:
        super(ConvVariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        self.observation_shape = input_shape
        self.input_channels = input_shape[0]
        
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            # now we are at 68h * 68w * 3ch
            #PrintSize(),
            nn.Conv2d(in_channels=self.input_channels, out_channels=16, kernel_size=5, padding=0),
            # Now we are at: 64h * 64w * 32ch
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(16),
            #PrintSize(),

            # Now we are at: 32h * 32w * 32ch
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            #PrintSize(),

            # Now we are at: 16h * 16w * 32ch
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            #PrintSize(),
            # Now we are at: 8h * 8w * 64ch

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(128),
            # Now we are at: 4h * 4w * 128ch
            #PrintSize(),

            #nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2),
            #nn.MaxPool2d(2, stride=2),
            #nn.LeakyReLU(negative_slope=0.01),
            #nn.BatchNorm2d(128),
            # Now we are at: 2h * 2w * 128ch = 512
            nn.Flatten(),
            nn.Linear(2048, latent_features*2)
            # Now we are at: 512
            ,PrintSize()
        )
        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            PrintSize(),
            # Now we are at: 256ch
            nn.Linear(256, 256),
            nn.Unflatten(1,(256,1,1)),
            nn.PixelShuffle(2),
            PrintSize(),
            # Now we are at: 2h * 2w * 64ch
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),
            PrintSize(),

            # Now we are at: 4h * 4w * 64ch
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            PrintSize(),

            # Now we are at: 8h * 8w * 32ch
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01),
            PrintSize(),

            # Now we are at: 16h * 16w * 32ch
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, padding=2, stride=2, output_padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.01),
            PrintSize(),
            
            # Now we are at: 32h * 32w * 32ch
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=5, padding=0, stride=2, output_padding=1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(negative_slope=0.01),
            PrintSize()
            # Now we are at: 68h * 68w * 32ch
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        

    def observation(self, z:Tensor) -> Tensor:
        """return the distribution `p(x|z)`"""
        #h_z = self.decoder(z)
        #mu, log_sigma = h_z.chunk(2, dim=-1)
        mu = self.decoder(z)
        mu = mu.view(-1, *self.input_shape) # reshape the output
        #log_sigma = log_sigma.view(-1, *self.input_shape) # reshape the output
        return mu

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        h_x = self.encoder(x)
        qz_mu, qz_log_sigma =  h_x.chunk(2, dim=-1)        
        eps = torch.empty_like(qz_mu).normal_()
        z = qz_mu + qz_log_sigma.exp() * eps
        
        x_hat = self.observation(z)
        
        return {'x_hat': x_hat, 'qz_log_sigma': qz_log_sigma, 'qz_mu': qz_mu, 'z': z}
    
    
