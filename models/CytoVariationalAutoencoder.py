import numpy as np
from torch import nn, Tensor
import torch
from torch.distributions import Distribution, Exponential, Cauchy, HalfCauchy, Normal
from gmfpp.models.PrintSize import *
from typing import List, Set, Dict, Tuple, Optional, Any
from gmfpp.models.ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian

class CytoVariationalAutoencoder(nn.Module):
   
    def __init__(self, input_shape, latent_features: int):
        super(CytoVariationalAutoencoder, self).__init__()
        
        self.input_shape = input_shape
        self.latent_features = latent_features
        self.observation_features = np.prod(input_shape)
        self.observation_shape = input_shape
        self.input_channels = input_shape[0]
        self.epsilon = 10e-3
        
        # Inference Network
        # Encode the observation `x` into the parameters of the posterior distribution
        # `q_\phi(z|x) = N(z | \mu(x), \sigma(x)), \mu(x),\log\sigma(x) = h_\phi(x)`
        self.encoder = nn.Sequential(
            # now we are at 68h * 68w * 3ch
            nn.Conv2d(in_channels=self.input_channels, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 64h * 64w * 32ch
            nn.MaxPool2d(2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            # Now we are at: 32h * 32w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 28h * 28w * 32ch
            nn.MaxPool2d(2),
            # Now we are at: 14h * 14w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=0),
            # Now we are at: 10h * 10w * 32ch
            nn.MaxPool2d(2),
            # Now we are at: 5h * 5w * 32ch
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),

            ##Output should be 5*5*32 now.
            nn.Conv2d(in_channels=32, out_channels=2*256, kernel_size=5, padding=0),
            # Now we are at: 1h * 1w * 512ch
            nn.BatchNorm2d(2*256),
            nn.Flatten()
        )

        
        # Generative Model
        # Decode the latent sample `z` into the parameters of the observation model
        # `p_\theta(x | z) = \prod_i B(x_i | g_\theta(x))`
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256,1,1)), # Now we are at: 1h * 1w * 256ch
            nn.Conv2d(in_channels=256, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=10),

            # Now we are at: 10h * 10w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=28),

            # Now we are at: 28h * 28w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            torch.nn.UpsamplingNearest2d(size=64),

            # Now we are at: 64h * 64w * 32ch
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),
            
            # Now we are at: 68h * 68w * 32ch
            nn.Conv2d(in_channels=32, out_channels=6, kernel_size=1, padding=0), # 6 channels because 3 for mean and 3 for variance
#            nn.BatchNorm2d(6),
            nn.LeakyReLU(negative_slope=0.01)
            #nn.Sigmoid()
        )
        
        # define the parameters of the prior, chosen as p(z) = N(0, I)
        self.register_buffer('prior_params', torch.zeros(torch.Size([1, 2*latent_features])))
        
    def posterior(self, x:Tensor) -> Distribution:
        """return the distribution `q(x|x) = N(z | \mu(x), \sigma(x))`"""
        
        # compute the parameters of the posterior
        h_x = self.encoder(x)
        
        mu, log_sigma =  h_x.chunk(2, dim=-1)

        #log_sigma = torch.maximum(log_sigma, torch.ones_like(log_sigma) * -10)
        log_sigma=torch.nn.functional.leaky_relu(log_sigma, negative_slope=0.01)
        # return a distribution `q(x|x) = N(z | \mu(x), \sigma(x))`
        return ReparameterizedDiagonalGaussian(mu, log_sigma, epsilon=self.epsilon)
    
    def prior(self, batch_size:int=1)-> Distribution:
        """return the distribution `p(z)`"""
        prior_params = self.prior_params.expand(batch_size, *self.prior_params.shape[-1:])
        mu, log_sigma = prior_params.chunk(2, dim=-1)
        
        # return the distribution `p(z)`
        return ReparameterizedDiagonalGaussian(mu, log_sigma)
    
    def observation_model(self, z:Tensor) -> Distribution:
        """return the distribution `p(x|z)`"""
        h_z = self.decoder(z)

        mu, log_sigma = h_z.chunk(2, dim=1)
        mu = mu.view(-1, *self.input_shape) # reshape the output
        log_sigma = log_sigma.view(-1, *self.input_shape) # reshape the output

        scale = torch.exp(log_sigma) + self.epsilon
        return Normal(loc=mu, scale=scale, validate_args=False)

    def forward(self, x) -> Dict[str, Any]:
        """compute the posterior q(z|x) (encoder), sample z~q(z|x) and return the distribution p(x|z) (decoder)"""
        
        # define the posterior q(z|x) / encode x into q(z|x)
        qz = self.posterior(x)
        
        # define the prior p(z)
        pz = self.prior(batch_size=x.size(0))
        
        # sample the posterior using the reparameterization trick: z ~ q(z | x)
        z = qz.rsample()

        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'qz': qz, 'z': z}
    
    
    def sample_from_prior(self, batch_size:int=100):
        """sample z~p(z) and return p(x|z)"""
        
        # degine the prior p(z)
        pz = self.prior(batch_size=batch_size)
        
        # sample the prior 
        z = pz.rsample()
        
        # define the observation model p(x|z) = B(x | g(z))
        px = self.observation_model(z)
        
        return {'px': px, 'pz': pz, 'z': z}