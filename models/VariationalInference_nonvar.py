from utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any
import math


def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    flat = view_flat_samples(x)
    return flat.sum(dim=1)

class VariationalInference_nonvar(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta

    def update_vi(self):
        pass
            
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = model(x)

        x_hat, qz_log_sigma, qz_mu, z = [outputs[k] for k in ["x_hat", "qz_log_sigma", "qz_mu", "z"]]
        qz_sigma = qz_log_sigma.exp()
        
        #mse_loss_all = (math.pi * 2 * (x_hat - x) ** 2).sqrt().log() + 0.5
        #mse_loss = mse_loss_all.sum(axis=[1,2,3])
        
        mse_loss = ((x_hat - x)**2).sum(axis=[1,2,3])
        #print("qz_sigma.shape: ", qz_sigma.shape)
        #print("mse_loss.shape", mse_loss.shape)        
        kl = - (.5 * (1 + (qz_sigma ** 2).log() - qz_mu ** 2 - qz_sigma**2)).sum(axis=[1])

        #elbo = log_px - kl
        #beta_elbo = log_px - self.beta * kl
        beta_elbo = -mse_loss - self.beta * kl
        #beta_elbo = -mse_loss
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'mse_loss':mse_loss, 'kl': kl}
            
        return loss, diagnostics, outputs
      
