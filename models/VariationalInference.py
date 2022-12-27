#from utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any
import math

def reduce(x:Tensor) -> Tensor:
    """for each datapoint: sum over all dimensions"""
    flat = view_flat_samples(x)
    return flat.sum(dim=1)

class VariationalInference(nn.Module):
    def __init__(self, beta:float=1.):
        super().__init__()
        self.beta = beta
        
    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = model(x)

        px, pz, qz, z = [outputs[k] for k in ["px", "pz", "qz", "z"]]
        
        #log_px = reduce(px.log_prob(x))
        #print("((px.mu - x)**2).shape: ", ((px.mean - x)**2).shape)
        
        sigma_sqr = ((px.mean - x)**2).mean(axis=[1,2,3])
        log_px_all = -((px.mean - x)**2)/(2*px.stddev**2) - px.stddev.log() - math.log(math.pi*2)*0.5
        #log_px = -((px.mean - x)**2).sum(axis=[1,2,3])/(2*sigma_sqr) - sigma_sqr.sqrt().log() - math.log(math.pi*2)*0.5
        #log_px_all = - (math.pi * 2 * (px.mean - x) ** 2).sqrt().log() - 0.5
        log_px = log_px_all.sum(axis=[1,2,3])
        #print("x.min(), x.max(): ", x.min(), x.max())
        #print("px.mean.min(), px.mean.max(): ", px.mean.min(), px.mean.max())
        #log_px = -torch.nn.functional.binary_cross_entropy(px.mean, x, size_average = False)
        #print("log_px.shape: ", log_px.shape)
        
        #log_pz = reduce(pz.log_prob(z))
        #log_qz = reduce(qz.log_prob(z))
        
        #kl = log_qz - log_pz

        kl = - (.5 * (1 + (qz.sigma ** 2).log() - qz.mu ** 2 - qz.sigma**2)).sum(axis=[1])
        #elbo = log_px - kl
        beta_elbo = log_px - self.beta*kl
        
        # loss
        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'log_px':log_px, 'kl': kl}
            
        return loss, diagnostics, outputs
      
