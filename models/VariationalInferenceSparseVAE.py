#from utils.data_transformers import *
from torch import nn, Tensor
import torch
from typing import List, Set, Dict, Tuple, Optional, Any
import math


class VariationalInferenceSparseVAE(nn.Module):
    def __init__(self, beta:float=1., alpha:float=0.0, alpha_increase:float=0.1, alpha_max:float=0.5, beta_increase:float=0.5):
        super().__init__()
        self.beta = beta
        self.alpha = alpha        
        self.beta_increase = beta_increase
        self.alpha_increase = alpha_increase
        self.alpha_max = alpha_max
    
    def update_vi(self):
        self.beta = self.beta_increase + self.beta
        self.alpha = min(self.alpha_increase + self.alpha, self.alpha_max)

    def forward(self, model:nn.Module, x:Tensor) -> Tuple[Tensor, Dict]:
        outputs = model(x)

        x_hat, z, qz_log_gamma, qz_mu, qz_log_sigma = [outputs[k] for k in ['x_hat', 'z', 'qz_log_gamma', 'qz_mu', 'qz_log_sigma']]
        
        # My implementation
        #qz_gamma = qz_log_gamma.exp()
        #qz_gamma = torch.clamp(qz_log_gamma.exp(), 1e-6, 1.0 - 1e-6) 
        #KL_part1 = qz_gamma.mul(1 + qz_log_sigma * 2 - qz_mu ** 2 - qz_log_sigma.exp() ** 2)/2
        #KL_part2 = -(1 - qz_gamma).mul(((1 - self.alpha).div(1 - qz_gamma)).log())
        #KL_part3 = -qz_gamma.mul((self.alpha.div(qz_gamma)).log())
        
        # implementation from github
        qz_gamma = torch.clamp(qz_log_gamma.exp(), 1e-6, 1.0 - 1e-6) 
        KL_part1 = 0.5 * qz_gamma.mul(1 + qz_log_sigma * 2 - qz_mu ** 2 - qz_log_sigma.exp() ** 2)
        KL_part2 = (1 - qz_gamma).mul(((1 - self.alpha)/(1 - qz_gamma)).log())
        KL_part3 = qz_gamma.mul((self.alpha/qz_gamma).log())
    
    
#        meh = torch.log((1 - qz_gamma)/(1 - self.alpha))
#        meh2 = 1 - qz_gamma
#        print("torch.log((1 - qz_gamma)/(1 - self.alpha) number of nans", meh.isnan().sum(axis=[1]))
#        print("(1 - qz_gamma) number of nans", meh2.isnan().sum(axis=[1]))
#        print("KL_part1 number of nans", KL_part1.isnan().sum(axis=[1]))
#        print("KL_part2 number of nans", KL_part2.isnan().sum(axis=[1]))
#        print("KL_part3 number of nans", KL_part3.isnan().sum(axis=[1]))
#        print("qz_gamma number of nans", qz_gamma.isnan().sum(axis=[1]))
        
#        print("(1 - qz_gamma)/(1 - self.alpha))==0 number of: ", (((1 - qz_gamma)/(1 - self.alpha))==0).sum(axis=[1]))
#        print("qz_gamma", qz_gamma)
        KL = -(KL_part1 + KL_part2 + KL_part3).sum(axis=[1])
#        print("KL", KL)
#        print("KL.shape", KL.shape)
        
        mse_loss = ((x_hat - x)**2).sum(axis=[1,2,3])
#        print("mse_loss.shape", mse_loss.shape)

        beta_elbo = -self.beta * KL - mse_loss

        loss = -beta_elbo.mean()
        
        # prepare the output
        with torch.no_grad():
            diagnostics = {'elbo': beta_elbo, 'mse_loss':mse_loss, 'kl': KL}
            
        return loss, diagnostics, outputs