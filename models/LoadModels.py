from models.CytoVariationalAutoencoder import *
from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict

from models.CytoVariationalAutoencoder import CytoVariationalAutoencoder
from models.CytoVariationalAutoencoder_nonvar import CytoVariationalAutoencoder_nonvar
from models.VariationalAutoencoder import VariationalAutoencoder
from models.ConvVariationalAutoencoder import ConvVariationalAutoencoder
from models.SparseVariationalAutoencoder import SparseVariationalAutoencoder
from models.VariationalInference_nonvar import VariationalInference_nonvar
from models.VariationalInference import VariationalInference


def LoadVAEmodel(folder, model_type=None, device="cpu", ):
    params = torch.load(folder + "params.pt", map_location=torch.device(device))

    validation_data = torch.load(folder + "validation_data.pt", map_location=torch.device(device))
    training_data = torch.load(folder + "training_data.pt", map_location=torch.device(device))
    
    model_type = params['model_type']
    if (model_type == None) or model_type == "Cyto":
        vae = CytoVariationalAutoencoder(params['image_shape'], params['latent_features'])
    if model_type == 'Cyto_nonvar':
        vae = CytoVariationalAutoencoder_nonvar(params['image_shape'], params['latent_features'])
        vi = VariationalInference_nonvar(beta=params['beta'])
    if model_type == 'basic':
        vae = VariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference(beta=params['beta'])
    if model_type == 'Conv_simon':
        vae = ConvVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference_nonvar(beta=params['beta'])
    if model_type == 'SparseVAE':
        vae = SparseVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference_nonvar(beta=params['beta'])
    
    vae.load_state_dict(torch.load(folder + "vae_parameters.pt", map_location=torch.device(device)))
    return vae, validation_data, training_data, params, vi


def initVAEmodel(params):

    model_type = params['model_type']

    training_performance = defaultdict(list)
    validation_performance = defaultdict(list)

    if (model_type == None) or model_type == "Cyto":
        vae = CytoVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference(beta=params['beta'])
    if model_type == 'Cyto_nonvar':
        vae = CytoVariationalAutoencoder_nonvar(params['image_shape'], params['latent_features'])
        vi = VariationalInference_nonvar(beta=params['beta'])
    if model_type == 'basic':
        vae = VariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference(beta=params['beta'])
    if model_type == 'Conv_simon':
        vae = ConvVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference_nonvar(beta=params['beta'])
    if model_type == 'SparseVAE':
        vae = SparseVariationalAutoencoder(params['image_shape'], params['latent_features'])
        vi = VariationalInference_nonvar(beta=params['beta'])
    
    return vae, validation_performance, training_performance, params, vi