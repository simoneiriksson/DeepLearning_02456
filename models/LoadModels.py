from gmfpp.models.CytoVariationalAutoencoder import *
from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict

from gmfpp.models.CytoVariationalAutoencoder import CytoVariationalAutoencoder
from gmfpp.models.CytoVariationalAutoencoder_nonvar import CytoVariationalAutoencoder_nonvar
from gmfpp.models.VariationalAutoencoder import VariationalAutoencoder
from gmfpp.models.ConvVariationalAutoencoder import ConvVariationalAutoencoder
from gmfpp.models.SparseVariationalAutoencoder import SparseVariationalAutoencoder

def LoadVAEmodel(folder, model_type=None, device="cpu"):
    validation_data = torch.load(folder + "validation_data.pt", map_location=torch.device(device))
    training_data = torch.load(folder + "training_data.pt", map_location=torch.device(device))
    VAE_settings = torch.load(folder + "VAE_settings.pt", map_location=torch.device(device))
    if "model_type" in VAE_settings.keys(): model_type = VAE_settings['model_type']
    if (model_type == None) or model_type == "Cyto":
        vae = CytoVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'Cyto_nonvar':
        vae = CytoVariationalAutoencoder_nonvar(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'basic':
        vae = VariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'Conv_simon':
        vae = ConvVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'SparseVAE':
        vae = SparseVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    
    vae.load_state_dict(torch.load(folder + "vae_parameters.pt", map_location=torch.device(device)))
    return vae, validation_data, training_data, VAE_settings


def initVAEmodel(latent_features= 256,
                    beta = 1.,
                    num_epochs = 1000,
                    batch_size = 32,
                    learning_rate = 1e-3,
                    weight_decay = 10e-4,
                    image_shape = np.array([3, 68, 68]),
                    model_type = "Cyto"):

    VAE_settings = {
        'latent_features' : latent_features,
        'beta' : beta,
        'num_epochs' : num_epochs,
        'batch_size' : batch_size,
        'learning_rate' : learning_rate,
        'weight_decay' : weight_decay,
        'image_shape' : image_shape,
        'model_type' : model_type
        }
        
    training_performance = defaultdict(list)
    validation_performance = defaultdict(list)

    if (model_type == None) or model_type == "Cyto":
        vae = CytoVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'Cyto_nonvar':
        vae = CytoVariationalAutoencoder_nonvar(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'basic':
        vae = VariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'Conv_simon':
        vae = ConvVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    if model_type == 'SparseVAE':
        vae = SparseVariationalAutoencoder(VAE_settings['image_shape'], VAE_settings['latent_features'])
    
    return vae, validation_performance, training_performance, VAE_settings