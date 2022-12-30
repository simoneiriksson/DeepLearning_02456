from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus, relu
from torch.distributions import Distribution, Normal
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset

from models.ReparameterizedDiagonalGaussian import ReparameterizedDiagonalGaussian
from models.CytoVariationalAutoencoder_nonvar import CytoVariationalAutoencoder_nonvar
from models.VariationalAutoencoder import VariationalAutoencoder
from models.SparseVariationalAutoencoder import SparseVariationalAutoencoder
from models.ConvVariationalAutoencoder import ConvVariationalAutoencoder
from models.VariationalInference_nonvar import VariationalInference_nonvar
from models.VariationalInferenceSparseVAE import VariationalInferenceSparseVAE

from models.LoadModels import LoadVAEmodel, initVAEmodel, initVAEmodel_old, LoadVAEmodel_old
from utils.data_transformers import normalize_every_image_channels_seperately_inplace
from utils.data_transformers import SingleCellDataset
from utils.plotting import plot_VAE_performance, plot_image_channels
from utils.training import create_directory, read_metadata, get_relative_image_paths, load_images
from utils.training import get_MOA_mappings, shuffle_metadata, split_metadata
from utils.utils import cprint, get_datetime, create_logfile, constant_seed, StatusString, DiscStatusString
from utils.data_preparation import get_server_directory_path
import importlib
######### Utilities #########

constant_seed()
datetime = get_datetime()
output_folder = "dump/outputs_{}/".format(datetime)
create_directory(output_folder)
logfile = create_logfile(output_folder + "log.log")
cprint("output_folder is: {}".format(output_folder), logfile)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}", logfile)
########## loading data #########

#path = get_server_directory_path()
path = "../data/all/"

#if metadata is sliced, then torch.load load can't be used. Instead, use images = load_images(...
metadata = read_metadata(path + "metadata.csv") #refactor? dtype=dataframe
metadata =shuffle_metadata(metadata)[:200]
cprint("loaded metadata",logfile)

cprint("loading images", logfile)
relative_paths = get_relative_image_paths(metadata) #refactor?
image_paths = [path + relative for relative in relative_paths] #absolute path
images = load_images(image_paths, verbose=True, log_every=10000, logfile=logfile)
#images = torch.load("../data/images.pt") #TODO SIZE OF TENSOR??
#create_directory('../data/') #refactor?? 
#torch.save(images, '../data/images.pt')
mapping = get_MOA_mappings(metadata) #sorts the metadata by moas
cprint("loaded images", logfile)

# With the below command, we normalize all the images, image- and channel-wise.
# Alternative, this can be uncommented and like in the Lafarge article, we can do batchwise normalization
normalize_every_image_channels_seperately_inplace(images, verbose=True)
cprint("normalized images", logfile)

metadata = shuffle_metadata(metadata)
metadata_train, metadata_validation = split_metadata(metadata, split_fraction = .90)
#metadata_train, metadata_validation = metadata, metadata

train_set = SingleCellDataset(metadata_train, images, mapping)
validation_set = SingleCellDataset(metadata_validation, images, mapping)

######### VAE Configs #########
cprint("VAE Configs", logfile)

# Config CytoVAE
params_VAEGAN = {
    'num_epochs' : 10,
    'batch_size' : min(64, len(train_set)),
    'learning_rate' : 1e-3,
    'weight_decay' : 1e-3,
    'image_shape' : np.array([3, 68, 68]),
    'latent_features' : 256,
    'model_type' : "Cyto_VAEGAN",
    'alpha': 0.05, 
    'alpha_max': 0.05,
    'beta': 0.5, 
    'beta_max': 1,
    'p_norm': 2
    }


[CytoVAE, DISCmodel], validation_data, training_data, params, vi_VAEGAN = initVAEmodel(params_VAEGAN)
cprint("params: {}".format(params_VAEGAN), logfile)
CytoVAE = CytoVAE.to(device)
DISCmodel = DISCmodel.to(device)

CytoVAE_optimizer = torch.optim.Adam(CytoVAE.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
DISCmodel_optimizer = torch.optim.Adam(DISCmodel.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

train_loader = DataLoader(train_set, batch_size=params_VAEGAN['batch_size'], shuffle=True, num_workers=0, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=max(2, params_VAEGAN['batch_size']), shuffle=False, num_workers=0, drop_last=False)

######### VAE Training #########
cprint("VAE Training", logfile)

num_epochs = params['num_epochs']
batch_size = params['batch_size']

print_every = 1

beta = params_VAEGAN['beta']



for epoch in range(num_epochs):
    training_epoch_data = defaultdict(list)
    disc_training_epoch_data = defaultdict(list)
    disc_data = defaultdict(list)

    _ = CytoVAE.train()
    _ = DISCmodel.train()
    for x, _ in train_loader:
        x = x.to(device)
        losses_mean, losses, outputs = vi_VAEGAN(CytoVAE, DISCmodel, x)

        # unfolding losses:
        image_loss = losses_mean['image_loss']
        kl_div = losses_mean['kl_div']
        disc_loss = losses_mean['disc_loss']
        disc_repr_loss = losses_mean['disc_repr_loss']

        loss_VAE = disc_repr_loss + image_loss + kl_div * 1.0

        CytoVAE_optimizer.zero_grad()
        loss_VAE.backward()
        #_ = nn.utils.clip_grad_norm_(CytoVAE.parameters(), 1_000)
        CytoVAE_optimizer.step()

        loss_discriminator = disc_loss

        DISCmodel_optimizer.zero_grad()    
            
        loss_discriminator.backward()
        #_ = nn.utils.clip_grad_norm_(DISCmodel.parameters(), 1_000)

        DISCmodel_optimizer.step()

        for k, v in losses.items():
            training_epoch_data[k] += list(v.cpu().data.numpy())
        disc_data['disc_false_negatives'] = (1 - outputs['disc_real_pred'])
        disc_data['disc_true_positives'] = outputs['disc_real_pred']
        disc_data['disc_true_negatives'] = (1 - outputs['disc_fake_pred'])
        disc_data['disc_false_positives'] = outputs['disc_fake_pred']
        for k, v in disc_data.items():
            disc_training_epoch_data[k] += list(v.cpu().data.numpy())

    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]
    for k, v in disc_training_epoch_data.items():
        training_data[k] += [np.sum(disc_training_epoch_data[k])]

    with torch.no_grad():
        validation_epoch_data = defaultdict(list)
        disc_validation_epoch_data = defaultdict(list)
        _ = CytoVAE.eval()
        _ = DISCmodel.eval()        
        for x, _ in validation_loader:
            # batchwise normalization. Only to be used if imagewise normalization has been ocmmented out.
            # x = batch_normalize_images(x)
            x = x.to(device)
            losses_mean, losses, outputs = vi_VAEGAN(CytoVAE, DISCmodel, x)

            # unfolding losses:
            image_loss = losses_mean['image_loss']
            kl_div = losses_mean['kl_div']
            disc_loss = losses_mean['disc_loss']
            disc_repr_loss = losses_mean['disc_repr_loss']
            loss_VAE = disc_repr_loss + image_loss + kl_div * 1.0
            loss_discriminator = disc_repr_loss

            for k, v in losses.items():
                validation_epoch_data[k] += list(v.cpu().data.numpy())
            disc_data['disc_false_negatives'] = (1 - outputs['disc_real_pred'])
            disc_data['disc_true_positives'] = outputs['disc_real_pred']
            disc_data['disc_true_negatives'] = (1 - outputs['disc_fake_pred'])
            disc_data['disc_false_positives'] = outputs['disc_fake_pred']
            for k, v in disc_data.items():
                disc_validation_epoch_data[k] += list(v.cpu().data.numpy())

        for k, v in validation_epoch_data.items():
            validation_data[k] += [np.mean(validation_epoch_data[k])]
        
        for k, v in disc_validation_epoch_data.items():
            validation_data[k] += [np.sum(disc_validation_epoch_data[k])]
        
        if epoch % print_every == 0:
            cprint("\n", logfile)
            cprint(f"epoch: {epoch}/{num_epochs}", logfile)
            train_string = StatusString("training", training_epoch_data)
            evalString = StatusString("evaluation", validation_epoch_data)
            cprint(train_string, logfile)
            cprint(evalString, logfile)

            train_string = DiscStatusString("training Discriminator accurracy", disc_training_epoch_data)
            evalString = DiscStatusString("evaluation Discriminator accurracy", disc_validation_epoch_data)
            cprint(train_string, logfile)
            cprint(evalString, logfile)


            #cprint("vi.beta: {}".format(vi.beta), logfile)
            #cprint("vi.alpha: {}".format(vi.alpha), logfile)        

    CytoVAE.update_()
    DISCmodel.update_()
    vi_VAEGAN.update_vi()


cprint("finished training", logfile)
######### Save VAE parameters #########
cprint("Save VAE parameters", logfile)

datetime = get_datetime()

torch.save(CytoVAE.state_dict(), output_folder + "vae_parameters.pt")
torch.save(DISCmodel.state_dict(), output_folder + "disc_parameters.pt")
torch.save(validation_data, output_folder + "validation_data.pt")
torch.save(training_data, output_folder + "training_data.pt")
torch.save(params, output_folder + "params.pt")

######### extract a few images already #########
cprint("Extract a few images already", logfile)
create_directory(output_folder + "images")

_ = CytoVAE.eval() # because of batch normalization
plot_VAE_performance(training_data, file=output_folder + "images/training_data.png", title='VAE - learning')
plot_VAE_performance(validation_data, file=output_folder + "images/validation_data.png", title='VAE - validation')

n = 10
for i in range(n):
    x, y = train_set[i]
    plot_image_channels(x, file=output_folder + "images/x_{}.png".format(i))
    #plot_image_channels(x)
    x = x.to(device)
    outputs = CytoVAE(x[None,:,:,:])
    x_hat = outputs["x_hat"]
    x_reconstruction = x_hat
    x_reconstruction = x_reconstruction[0].detach()
    plot_image_channels(x_reconstruction.cpu(), file=output_folder + "images/x_reconstruction_{}.png".format(i))
    #plot_image_channels(x_reconstruction.cpu())

cprint("saved images", logfile)
cprint("output_folder is: {}".format(output_folder), logfile)
cprint("script done.", logfile)

