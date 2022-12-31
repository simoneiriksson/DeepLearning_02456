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

from models.LoadModels import LoadVAEmodel, initVAEmodel
from utils.data_transformers import normalize_every_image_channels_seperately_inplace
from utils.data_transformers import normalize_channels_inplace, batch_normalize_images
from utils.data_transformers import SingleCellDataset
from utils.plotting import plot_VAE_performance, plot_image_channels
from utils.data_preparation import create_directory, read_metadata, get_relative_image_paths, load_images
from utils.data_preparation import get_MOA_mappings, shuffle_metadata, split_metadata
from utils.utils import cprint, get_datetime, create_logfile, constant_seed, StatusString
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


#######
# ## loading data #########

#path = get_server_directory_path()
path = "../data/all/"

#if metadata is sliced, then torch.load load can't be used. Instead, use images = load_images(...
metadata = read_metadata(path + "metadata.csv") #refactor? dtype=dataframe
metadata =shuffle_metadata(metadata)[:100]
cprint("loaded metadata",logfile)

cprint("loading images", logfile)
relative_paths = get_relative_image_paths(metadata) #refactor?
image_paths = [path + relative for relative in relative_paths] #absolute path
images = load_images(image_paths, verbose=True, log_every=10000, logfile=logfile)
#images = torch.load("../data/images.pt") #TODO SIZE OF TENSOR??
#create_directory('../data/') #refactor?? 
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

# start another training session

params = {
    'num_epochs' : 10,
    'batch_size' : min(64, len(train_set)),
    'learning_rate' : 1e-3,
    'weight_decay' : 1e-3,
    'image_shape' : np.array([3, 68, 68]),
    'latent_features' : 256,
    'model_type' : "SparseVAE",
    'alpha': 0.05, 
    'beta': 0.5, 
    'p_norm': 2.0
    }

vae, validation_data, training_data, params, vi = initVAEmodel(params)
cprint("params: {}".format(params), logfile)
vae = vae.to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])

train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=max(2, params['batch_size']), shuffle=False, num_workers=0, drop_last=False)
#train_batcher = TreatmentBalancedBatchGenerator(images, metadata_train)

######### VAE Training #########
cprint("VAE Training", logfile)

num_epochs = params['num_epochs']
batch_size = params['batch_size']

print_every = 1

for epoch in range(num_epochs):
    training_epoch_data = defaultdict(list)
    _ = vae.train()
    for x, _ in train_loader:
        # batchwise normalization. Only to be used if imagewise normalization has been ocmmented out.
        # x = batch_normalize_images(x)
        x = x.to(device)
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        optimizer.zero_grad()
        loss.backward()
        _ = nn.utils.clip_grad_norm_(vae.parameters(), 10_000)
        optimizer.step()
        for k, v in diagnostics.items():
            training_epoch_data[k] += list(v.cpu().data.numpy())

    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]

    with torch.no_grad():
        _ = vae.eval()
        
        validation_epoch_data = defaultdict(list)
        
        for x, _ in validation_loader:
            # batchwise normalization. Only to be used if imagewise normalization has been ocmmented out.
            # x = batch_normalize_images(x)
            x = x.to(device)
            
            loss, diagnostics, outputs = vi(vae, x)
            
            for k, v in diagnostics.items():
                validation_epoch_data[k] += list(v.cpu().data.numpy())
        
        for k, v in diagnostics.items():
            validation_data[k] += [np.mean(validation_epoch_data[k])]
            
        if epoch % print_every == 0:
            cprint(f"epoch: {epoch}/{num_epochs}", logfile)
            train_string = StatusString("training", training_epoch_data)
            evalString = StatusString("evaluation", validation_epoch_data)
            cprint(train_string, logfile)
            cprint(evalString, logfile)
            #cprint("vi.beta: {}".format(vi.beta), logfile)
            #cprint("vi.alpha: {}".format(vi.alpha), logfile)        

    vae.update_()
    vi.update_vi()


cprint("finished training", logfile)
print(training_data)

######### Save VAE parameters #########
cprint("Save VAE parameters", logfile)

datetime = get_datetime()
torch.save(vae.state_dict(), output_folder + "vae_parameters.pt")
torch.save(validation_data, output_folder + "validation_data.pt")
torch.save(training_data, output_folder + "training_data.pt")
torch.save(params, output_folder + "params.pt")

######### extract a few images already #########
cprint("Extract a few images already", logfile)
create_directory(output_folder + "images")

_ = vae.eval() # because of batch normalization
#plot_VAE_performance(training_data, file=None, title='VAE - learning')
plot_VAE_performance(training_data, file=output_folder + "images/training_data.png", title='VAE - learning')
plot_VAE_performance(validation_data, file=output_folder + "images/validation_data.png", title='VAE - validation')

n = 10
for i in range(n):
    x, y = train_set[i]
    plot_image_channels(x, file=output_folder + "images/x_{}.png".format(i))
    #plot_image_channels(x)
    x = x.to(device)
    outputs = vae(x[None,:,:,:])
    x_hat = outputs["x_hat"]
    x_reconstruction = x_hat
    x_reconstruction = x_reconstruction[0].detach()
    plot_image_channels(x_reconstruction.cpu(), file=output_folder + "images/x_reconstruction_{}.png".format(i))
    #plot_image_channels(x_reconstruction.cpu())

cprint("saved images", logfile)
cprint("output_folder is: {}".format(output_folder), logfile)
cprint("script done.", logfile)

