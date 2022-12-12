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

from utils.data_preparation import *
from utils.data_transformers import *
from utils.plotting import *

from models.ReparameterizedDiagonalGaussian import *
from models.CytoVariationalAutoencoder import *
from models.VariationalAutoencoder import *
from models.ConvVariationalAutoencoder import *
from models.VariationalInference import *
from utils.utils import *
from models.LoadModels import *

######### Utilities #########

constant_seed()
datetime = get_datetime()
output_folder = "dump/outputs_{}/".format(datetime)
create_directory(output_folder)
logfile = create_logfile(output_folder + "log.log")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}", logfile)


######### loading data #########

#path = get_server_directory_path()
path = "data/all/"

metadata = read_metadata(path + "metadata.csv")
metadata = metadata[:2]
cprint("loaded metadata",logfile)

cprint("loading images", logfile)
relative_paths = get_relative_image_paths(metadata)
image_paths = [path + relative for relative in relative_paths]
images = load_images(image_paths, verbose=True, log_every=10000, logfile=logfile)
#images = torch.load("images.pt")
mapping = get_MOA_mappings(metadata)
cprint("loaded images", logfile)
normalize_every_image_channels_seperately_inplace(images)
cprint("normalized images", logfile)


metadata = shuffle_metadata(metadata)
metadata_train, metadata_validation = split_metadata(metadata, split_fraction = .90)

train_set = SingleCellDataset(metadata_train, images, mapping)
validation_set = SingleCellDataset(metadata_validation, images, mapping)


######### VAE Configs #########
cprint("VAE Configs", logfile)

# start another training session
vae, validation_data, training_data, VAE_settings = initVAEmodel(latent_features= 256,
                                                                    beta = 1.,
                                                                    num_epochs = 500,
                                                                    batch_size = min(64, len(train_set)),
                                                                    learning_rate = 1e-3,
                                                                    weight_decay = 10e-4,
                                                                    image_shape = np.array([3, 68, 68]),
                                                                    model_type = "Cyto"
                                                                    )
cprint("VAE_settings: {}".format(VAE_settings), logfile)
vae = vae.to(device)
optimizer = torch.optim.Adam(vae.parameters(), lr=VAE_settings['learning_rate'], weight_decay=VAE_settings['weight_decay'])

vi = VariationalInference(beta=VAE_settings['beta'])

train_loader = DataLoader(train_set, batch_size=VAE_settings['batch_size'], shuffle=True, num_workers=0, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=VAE_settings['batch_size'], shuffle=False, num_workers=0, drop_last=False)

######### VAE Training #########
cprint("VAE Training", logfile)

num_epochs = VAE_settings['num_epochs']
batch_size = VAE_settings['batch_size']

impatience_level = 0
max_patience = 100

best_elbo = np.finfo(np.float64).min
print_every = 100

for epoch in range(num_epochs):
    training_epoch_data = defaultdict(list)
    vae.train()
    
    for x, _ in train_loader:
        x = x.to(device)
        
        # perform a forward pass through the model and compute the ELBO
        loss, diagnostics, outputs = vi(vae, x)
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(), 1)
        optimizer.step()
        
        for k, v in diagnostics.items():
            training_epoch_data[k] += [v.mean().item()]
        
    for k, v in training_epoch_data.items():
        training_data[k] += [np.mean(training_epoch_data[k])]
    
    with torch.no_grad():
        vae.eval()
        
        validation_epoch_data = defaultdict(list)
        
        for x, _ in validation_loader:
            x = x.to(device)
            
            loss, diagnostics, outputs = vi(vae, x)
            
            for k, v in diagnostics.items():
                validation_epoch_data[k] += [v.mean().item()]
        
        for k, v in diagnostics.items():
            validation_data[k] += [np.mean(validation_epoch_data[k])]
        
        impatience_level += 1
        
        current_elbo = validation_data["elbo"][-1]
        if current_elbo > best_elbo:
            impatience_level = 0
            best_elbo = current_elbo
        
        #if impatience_level > max_patience:
        #    cprint("no more patience left at epoch {}".format(epoch), logfile)
        #    break
    
        if epoch % print_every == 0:
            cprint(f"epoch: {epoch}/{num_epochs}", logfile)
            train_string = StatusString("training", training_epoch_data)
            evalString = StatusString("evaluation", validation_data)
            cprint(train_string, logfile)
            cprint(evalString, logfile)


cprint("finished training", logfile)

######### Save VAE parameters #########
cprint("Save VAE parameters", logfile)

datetime = get_datetime()
torch.save(vae.state_dict(), output_folder + "vae_parameters.pt")
torch.save(validation_data, output_folder + "validation_data.pt")
torch.save(training_data, output_folder + "training_data.pt")
torch.save(VAE_settings, output_folder + "VAE_settings.pt")

######### extract a few images already #########
cprint("Extract a few images already", logfile)
create_directory("dump/images")

vae.eval() # because of batch normalization

plot_VAE_performance(training_data, file=output_folder + "images/training_data.png", title='VAE - learning')
plot_VAE_performance(validation_data, file=output_folder + "images/validation_data.png", title='VAE - validation')
#
n = 2
for i in range(n):
    x, y = train_set[i]
    plot_image_channels(x, file=output_folder + "images/x_{}.png".format(i))
    #plot_image_channels(x)
    x = x.to(device)
    outputs = vae(x[None,:,:,:])
    px = outputs["px"]
    
    
    x_reconstruction = px.sample()
    x_reconstruction = x_reconstruction[0]
    plot_image_channels(x_reconstruction.cpu(), file=output_folder + "images/x_reconstruction_{}.png".format(i))
    #plot_image_channels(x_reconstruction.cpu())
    

cprint("saved images", logfile)
cprint("script done.", logfile)