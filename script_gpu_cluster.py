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
from utils.plotting import plot_VAE_performance, plot_image_channels, extract_a_few_images
from utils.data_preparation import create_directory, read_metadata, get_relative_image_paths, load_images
from utils.data_preparation import read_metadata_and_images
from utils.data_preparation import get_MOA_mappings, shuffle_metadata, split_metadata
from utils.utils import cprint, get_datetime, create_logfile, constant_seed, StatusString
from utils.utils import save_model
from utils.profiling import LatentVariableExtraction
from utils.plotting import heatmap, plot_cosine_similarity
from utils.plotting import NSC_NearestNeighbor_Classifier, moa_confusion_matrix, Accuracy


import importlib

from VAE_trainer import VAE_trainer 
from VAEGAN_trainer import VAEGAN_trainer

######### Utilities #########

constant_seed()
datetime = get_datetime()
output_folder = "dump/outputs_{}/".format(datetime)
create_directory(output_folder)
logfile = create_logfile(output_folder + "log.log")
cprint("output_folder is: {}".format(output_folder), logfile)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}", logfile)

images, metadata, mapping = read_metadata_and_images(use_server_path = False, \
                                                        load_images_from_individual_files = True, 
                                                        load_subset_of_images = 1000, 
                                                        save_images_to_singlefile = False,
                                                        logfile = logfile)
# Settings for handing in:
#images, metadata, mapping = read_metadata_and_images(use_server_path = True, \
#                                                        load_images_from_individual_files = True, 
#                                                        load_subset_of_images = None, 
#                                                        save_images_to_singlefile = False)

# With the below command, we normalize all the images, image- and channel-wise.
# Alternative, this can be uncommented and like in the Lafarge article, we can do batchwise normalization
normalize_every_image_channels_seperately_inplace(images, verbose=True)

metadata = shuffle_metadata(metadata)
metadata_train, metadata_validation = split_metadata(metadata, split_fraction = .90)

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

models, validation_data, training_data, params, vi = initVAEmodel(params)
cprint("params: {}".format(params), logfile)

train_loader = DataLoader(train_set, batch_size=params['batch_size'], shuffle=True, num_workers=0, drop_last=True)
validation_loader = DataLoader(validation_set, batch_size=max(2, params['batch_size']), shuffle=False, num_workers=0, drop_last=False)
vae = models[0]

if params['model_type'] in ['SparseVAEGAN', 'CytoVAEGAN']:
    Trainer = VAEGAN_trainer
    gan = models[1]

if params['model_type'] in  ['Cyto_nonvar', 'CytoVAE', 'SparseVAE']:
    Trainer = VAE_trainer

Trainer(models=models, \
    validation_data=validation_data, \
    training_data=training_data, \
    params=params, 
    vi=vi, 
    train_loader=train_loader, 
    device=device, 
    validation_loader=validation_loader, 
    print_every=1, 
    logfile=logfile)

cprint("finished training", logfile)
print(training_data)

######### Save VAE parameters #########
cprint("Save VAE parameters", logfile)
save_model(models, validation_data, training_data, params, output_folder)


########################################################
#                                                      #
#                 DOWNSTREAM TASKS                     #
#                                                      #
########################################################
cprint("Starting downstream tasks", logfile)

_ = vae.eval() # because of batch normalization
#plot_VAE_performance(training_data, file=None, title='VAE - learning')
cprint("Plotting VAE performance", logfile)
create_directory(output_folder + "images")
plot_VAE_performance(training_data, file=output_folder + "images/training_data.png", title='VAE - learning')
plot_VAE_performance(validation_data, file=output_folder + "images/validation_data.png", title='VAE - validation')

cprint("Extract a few images", logfile)
extract_a_few_images(output_folder + "images", vae=vae, no_images=10, dataset=train_set, device=device)
cprint("saved images", logfile)

#### CALCULATE LATENT REPRESENTATION FOR ALL IMAGES ####
cprint("Calculate latent representation for all images", logfile)
batch_size= 10000
metadata_latent = LatentVariableExtraction(metadata, images, batch_size, vae)
cprint("Done calculating latent sapce", logfile)

#### PLOT INTERPOLATION OF RECONSTRUCTONS ####
cprint("Plotting interpolations of reconstructions", logfile)
create_directory(output_folder + "interpolations")
#treatments list
tl = metadata['Treatment'].sort_values().unique()
#choosing the (target) treatment to plot
#for target in [tl[0]]:
for target in tl:
#target = tl[0]  #'ALLN_100.0'
    plot_cosine_similarity(target, metadata_latent, vae, output_folder + "interpolations/" + target + ".png")

#### PLOT LATENT SPACE HEATMAP ####
cprint("Plotting latent space heatmap", logfile)
# heatmap of (abs) correlations between latent variables and MOA classes
heatmap = heatmap(metadata_latent)
# plot heatmap
plt.figure(figsize = (8,4))
heat = sns.heatmap(heatmap)
figure = heat.get_figure()
plt.gcf()
figure.savefig(output_folder + "images/latent_var_heatmap.png", bbox_inches = 'tight')


#### NEAREST NEIGHBOR CLASSIFICATION (Not-Same-Compound) ####
cprint("Nearest neighbor classification (Not-Same-Compound)", logfile)
targets, predictions = NSC_NearestNeighbor_Classifier(metadata_latent, mapping, p=2)

#### PLOT CONFUSION MATRIX ####
confusion_matrix = moa_confusion_matrix(targets, predictions)
df_cm = pd.DataFrame(confusion_matrix/np.sum(confusion_matrix) *100, index = [i for i in mapping],
                         columns = [i for i in mapping])
plt.figure(figsize = (12,7))
cm = sns.heatmap(df_cm, annot=True)
figure = cm.get_figure()
plt.gcf()
figure.savefig(output_folder + "conf_matrix.png", bbox_inches = 'tight')


#### PRINT ACCURACY ####
cprint("Model Accuracy: {}".format(Accuracy(confusion_matrix)), logfile)



cprint("output_folder is: {}".format(output_folder), logfile)
cprint("script done.", logfile)

