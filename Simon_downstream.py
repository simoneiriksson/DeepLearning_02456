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
from utils.plotting import heatmap

from utils.plotting import plot_cosine_similarity

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


#### LOAD TRAINED MODEL ####
# choose correct output folder for LoadVAEmodel() below!!!
model_type  = "nonvar"
output_folder = "./dump/outputs_2023-01-01 - 11-05-09/"
model, validation_data, training_data, params, vi = LoadVAEmodel(output_folder, model_type)

vae = model[0]

if params['model_type'] in ['SparseVAEGAN', 'CytoVAEGAN']:
    gan = model[1]

cprint("model is of type {}".format(params['model_type']), logfile)
cprint("model parameters are: {}".format(params), logfile)


########################################################
#                                                      #
#                 DOWNSTREAM TASKS                     #
#                                                      #
########################################################

_ = vae.eval() # because of batch normalization
#plot_VAE_performance(training_data, file=None, title='VAE - learning')
create_directory(output_folder + "images")
plot_VAE_performance(training_data, file=output_folder + "images/training_data.png", title='VAE - learning')
plot_VAE_performance(validation_data, file=output_folder + "images/validation_data.png", title='VAE - validation')

cprint("Extract a few images already", logfile)
extract_a_few_images(output_folder + "images", vae=vae, no_images=10, dataset=train_set, device=device)
cprint("saved images", logfile)

#### CALCULATE LATENT SPACE FOR ALL IMAGES ####
batch_size= 10000
metadata_latent = LatentVariableExtraction(metadata, images, batch_size, vae)


#### PLOT INTERPOLATION OF RECONSTRUCTONS ####
create_directory(output_folder + "interpolations")
#treatments list
tl = metadata['Treatment'].sort_values().unique()
#choosing the (target) treatment to plot
for target in [tl[0]]:
#for target in tl:
#target = tl[0]  #'ALLN_100.0'
    plot_cosine_similarity(target, metadata_latent, vae, output_folder + "interpolations/" + target + ".png")


#### PLOT LATENT SPACE HEATMAP ####
# heatmap of (abs) correlations between latent variables and MOA classes
heatmap = heatmap(metadata_latent)
# plot heatmap
plt.figure(figsize = (8,4))
heat = sns.heatmap(heatmap)
figure = heat.get_figure()
plt.gcf()
figure.savefig(output_folder + "images/latent_var_heatmap.png", bbox_inches = 'tight')


# #### NEAREST NEIGHBOR CLASSIFICATION (Not-Same-Compound) ####
# targets, predictions = NSC_NearestNeighbor_Classifier(metadata_latent, mapping, p=2)


# #### PLOT CONFUSION MATRIX ####
# confusion_matrix = moa_confusion_matrix(targets, predictions)
# df_cm = pd.DataFrame(confusion_matrix/np.sum(confusion_matrix) *100, index = [i for i in mapping],
#                          columns = [i for i in mapping])
# plt.figure(figsize = (12,7))
# cm = sns.heatmap(df_cm, annot=True)
# figure = cm.get_figure()
# plt.gcf()
# figure.savefig(downstream_folder + "conf_matrix.png", bbox_inches = 'tight')


# #### PRINT ACCURACY ####
# print("Model Accuracy:", Accuracy(confusion_matrix))



cprint("output_folder is: {}".format(output_folder), logfile)
cprint("script done.", logfile)

