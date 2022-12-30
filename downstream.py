%matplotlib inline
import matplotlib.pyplot as plt

from typing import List, Set, Dict, Tuple, Optional, Any
from collections import defaultdict

import pandas as pd
import seaborn as sns
import numpy as np

import math 
import torch
from torch import nn, Tensor
from torch.nn.functional import softplus, relu
from torch.distributions import Distribution, Normal
from torch.utils.data import DataLoader, Dataset

from utils.data_preparation import *
from utils.data_transformers import *
from utils.plotting import *
from utils.profiling import *
from utils.utils import *

from models.ReparameterizedDiagonalGaussian import *
from models.CytoVariationalAutoencoder import CytoVariationalAutoencoder
from models.VariationalAutoencoder import VariationalAutoencoder
from models.ConvVariationalAutoencoder import ConvVariationalAutoencoder
from models.VariationalInference import VariationalInference
from models.VariationalInference_nonvar import VariationalInference_nonvar
from models.LoadModels import *


constant_seed()
datetime = get_datetime()
downstream_folder = "dump/downstream_{}/".format(datetime)
create_directory(downstream_folder)
logfile = create_logfile(downstream_folder + "log.log")
cprint("downstream_folder is: {}".format(downstream_folder), logfile)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}", logfile)

#### LOAD DATA ####
data_root = get_server_directory_path()   #use this for GPU/HPC
metadata_all = read_metadata(data_root + "metadata.csv")

####### use the 3 lines below for local PC / ThinLinc ########

####### use line below for GPU/HPC ########
images = torch.load("../data/images.pt")        #use this for GPU/HPC

mapping = get_MOA_mappings(metadata)

#### NORMALISE DATA ####
normalize_every_image_channels_seperately_inplace(images, verbose=True)

#### SPLIT DATA ####
metadata_train_all, metadata_test = split_metadata(metadata, split_fraction = .90)
metadata_train, metadata_validation = split_metadata(metadata_train_all, split_fraction = .90)
train_set = SingleCellDataset(metadata_train, images, mapping)
validation_set = SingleCellDataset(metadata_validation, images, mapping)
test_set = SingleCellDataset(metadata_test, images, mapping)

#### LOAD TRAINED MODEL ####
# choose correct output folder for LoadVAEmodel() below!!!
model_type  = "nonvar"
#vae, validation_data, training_data, VAE_settings, vi = LoadVAEmodel("./dump/outputs_2022-12-28 - 09-40-32/", model_type)
#vae, validation_data, training_data, VAE_settings, vi = LoadVAEmodel("./dump/outputs_2022-12-29 - 19-45-02/", model_type)
model, validation_data, training_data, params, vi = LoadVAEmodel("./dump/outputs_2022-12-30 - 17-07-33/")

cprint("model is of type {}".format(params['model_type']), logfile)
cprint("model parameters are: {}".format(params), logfile)

if type(model)==list:
    vae=model[0]
    gan=model[1]
    print("meh")
else: vae=model

#### PLOT MODEL PERFORMANCE####
plot_VAE_performance(training_data, file=downstream_folder + "training_data.png", title='VAE - learning')
plot_VAE_performance(validation_data, file=downstream_folder + "validation_data.png", title='VAE - validation')

#### PLOT INPUT IMAGES AND RECONSTUCTIONS ####
# choose number of images/reconstructions to plot: "n" below
n = 3
for i in range(10,n+10):
    x, y = train_set[i] 
    _=vae.eval()
    # plot input image sample
    plot_image_channels(img_saturate(x), title="X's", file=downstream_folder + "x_{}.png".format(i))
    # extract model outputs for the input image
    outputs = vae(x[None,:,:,:])
    # extract reconstructed image(s) from model output
    x_reconstruction = outputs["x_hat"].detach()
    x_reconstruction = x_reconstruction[0]
    # plot reconstructed image
    plot_image_channels(img_saturate(x_reconstruction.cpu()), file=downstream_folder + "x_reconstruction_{}.png".format(i))
    

#### PLOT INTERPOLATION OF RECONSTRUCTONS? (Cosine Similarity?)####
#treatments list
tl = metadata['Treatment'].sort_values().unique()
#choosing the (target) treatment to plot
target = tl[0]  #'ALLN_100.0'
model = "../dump/outputs_2022-12-28 - 09-40-32/"
model_type = "nonvar"
filefolder = "../" + downstream_folder + "latent_interpolation.png"
plot_cosine_similarity(target, metadata_latent, 
                        model, 
                        model_type,
                        filefolder)


#### PLOT LATENT SPACE HEATMAP ####
# heatmap of (abs) correlations between latent variables and MOA classes
batch_size= 10000
metadata_latent = LatentVariableExtraction(metadata, images, batch_size, vae)
heatmap = heatmap(metadata_latent)
# plot heatmap
plt.figure(figsize = (8,4))
heat = sns.heatmap(heatmap)
figure = heat.get_figure()
plt.gcf()
figure.savefig(downstream_folder + "latent_var_heatmap.png", bbox_inches = 'tight')


#### NEAREST NEIGHBOR CLASSIFICATION (Not-Same-Compound) ####
targets, predictions = NSC_NearestNeighbor_Classifier(metadata_latent, mapping, p=2)


#### PLOT CONFUSION MATRIX ####
confusion_matrix = moa_confusion_matrix(targets, predictions)
df_cm = pd.DataFrame(confusion_matrix/np.sum(confusion_matrix) *100, index = [i for i in mapping],
                         columns = [i for i in mapping])
plt.figure(figsize = (12,7))
cm = sns.heatmap(df_cm, annot=True)
figure = cm.get_figure()
plt.gcf()
figure.savefig(downstream_folder + "conf_matrix.png", bbox_inches = 'tight')


#### PRINT ACCURACY ####
print("Model Accuracy:", Accuracy(confusion_matrix))