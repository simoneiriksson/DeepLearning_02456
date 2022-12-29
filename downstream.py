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


datetime=get_datetime()
create_directory("dump/logs")
logfile = open("./dump/logs/log_{}.log".format(datetime), "w")
logfile=None

def constant_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
constant_seed()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cprint(f"Using device: {device}", logfile)

#### LOAD DATA ####
data_root = get_server_directory_path()                #use this for GitHub
#data_root  = "../data/mix_from_all/"           #use this for local PC
metadata_all = read_metadata(data_root + "metadata.csv")

metadata = shuffle_metadata(metadata_all)[:100000]
relative_path = get_relative_image_paths(metadata)
image_paths = [data_root  + path for path in relative_path]
#image_paths = [data_root  + path for path in relative_path]
images = load_images(image_paths, verbose=True, log_every=10000)
mapping = get_MOA_mappings(metadata)

#### NORMALISE DATA ####
normalize_channels_inplace(images)
print(images.shape)

#### SPLIT DATA ####
metadata_train_all, metadata_test = split_metadata(metadata, split_fraction = .90)
metadata_train, metadata_validation = split_metadata(metadata_train_all, split_fraction = .90)
train_set = SingleCellDataset(metadata_train, images, mapping)
validation_set = SingleCellDataset(metadata_validation, images, mapping)
test_set = SingleCellDataset(metadata_test, images, mapping)

#### LOAD TRAINED MODEL ####
model_type  = "nonvar"
#vae, validation_data, training_data, VAE_settings = LoadVAEmodel_old("./dump/outputs_2022-12-28 - 09-40-32/", model_type)
vae, validation_data, training_data, VAE_settings, vi = LoadVAEmodel("./dump/outputs_2022-12-28 - 09-40-32/", model_type)


#### PLOT MODEL PERFORMANCE####
plot_VAE_performance(validation_data, title='VAE - validation')
plot_VAE_performance(training_data, title='VAE - training')

#### PLOT INPUT IMAGES AND RECONSTUCTIONS ####
n = 1
for i in range(10,n+10):
    x, y = train_set[i] 
    vae.eval()
    # plot input image sample
    plot_image_channels(img_saturate(x), title="X's")
    # extract model outputs for the input image
    outputs = vae(x[None,:,:,:])
    # extract reconstructed image(s) from model output
    x_reconstruction = outputs["x_hat"].detach()
    x_reconstruction = x_reconstruction[0]
    # plot reconstructed image
    plot_image_channels(img_saturate(x_reconstruction), title="Reconstructions")

#### PLOT INTERPOLATION OF RECONSTRUCTONS? (Cosine Similarity?)####


#### PLOT LATENT SPACE HEATMAP ####
# heatmap of (abs) correlations between latent variables and MOA classes
batch_size= 10000
metadata_latent = LatentVariableExtraction(metadata, images, batch_size, vae)
heatmap(metadata_latent)


#### NEAREST NEIGHBOR CLASSIFICATION (Not-Same-Compound) ####
targets, predictions = NSC_NearestNeighbor_Classifier(metadata_latent, mapping, p=2)


#### PLOT CONFUSION MATRIX ####
confusion_matrix = moa_confusion_matrix(targets, predictions)
plot_confusion_matrix(confusion_matrix, mapping)


#### PRINT ACCURACY ####
print("Model Accuracy:", Accuracy(confusion_matrix))



























