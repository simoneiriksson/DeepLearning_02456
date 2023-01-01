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

from utils.plotting import plot_VAE_performance, plot_image_channels, extract_a_few_images
from utils.data_preparation import create_directory, read_metadata, get_relative_image_paths, load_images
from utils.utils import cprint
from utils.profiling import LatentVariableExtraction
from utils.plotting import heatmap, plot_cosine_similarity
from utils.plotting import NSC_NearestNeighbor_Classifier, moa_confusion_matrix, Accuracy

def downstream_task(vae, metadata, train_set, images, mapping, device, output_folder, logfile=None):
    cprint("Starting downstream tasks", logfile)

    _ = vae.eval() # because of batch normalization

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
