from typing import List, Set, Dict, Tuple, Optional, Any
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn

from utils.data_transformers import view_as_image_plot_format, clip_image_to_zero_one
from utils.profiling import treatment_profiles


def img_saturate(img):
    return img / img.max()

def plot_image(image: torch.Tensor, clip: bool = True, file=None, title=None):
    image = image.clone()

    if clip:
        image = clip_image_to_zero_one(image)

    plot_image = view_as_image_plot_format(image)
    plt.imshow(plot_image)

    if file==None:
        plt.show()
    else: 
        plt.savefig(file)

    plt.close()


def plot_image_channels(image: torch.Tensor, clip: bool = True, colorized: bool = True, file=None, title=None):
    image = image.clone()

    if clip:
        image = clip_image_to_zero_one(image)

    fig, axs = plt.subplots(1, 4, figsize=(14,6))
    if not title == None: fig.suptitle(title, fontsize=14)

    channel_names = ["DNA", "F-actin", "B-tubulin"]
    for i, name in enumerate(channel_names):
        if colorized:
            channel_image = torch.zeros_like(image)
            channel_image[i] = image[i]
        else:
            channel_image = image[i]
            channel_image = channel_image[None,:,:].expand(3, -1, -1)

        ax = axs[i]
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)
        ax.imshow(view_as_image_plot_format(channel_image))

    ax = axs[3]
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Combined")
    ax.imshow(view_as_image_plot_format(image))

    if file==None:
        plt.show()
    else: 
        plt.savefig(file)

    plt.close()

def plot_VAE_performance(plotdata, file=None, title=None):
    keys = plotdata.keys()
    fig, axs = plt.subplots(1, len(keys), figsize=(14,6), constrained_layout = True)
    fig.suptitle(title, fontsize=26)
    for no, key in enumerate(keys):
        ax = axs[no]
        ax.grid()
        ax.plot(plotdata[key])
        ax.set_ylabel(key, fontsize=26)
        ax.set_xlabel("epoch", fontsize=26)
        
    if file == None:
        plt.show()
    else: 
        plt.savefig(file)
    
    plt.close()

def plot_cosine_similarity(x0, x1, model, file=None, title=None):
    #model could be eg. "model_dump/outputs_2022-12-04 - 12-20-15/"
    #x0.shape and x1.shape should be torch.Size([3, 68, 68])
    
    #vae, validation_data, training_data, VAE_settings = LoadVAEmodel(model)
    vae = model
    outputs0 = vae(x0[None,:,:,:])
    outputs1 = vae(x1[None,:,:,:])

    z0 = outputs0["z"].detach().numpy()
    z1 = outputs1["z"].detach().numpy()

    zs = np.linspace(z0, z1, num=10)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    cp = []
    for i in range(len(zs)):
        input1 = vae.observation(torch.Tensor(zs[i]))[0]
        benchmark = vae.observation(torch.Tensor(zs[9]))[0]
        cp.append(cos(input1, benchmark).mean().detach().numpy().round(2))

    # create figure
    fig = plt.figure(figsize=(25, 10))

    # setting values to rows and column variables
    rows = 1
    columns = len(zs)

    # Adds a subplot at the 1st position
    for i in range(columns):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow((torch.permute(vae.observation(torch.Tensor(zs[i]))[0], (1, 2, 0))* 255).detach().numpy().astype(np.uint8))
        plt.axis('off')
        if i == 0:
            plt.title('Control', y=-0.20, fontsize=26)
        elif i == len(zs)-1:
            plt.title('Target', y=-0.20, fontsize=26)
        else:
            plt.title(cp[i], fontsize=26)
    if file == None:
        plt.show()
    else: 
        plt.savefig(file)
    
    plt.close()
    
    
def heatmap(metadata_latent):
    metadata_onehot = pd.concat([metadata_latent, pd.get_dummies(metadata_latent["moa"], prefix = 'onehot_moa')], axis=1)
    latent_cols = [col for col in metadata_onehot.columns if type(col)==str and col[0:7]=='latent_']
    one_hot_cols = [col for col in metadata_onehot.columns if type(col)==str and col[0:7]=='onehot_']
    heatmap_matrix = metadata_onehot[one_hot_cols + latent_cols].corr().filter(items=one_hot_cols, axis=0)[latent_cols]
    return plt.matshow(heatmap_matrix.abs())
   

def NSC_NearestNeighbor_Classifier(metadata_latent, mapping, p=2):
    treatment_profiles_df = treatment_profiles(metadata_latent)
    latent_cols = [col for col in metadata_latent.columns if type(col)==str and col[0:7]=='latent_']
    
    for compound in metadata_latent['Image_Metadata_Compound'].unique():
        A_set = treatment_profiles_df[treatment_profiles_df['Image_Metadata_Compound'] == compound]
        B_set = treatment_profiles_df[treatment_profiles_df['Image_Metadata_Compound'] != compound]
        for A in A_set.index:
            A_treatment = A_set.loc[A]['Treatment']
            diffs = (abs(B_set[latent_cols] - A_set.loc[A][latent_cols]))**p
            diffs_sum = diffs.sum(axis=1)**(1/p)
            diffs_min = diffs_sum.min()
            treatment_profiles_df.loc[treatment_profiles_df['Treatment']==A_treatment,'moa_pred'] = B_set.at[diffs[diffs_sum == diffs_min].index[0], 'moa']
    return treatment_profiles_df['moa'], treatment_profiles_df['moa_pred']

    
def moa_confusion_matrix(targets, predictions, mapping):
    nb_classes = len(targets.unique())
    moa_classes = targets.sort_values().unique()
    classes = np.zeros((nb_classes, nb_classes))
    for i in range(nb_classes):
        for j in range(nb_classes):
            for t in range(len(targets)):
                if targets[t] == moa_classes[i] and predictions[t] == moa_classes[j]:
                    classes[i,j] += 1
    
    cf_matrix = classes  
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *100, index = [i for i in mapping],
                         columns = [i for i in mapping])
    return cf_matrix, df_cm


def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize = (12,7))
    sns.heatmap(confusion_matrix, annot=True)
    
    
def Accuracy(confusion_matrix):
    class_accuracy = 100*confusion_matrix.diagonal()/confusion_matrix.sum(1)
    class_accuracy = class_accuracy.mean()
    return class_accuracy
   
    




