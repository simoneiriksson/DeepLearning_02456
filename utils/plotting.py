from typing import List, Set, Dict, Tuple, Optional, Any
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from utils.data_transformers import view_as_image_plot_format, clip_image_to_zero_one

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