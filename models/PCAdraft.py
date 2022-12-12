# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 12:40:14 2022

@author: JUCA
"""

from utils.data_preparation import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

df_small = pd.read_csv('C:\git\generative-models-for-phenotypic-profiling\data\small\metadata.csv', sep=',')

image_paths_list = get_relative_image_paths(df_small)

image__abs_paths_list = []

for i in image_paths_list:
    abs_path = 'C:/git/generative-models-for-phenotypic-profiling/data/small/' + i
    image__abs_paths_list.append(abs_path)
    
images_arr_list = load_images(image__abs_paths_list)

images_flattened_list = []

for i in images_arr_list:
    arr = i
    vector_flattened = arr.flatten()
    images_flattened_list.append(vector_flattened)
    
X = np.array(images_flattened_list)

def PCA_julia(mat, n_components):
    
    sc = StandardScaler()
    X_transformed = sc.fit_transform(X)
    pca = PCA(n_components)
    result = pca.fit(X_transformed)
    
    return result

PCA_julia(mat=X, n_components=1000)

sc = StandardScaler()
X_n = sc.fit_transform(X)

pca = PCA(n_components=1000)
pca.fit(X_n)

#take one timage and apply the pca
img = X_n[0]

#https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html

print(pca.explained_variance_ratio_)

pca.explained_variance_ratio_.sum()

np.cumsum(pca.explained_variance_ratio_)[:150]

#kwargs: dimens or exaplined

def testPCAFit(matrix,n,three_D=True,scatter=True):
    pca = PCA(n_components=n)
    pca.fit(matrix)

    reducedMatrixPCA = pca.transform(matrix)

    reconMatrixPCA = pca.inverse_transform(reducedMatrixPCA)
    reconCostPCA = np.mean(np.power(reconMatrixPCA - matrix,2),axis = 1)
    reconCostPCA = reconCostPCA.reshape(-1, 1)
    print('Reconstruction MSE : ',np.mean(reconCostPCA))
    
    if three_D:
        if scatter:
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            #ax.plot_wireframe(matrix[:,0],matrix[:,1],matrix[:,2])
            ax.scatter3D(reconMatrixPCA[:,0],reconMatrixPCA[:,1],reconMatrixPCA[:,2])
        else:
            X = reconMatrixPCA[:,0].reshape(samples,samples)
            Y = reconMatrixPCA[:,1].reshape(samples,samples)
            Z = reconMatrixPCA[:,2].reshape(samples,samples)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.plot_wireframe(X,Y,Z)
    else:
        plt.plot(reconMatrixPCA[:,0],reconMatrixPCA[:,1])
    
    return np.mean(reconCostPCA)

res = testPCAFit(X,1000)
res

