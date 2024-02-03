#!/usr/bin/env python
# coding: utf-8

# # Load packages and dataset
# this script computes the (linear) PCA for the Iris data set
# author: Dr. Daniel Tenbrinck, Dr. Leon Bungert, modified by Florian Roesel
# date: 28.04.2021, 03.03.22

import numpy as np
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# choose dimension of reduced data -> visualization only implemented for k=2 and k=3
k = 3

# import data from Iris data set using pandas
columns = ['Sepal length in cm', 'Sepal width in cm', 'Petal length in cm', 'Petal width in cm', 'Class']
dataframe = pd.read_csv('irisdata.sec', names=columns)

# change the Class from string to int
dataframe['Class'] = dataframe['Class'].astype('category')
dataframe['Class'] = dataframe['Class'].cat.codes


print("Initial data loaded...")
print(dataframe)

# extract all numeric data columns by excluding the "Class" column
data_numeric = dataframe.loc[:, ~dataframe.columns.str.contains('Class')]

# convert dataframe to numeric data
data_numeric = data_numeric.to_numpy()

# determine size of data -> N is number of data points, M is data dimension
N, M = data_numeric.shape


# # Normalize data
# compute mean value of data
data_mean = np.true_divide(sum(data_numeric),N)

print("Mean...")
print(data_mean)

# center data using mean value
data_centered = data_numeric - data_mean

print("Centered data...")
print(data_centered)

# # Compute PCA
# quick version
covariance_matrix = np.matmul(data_centered.T, data_centered) / N
print(data_centered.shape)

print("Cov Matrix...")
print(covariance_matrix)

# numerically compute eigenvalues and respective eigenvectors
eigenvalues, eigenvectors = la.eig(covariance_matrix)

print("Eigenvalues, Eigenvectors...")
print(eigenvalues)
print(eigenvectors)

# extract only eigenvectors of k biggest eigenvalues
transformation = eigenvectors[:,0:k]

# compute principal components
principal_components = np.matmul(data_centered, transformation)

print("Transformed data points...")
print(principal_components)

print("Close window to exit...")
# plot reduced data
if k == 1: #1D plotting
    # create figure to plot
    fig = plt.figure(1, figsize=(8, 6))
    
    # generate plot from principal components
    scatter = plt.scatter(principal_components, 0*principal_components, c=dataframe['Class'])
    
    # add a legend
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles, ("Iris Setosa", "Iris Versicolor", "Iris Virginica"))
    plt.show()
    
if k == 2: # 2D plotting
    # create figure to plot
    fig = plt.figure(1, figsize=(8, 6))
    
    # generate scatter data from principal components
    scatter = plt.scatter(principal_components[:, 0], principal_components[:, 1], c=dataframe['Class'], edgecolor='k')
    
    # add a legend
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
    plt.legend(handles, ("Iris Setosa", "Iris Versicolor", "Iris Virginica"))
    plt.show()
    
if k == 3: # 3D plotting
    # create figure to plot
    fig = plt.figure(1, figsize=(8, 6))
    
    # create 3D view with certain angle
    ax = Axes3D(fig, elev=-130, azim=130)
    
    # generate scatter data from principal components
    scatter = ax.scatter(principal_components[:, 0], principal_components[:, 1], principal_components[:, 2], c=dataframe['Class'], edgecolor='k', s=50)
        
    # add a legend in lower left corner
    handles, labels = scatter.legend_elements(prop="colors", alpha=0.7)
    legend = ax.legend(handles, ("Iris Setosa", "Iris Versicolor", "Iris Virginica"), loc="lower left", title="Classes")    
    ax.add_artist(legend)
    plt.show()



