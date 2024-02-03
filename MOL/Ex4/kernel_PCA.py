#!/usr/bin/env python
# coding: utf-8

# # Load packages and dataset

# this script computes the (linear) PCA for the Iris data set
# author: Dr. Daniel Tenbrinck, Dr. Leon Bungert; modified by Florian Roesel
# date: 28.04.2021, 03.03.22

import numpy as np
import sys
from numpy import linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def poly_kernel(a, d, x1, x2):
    return (x1 @ x2 + a)**d


def gaussian_kernel(sigma, x1, x2):
    return np.exp(-np.linalg.norm(x1-x2)**2/2*sigma**2)


def read_circle_data(filename):
    # import data from Iris data set using pandas
    columns = ['X1', 'X2', 'Class']
    dataframe = pd.read_csv(filename, names=columns)

    # extract all numeric data columns by excluding the "Class" column
    data_numeric = dataframe.loc[:, ~dataframe.columns.str.contains('Class')]
    classes = dataframe.loc[:, dataframe.columns.str.contains("Class")]
    
    # convert dataframe to numeric data
    data_numeric = data_numeric.to_numpy()
    classes = classes.to_numpy()

    # determine size of data -> N is number of data points, M is data dimension
    return data_numeric, classes


def kernel_PCA(data_numeric, kernel, k, N):
    # data numeric: some input data
    # kernel: a function handle; which kernel to use
    # k: how many components do you want
    # N: how much data to use to calculate the components
    
    # return value: a function handle, mapping points to PC
    M = data_numeric.shape[1]
    K = np.ndarray([N, N])
    oneK = np.ndarray([N, N])
    # fill matrices
    for i in range(N):
        for j in range(N):
            K[i, j] = kernel(data_numeric[i], data_numeric[j])
            oneK[i, j] = 1./N
    
    Ktilde = (K - oneK @ K - K @ oneK + oneK @ K @ oneK)
    
    # eigenvalues, eigenvectors
    w, v = np.linalg.eig(Ktilde)
    # remove imaginary part (is valid since Ktilde is symmetric)
    w=np.real(w)
    v=np.real(v)
    # sorting
    v = np.flip(v[:,w.argsort()], axis=1)
    w = np.flip(w[w.argsort()], axis=0)
    # test if eigenvalue equation is still ok (sorting is prone to human mistakes)
    # print(Ktilde @ v[:,0] - w[0] * v[:,0])
    
    # select sub arrays
    w = w[range(k)]
    v = v[:, range(k)]
    
    # normalize
    for i in range(k):
        scale = np.sqrt(1/(w[i]*N*v[:, i] @ v[:, i]))
        v[:, i] = v[:, i] * scale
    
    # prepare function handle which transforms original data in kernel PC
    def transform_function(data_point):
        return np.array([sum(v[i, j] * kernel(data_numeric[i], data_point) for i in range(N)) for j in range(k)])
    
    # return
    return transform_function


# execute the transformations
k = 2
data_numeric, classes = read_circle_data("Circledata.sec")


# color points in original classes
color_dict = {0: "red", 1: "blue"}
color_list = [color_dict[int(label)] for label in classes]


# gaussian kernel seems to work    
transformer = kernel_PCA(data_numeric, lambda x1, x2: gaussian_kernel(1., x1, x2), k, 10)
transformed_data = np.array([transformer(point) for point in data_numeric])
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color=color_list)
plt.show()


# polynomial kernel seems to work not that well...
for a in range(-10, 10):
    transformer = kernel_PCA(data_numeric, lambda x1, x2: poly_kernel(a/10, 2., x1, x2), k, 10)
    transformed_data = np.array([transformer(point) for point in data_numeric])
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color=color_list)
    plt.show()



