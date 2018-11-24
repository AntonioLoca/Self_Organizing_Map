from __future__ import print_function
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from neupy import algorithms, environment

import csv

if __name__ == '__main__':
    ggplot_colors = plt.rcParams['axes.prop_cycle']
    colors = np.array([c['color'] for c in ggplot_colors])

def center(mtx):
     media = mtx.mean(axis=0)
     Mtx_centr = mtx - media
     return Mtx_centr

# Funcao para normalizar os dados.

def normalize(MTX):
     normalizado = MTX.std(axis=0)
     normalizado[normalizado == 0] = 1
     MT_centr = center(MTX)
     MT_normal = MT_centr / normalizado
     return MT_normal



def csvread(filename, delimiter='\t'):
    f = open(filename, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    ncol = len(next(reader))
    nfeat = ncol - 1
    f.seek(0)
    x = np.zeros(nfeat)
    X = np.empty((0, nfeat))

    y = []
    for row in reader:
        for j in range(nfeat):
            x[j] = float(row[j])

        X = np.append(X, [x], axis=0)
        label = row[nfeat]
        y.append(label)

    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    classname = lb.classes_

    le = LabelEncoder()
    ynum = le.fit_transform(y)

    return X, ynum

def read_arq(A):
    filename = A
    delimiter = '\t'
    X1, ynum = csvread(filename=filename, delimiter=delimiter)
    X1 = normalize(X1)
    std = X1.std(axis=0)

    return X1, ynum

var = read_arq('all1.csv')

dataset = np.array(var[0])

plt.style.use('ggplot')
environment.reproducible()
target = var[-1]

data = np.zeros((len(var[1]),2))

for i in range(len(dataset)):
        data[i][:2] = dataset[i, [2,13]]

sofm = algorithms.SOFM(

        n_inputs=2,
        n_outputs=2,
        learning_radius=0,
        step=0.25,
        shuffle_data=True,
        weight='sample_from_data',
        verbose=True,
    )
sofm.train(data, epochs=200)

plt.title('Clustering Tennessee Eastman with SOFM')
plt.xlabel('Feature #')
plt.ylabel('Feature ##')

plt.scatter(*data.T, c=colors[target], s=50, alpha=1)
cluster_centers = plt.scatter(*sofm.weight, s=100, c=colors[3])

plt.legend([cluster_centers], ['Cluster center'], loc='upper left')
plt.show()
