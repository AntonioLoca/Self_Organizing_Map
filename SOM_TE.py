from __future__ import print_function
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tqdm import tqdm_notebook as tqdm
import numpy as np
import matplotlib.animation as animation
from IPython.display import HTML
from neupy import algorithms, environment
import csv
import matplotlib.pyplot as plt
environment.reproducible()

def center(mtx):
    media = mtx.mean(axis=0)
    Mtx_centr = mtx - media
    return Mtx_centr


# Funcao para normalizar os dados.

def normalize(MTX):
    normalizado = MTX.std (axis=0)
    normalizado[normalizado == 0] = 1
    MT_centr = center (MTX)
    MT_normal = MT_centr / normalizado
    return MT_normal


def csvread(filename, delimiter='\t'):
    f = open (filename, 'r')
    reader = csv.reader (f, delimiter=delimiter)
    ncol = len (next (reader))
    nfeat = ncol - 1
    f.seek (0)
    x = np.zeros (nfeat)
    X = np.empty ((0, nfeat))

    y = []
    for row in reader:
        for j in range (nfeat):
            x[j] = float (row[j])

        X = np.append (X, [x], axis=0)
        label = row[nfeat]
        y.append (label)

    lb = LabelBinarizer ()
    Y = lb.fit_transform (y)
    classname = lb.classes_

    le = LabelEncoder ()
    ynum = le.fit_transform (y)

    return X, ynum


def read_arq(A):
    filename = A
    delimiter = '\t'
    X1, ynum = csvread (filename=filename, delimiter=delimiter)
    X1 = normalize (X1)
    std = X1.std (axis=0)

    return X1, ynum


var = read_arq('all1.csv')

dataset = np.array(var[0])

plt.style.use('ggplot')
environment.reproducible()
target = var[-1]

data = np.zeros((len(var[1]), 2))

for i in range(len(dataset)):
    data[i][:2] = dataset[i, [2, 13]]

gng = algorithms.GrowingNeuralGas(
    n_inputs=2,
    n_start_nodes=2,

    shuffle_data=True,
    verbose=True,

    step=0.1,
    neighbour_step=0.001,

    max_edge_age=50,
    max_nodes=100,

    n_iter_before_neuron_added=100,
    after_split_error_decay_rate=0.5,
    error_decay_rate=0.995,
    min_distance_for_update=0.2,
)

fig = plt.figure()
plt.scatter(*data.T, alpha=0.02)
plt.xticks([], [])
plt.yticks([], [])


def animate(i):
    for line in animate.prev_lines:
        line.remove()

    # Training will slow down overtime and we increase number
    # of data samples for training
    n = int(0.5 * gng.n_iter_before_neuron_added * (1 + i // 100))

    sampled_data_ids = np.random.choice(len(data), n)
    sampled_data = data[sampled_data_ids, :]
    gng.train(sampled_data, summary='inline', epochs=1)

    lines = []
    for node_1, node_2 in gng.graph.edges:
        weights = np.concatenate([node_1.weight, node_2.weight])
        line, = plt.plot(*weights.T, color='black')

        plt.setp(line, linewidth=1, color='red')

        lines.append(line)
        lines.append(plt.scatter(*weights.T, color='blue', s=10))

    animate.prev_lines = lines
    return lines


animate.prev_lines = []
anim = animation.FuncAnimation(fig, animate, tqdm(np.arange(20)), interval=30, blit=True)
HTML(anim.to_html5_video())





