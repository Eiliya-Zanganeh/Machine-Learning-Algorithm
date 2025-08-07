from sklearn.metrics import pairwise_distances_argmin
import numpy as np


def find_clusters(x, clusters, seed=2):
    rng = np.random.RandomState(seed)
    i = rng.permutation(x.shape[0])[:clusters]
    centers = x[i]

    while True:
        labels = pairwise_distances_argmin(x, centers)

        new_centers = np.array([x[labels == i].mean(0) for i in range(clusters)])

        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels
