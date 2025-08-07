from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
import matplotlib.pyplot as plt
import numpy as np

from Machine_Learning_Module.Classifier import Classifier


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


def train(save_path):
    x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0, n_features=2)

    # plt.scatter(x[:, 0], x[:, 1], s=10)
    # plt.show()

    model = Classifier(
        model=KMeans(n_clusters=4),
        model_type='classification',
        x=x,
        test_size=.2
    )

    model.train()
    model.save_model(save_path)

    y_pred = model.model.predict(x)

    centers, labels = find_clusters(x, 4)
    plt.scatter(x[:, 0], x[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    # Train Model
    train('K Means Clustering.pkl')
