from sklearn import svm
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np

from Machine_Learning_Module.Classifier import Classifier


def train(save_path):
    x, y = make_blobs(n_samples=200, centers=2, random_state=1, n_features=2)

    model = Classifier(
        model=svm.SVC(kernel='linear', C=1000),
        model_type='classification',
        x=x,
        y=y,
        test_size=.2
    )

    model.train()
    model.test()
    model.save_model(save_path)

    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    plt.show()

    plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    ax.scatter(model.model.support_vectors_[:, 0], model.model.support_vectors_[:, 1], s=100, linewidths=1,
               facecolors='none')
    plt.show()


if __name__ == '__main__':
    # Train Model
    train('svm.pkl')
