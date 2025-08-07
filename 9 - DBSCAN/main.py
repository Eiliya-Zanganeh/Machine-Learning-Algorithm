import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from Machine_Learning_Module.Classifier import Classifier
from Machine_Learning_Module.utils import find_clusters


def train(save_path):
    data = pd.read_csv('dataset/Mall_Customers.csv')
    # print(data.head(10))

    data = data.iloc[:, [3, 4]].values
    # print(data.head(10))

    # plt.scatter(data[:, 0], data[:, 1], s=10, c='blue')
    # plt.show()

    model = Classifier(
        model=DBSCAN(eps=1, min_samples=5),
        model_type='classification',
        x=data,
        test_size=.2
    )

    model.train()
    model.save_model(save_path)

    y_pred = model.model.fit_predict(data)

    centers, labels = find_clusters(data, 4)
    plt.scatter(data[:, 0], data[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    # Train Model
    train('DBSCAN.pkl')
