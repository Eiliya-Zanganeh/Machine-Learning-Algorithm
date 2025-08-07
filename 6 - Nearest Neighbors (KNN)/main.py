from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

from Machine_Learning_Module.Classifier import Classifier


def train(save_path):
    data = pd.read_csv('dataset/diabetes.csv')
    # print(data.columns)
    # print(data.head())

    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in zero_not_accepted:
        data[col] = data[col].replace(0, np.nan)
        mean = int(data[col].mean(skipna=True))
        data[col] = data[col].replace(np.nan, mean)

    x = data.iloc[:, :8]
    y = data.iloc[:, 8]

    sc_x = StandardScaler()
    x = sc_x.fit_transform(x)

    model = Classifier(
        model=KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean'),
        model_type='classification',
        x=x,
        y=y,
        test_size=.2
    )

    model.train()
    model.test()
    model.save_model(save_path)


if __name__ == '__main__':
    # Train Model
    train('knn.pkl')
