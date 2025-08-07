from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

from Machine_Learning_Module.Classifier import Classifier


def train(save_path):
    data = pd.read_csv('dataset/diabetes.csv')
    # print(data.head(10))

    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    for col in zero_not_accepted:
        data[col] = data[col].replace(0, np.nan)
        mean = int(data[col].mean(skipna=True))
        data[col] = data[col].replace(np.nan, mean)

    x = data.iloc[:, :8]
    y = data.iloc[:, 8]

    model = Classifier(
        model=DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=3, min_samples_leaf=5),
        model_type='classification',
        x=x,
        y=y,
        test_size=.2
    )

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model.set_grid_search(**param_grid)

    model.train()
    model.test()
    model.save_model(save_path)


if __name__ == '__main__':
    # Train Model
    train('decision_trees.pkl')