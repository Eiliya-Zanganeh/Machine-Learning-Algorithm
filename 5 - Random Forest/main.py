from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from Machine_Learning_Module.Classifier import Classifier


def train(save_path):
    iris = load_iris()
    x = iris.data
    y = iris.target

    model = Classifier(
        model=RandomForestClassifier(n_jobs=2, random_state=0),
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
    train('random_forest.pkl')
