import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

from Machine_Learning_Module.Classifier import Classifier


def train(save_path):
    dataset = pd.read_csv('dataset/1000_Companies.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    label_encoder = LabelEncoder()
    x[:, 3] = label_encoder.fit_transform(x[:, 3])

    model = Classifier(
        model=LinearRegression(),
        x=x,
        y=y,
        test_size=.2
    )

    model.train()
    model.test()
    model.save_model(save_path)


if __name__ == '__main__':
    # Train Model
    # train('linear_regression.pkl')

    # Use Model
    model = Classifier('linear_regression.pkl')
    result = model.predict(
        RD_Spend=165349.20,
        Administration=136897.80,
        Marketing_Spend=471784.10,
        State=2
    )
    print(result)
