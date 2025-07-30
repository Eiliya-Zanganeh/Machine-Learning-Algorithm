from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from Machine_Learning_Module.Classifier import Classifier


def train(save_path):
    digits = load_digits()

    print(f'Image : {digits.data.shape}')
    print(f'Label : {digits.target.shape}')

    print(digits.data[0].shape)
    # print(digits.data[0])

    # df = pd.DataFrame(digits.data, columns=digits.feature_names)
    # df['result'] = pd.Categorical.from_codes(digits.target, digits.target_names)
    # print(df.head())

    # index = 20
    # img = digits.data[index]
    # label = digits.target[index]
    # img = np.reshape(img, (8, 8))
    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.title(f'Label : {label}')
    # plt.show()

    sc_x = StandardScaler()
    x = sc_x.fit_transform(digits.data)
    y = digits.target

    model = Classifier(
        model=LogisticRegression(),
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
    # train('logistic_regression.pkl')

    model = Classifier('logistic_regression.pkl')
    index = 0
    img = load_digits().data[index]
    label = load_digits().target[index]
    print(f'Predict : {model.predict(img)}')
    print(f'Label : {label}')