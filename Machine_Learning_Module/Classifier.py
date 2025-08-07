import joblib
from numpy import array
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Classifier:
    def __init__(self, model, model_type=None, x=None, y=None, test_size=.2):
        self.model_type = model_type
        if isinstance(model, str):
            self.model = joblib.load(model)
            self.is_train = False
        else:
            self.model = model
            self.test_size = test_size
            self.is_train = True

            self.x_train, self.x_test, self.y_train, self.y_test = self.split_data(x, y)

    @staticmethod
    def model_train_check(func):
        def wrapper(self, *args, **kwargs):
            if self.is_train:
                return func(self, *args, **kwargs)
            else:
                raise Exception('Model has already been trained.')

        return wrapper

    @model_train_check
    def split_data(self, x, y):
        if self.model_type == "classification":
            stratify = y
        else:
            stratify = None
        result = train_test_split(
            x, y,
            test_size=self.test_size,
            random_state=42,
            stratify=stratify
        )
        return result

    @model_train_check
    def train(self):
        self.model.fit(self.x_train, self.y_train)

    def set_grid_search(self, **kwargs):
        grid_search = GridSearchCV(self.model, kwargs, cv=5)
        grid_search.fit(self.x_train, self.y_train)
        self.model = grid_search.best_estimator_

    def predict(self, *args, **kwargs):
        args = list(args)
        args.extend(kwargs.values())
        args = array(args)
        args = args.reshape(1, -1)
        return self.model.predict(args)

    @model_train_check
    def test(self):
        y_pred = self.model.predict(self.x_test)
        r2 = r2_score(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        print("R2 Score =", r2)
        print("MAE Score =", mae)

        if self.model_type == 'classification':
            acc = accuracy_score(self.y_test, y_pred)
            print("Accuracy Score =", acc)
            cm = confusion_matrix(self.y_test, y_pred)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
            plt.xlabel('Predicted')
            plt.ylabel('Y Test')
            plt.show()

    @model_train_check
    def save_model(self, path):
        joblib.dump(self.model, path)
        print("Model saved")
