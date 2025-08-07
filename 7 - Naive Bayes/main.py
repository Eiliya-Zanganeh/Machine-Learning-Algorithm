from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from Machine_Learning_Module.Classifier import Classifier


def train(save_path):
    all_data = fetch_20newsgroups(subset='all')
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')

    print(len(all_data.data))
    print(len(train_data.data))
    print(len(test_data.data))

    print(all_data.data[0])
    print(all_data.target[0])
    print(all_data.target_names)

    pipeline = make_pipeline(TfidfVectorizer(), MultinomialNB())

    model = Classifier(
        model=pipeline,
        model_type='classification',
        x=all_data.data,
        y=all_data.target,
        test_size=.2
    )

    model.train()
    model.test()
    model.save_model(save_path)


if __name__ == '__main__':
    # Train Model
    train('naive_bayes.pkl')
