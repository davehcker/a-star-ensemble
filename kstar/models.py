from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

generic_models = [
        {"algorithm": SVC,
         "params": [
             {'kernel': 'linear', 'gamma': 'auto', 'degree': 3, 'class_weight': 'balanced', 'cache_size': 10}
         ]},
        {"algorithm": AdaBoostClassifier,
         "params": [
             {'n_estimators': 100},
         ]},
        {"algorithm": GaussianNB},
        {"algorithm": RandomForestClassifier,
         "params": [
             {'n_estimators': 20},
         ]},
        {"algorithm": BernoulliNB},
        {"algorithm": MultinomialNB},
        {"algorithm": SGDClassifier},
        {"algorithm": MLPClassifier,
         "params": [
             {'max_iter': 1000},
         ]},
    ]


class SingleModel():
    """Hold a single ML model with additional info"""
    def __init__(self, model, dt, performance):
        self.model = model
        self.far = performance[0]
        self.frr = performance[1]
        self.tpr = performance[2]

        self.dt = dt

    def __str__(self):
        return str(self.model)