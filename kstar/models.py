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


def measure_performance(actual, prediction):
    """
    Return a performance breakdown in the form FAR, FRR and TPR

    Args:
        actual: correct (a priori) predictions
        prediction: predictions of a model

    Returns:
        res: List<FAR:float, FRR:float, TPR:float>

    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    assert len(actual) == len(prediction), "Array lengths do not match"

    for i in range(len(actual)):
        if actual[i] == 1:
            if prediction[i] == 1:
                tp += 1
            else:
                fn += 1
        elif actual[i] == 0:
            if prediction[i] == 0:
                tn += 1
            if prediction[i] == 1:
                fp += 1

    assert tp + fp + tn + fn == len(actual), "Class labels should be binary (1 or 0)"
    res = [fp / (fp + tn) if (fp + tn) != 0 else 0,  # FAR
           fn / (tp + fn) if (tp + fn) != 0 else 0,  # FRR
           (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0,  # TPR
           ]

    return res



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