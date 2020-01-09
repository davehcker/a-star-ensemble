import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class _Ensemble:
    """Generic Ensemble class.

    Args:
        k (int): class identifier

    Returns:
        ensemble (Ensemble): An Ensemble for the specified class k.
    """

    def __init__(self):
        pass

    def train(self):
        """Train self (Ensemble)

        Returns: self (Ensemble)
        """
        pass

    def optimize(self):
        """Do hyper-parameter tuning.

        Returns: self """

    def info(self,
             metrics=False):
        """Print info about the ensemble classifier.

        Args:
            metrics (bool): Flag to print metrics such as accuracy.
        Returns: None
        """
        pass

    def predict(self, X):
        """Predict if the X is self.class or not

        Returns:
            prediction(bool): True if true else False

        """
        pass


class Ensemble():
    """Generate an ensemble model for combining multiple modalities"""

    def __init__(self, X, Y, class_label=1):
        self.X = np.array(X)
        self.Y = np.where(np.array(Y) == class_label, 1, 0)