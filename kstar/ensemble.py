from .data import data_transformations
from .models import generic_models

import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

class _Ensemble:
    """Generic Ensemble class.

    Args:
        k (int): class identifier

    Returns:
        ensemble (Ensemble): An Ensemble for the specified class k.
    """
    data_transformations = data_transformations

    learning_models = generic_models

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


class Ensemble(_Ensemble):
    """Generate an ensemble model for combining multiple modalities"""

    def __init__(self, X, Y, y=1, build=False):
        self.X = np.array(X)
        self.Y = np.array(Y)

        self.trainX, self.trainY, self.testX, self.testY = None, None, None, None

        assert len(self.X) == len(self.Y), "sizes of X and Y has to be same"

        self.y = y

        self.trained = False
        self.optimized = False

    def build_dataset(self, source_ratio=(0.5, 0.5), test_size = 0.2):
        """ Build the dataset that would be used in training and optimization

        This method takes into consideration the following steps:
            1. A filtering on total number of examples to consider
            2. A split between training and test set (this is further used in optimization steps)
        Args:
            train_instances (float, float): ratio/number of training instance to take for (self.y, !self.y)
        """
        source_ratio_self, source_ratio_other = source_ratio

        if type(source_ratio_self) == float:
            source_ratio_self = int(sum(np.where(self.Y == self.y, 1, 0)) * source_ratio[0])
        if type(source_ratio_other) == float:
            source_ratio_other = int(sum(np.where(self.Y != self.y, 1, 0)) * source_ratio[1])

        source_ratio = (source_ratio_self, source_ratio_other)

        x_class_self = np.random.choice(self.X[np.where(self.Y == self.y, True, False)], source_ratio[0])
        x_class_other = np.random.choice(self.X[np.where(self.Y == self.y, False, True)], source_ratio[1])

        self.trainX = np.concatenate((np.random.choice(x_class_self, source_ratio[0]),
                        np.random.choice(x_class_other, source_ratio[1])))

        self.trainY = np.concatenate(([self.y for _ in range(len(x_class_self))],
                                      [0 for _ in range(len(x_class_other))]))

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(
            self.trainX, self.trainY, test_size=test_size)

    def build_ensemble(self, models=None):
        """ Generate an ensemble model for the ensemble """
        print(self.learning_models)

    def __str__(self):
        return 'Ensemble model: \nlen(X) = {}, \nTrained: {}, \nOptimized: {}'.format(
                 len(self.X), self.trained, self.optimized)