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

    def __init__(self, y=1, build=False, data_transformations=None, keyboard=False):

        self.trainX, self.trainY, self.testX, self.testY = None, None, None, None

        self.mean_length = None

        self.y = y

        self.trained = False
        self.optimized = False
        self.min_max = None

        if data_transformations:
            self.data_transformations = data_transformations

        if keyboard:
            self.data_transformations = [
                lambda x: np.histogram(x, bins=50)[0].flatten(),
                lambda x: np.histogram(x, bins=50, range=self.min_max)[0].flatten(),
                lambda x: np.histogram(x, bins=10, range=self.min_max)[0].flatten(),
            ]

    def build_dataset(self, trainX, testX, trainY, testY, min_max=False):
        """ Build the dataset that would be used in training and optimization

        """
        self.trainX = trainX
        self.testX = testX

        self.trainY = np.array(trainY)
        self.testY = np.array(testY)
        assert all([len(_)!=0 for _ in trainX]), "traind data with length 0"
        assert all([len(_)!=0 for _ in testX]), "train data with length 0"
        assert len(set(testY)) == 2, print(testY)
        assert len(set(trainY)) == 2, print(trainY)

        self.mean_length = int(np.mean([len(_) for _ in
                                        list(map(lambda x: x[0],
                                        filter(lambda x: x[1] == 1,
                                        zip(self.testX, self.testY))))]))

        if min_max:
            min_max = np.array([np.array([np.min(_), np.max(_)]) for _ in
                                        list(map(lambda x: x[0],
                                        filter(lambda x: x[1] == 1,
                                        zip(self.testX, self.testY))))])
            self.min_max = (int(np.median(min_max[:,0])), int(np.median(min_max[:,1])))
        else:
            self.min_max = None


    def build_ensemble(self, models=None):
        """ Generate an ensemble model for the ensemble """
        print(self.learning_models)

    def __str__(self):
        return 'Ensemble model: \nlen(X) = {}, \nTrained: {}, \nOptimized: {}'.format(
                 len(self.X), self.trained, self.optimized)


def train_ensemble(ensemble: Ensemble, population=0, generations=10):
    trainX, testX, trainY, testY = train_test_split(ensemble.trainX,
                                                    ensemble.trainY)
    models = []

    for lm in ensemble.learning_models:
        for param in lm.get('params', [{}]):
            for dt in ensemble.data_transformations:
                model = lm['algorithm']
                model = model(**param)

                model.fit([dt(_) for _ in trainX], trainY)

                performance = measure_performance(
                    testY, model.predict([dt(_) for _ in testX]))

                models.append(SingleModel(model, dt, performance))

    models = list(filter(lambda x: x.far != 0 and x.frr != 0, models))

    for generation in range(generations):
        print('after generation ', generation)

        trainX, testX, trainY, testY = train_test_split(ensemble.trainX,
                                                        ensemble.trainY)

        for single_model in models:
            single_model.model.fit([single_model.dt(_) for _ in trainX], trainY)

            perf = measure_performance(
                testY,
                single_model.model.predict([single_model.dt(_) for _ in testX])
            )
            single_model.far += perf[0]
            single_model.frr += perf[1]
            single_model.tpr += perf[2]

    return models

