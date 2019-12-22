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
    ...