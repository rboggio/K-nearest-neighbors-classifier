import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from scipy import stats
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.validation import check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.metrics.pairwise import pairwise_distances


class KNearestNeighbors(BaseEstimator, ClassifierMixin):
    """KNearestNeighbors classifier."""

    def __init__(self, n_neighbors=1): 
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """
        Fitting function.

         Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to train the model.
        y : ndarray, shape (n_samples,)
            Labels associated with the training data.

        Returns
        ----------
        self : instance of KNearestNeighbors
            The current instance of the classifier
        """
        self.X_, self.y_ = check_X_y(X, y)
        self.classes_ = np.unique(self.y_)
        check_classification_targets(self.y_)
        self.n_features_in_ = self.X_.shape[1]

        return self

    def predict(self, X):
        """
        Predict function.

        Parameters
        ----------
        X : ndarray, shape (n_test_samples, n_features)
            Data to predict on.

        Returns
        ----------
        y : ndarray, shape (n_test_samples,)
            Predicted class labels for each test data sample.
        """
        check_is_fitted(self)
        X = check_array(X)

        dist = pairwise_distances(X, Y=self.X_)

        idx_sort = np.argsort(dist, axis=1)
        idx_neighbors = idx_sort[:, :self.n_neighbors]

        Y_neighbors = self.y_[idx_neighbors]

        mode, count = stats.mode(Y_neighbors, axis=1, keepdims=False)

        y_pred = np.asarray(mode.reshape(-1))

        return y_pred

    def score(self, X, y):
        """
        Calculate the score of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data to score on.
        y : ndarray, shape (n_samples)
            target values.

        Returns
        ----------
        score : float
            Accuracy of the model computed for the (X, y) pairs.
        """
        return (self.predict(X) == y).mean()