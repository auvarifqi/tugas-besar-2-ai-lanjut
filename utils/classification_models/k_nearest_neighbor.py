import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        """
        Initialize the KNN classifier.
        
        Parameters:
        - k (int): Number of neighbors to consider.
        """
        self.k = k

    def fit(self, X, y=None):
        """
        Fit the model to the data.
        
        Parameters:
        - X (np.ndarray): Feature matrix for the training data.
        - y (np.ndarray, optional): Labels for the training data. Optional for unsupervised tasks.
        """
        self.X_train = X
        if y is not None:
            self.y_train = y

    def predict(self, X):
        """
        Predict the class labels for the provided test data.
        
        Parameters:
        - X (np.ndarray): Feature matrix for test data.
        
        Returns:
        - np.ndarray: Predicted class labels.
        """
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        """
        Predict the class label for a single data point.
        
        Parameters:
        - x (np.ndarray): A single test data point.
        
        Returns:
        - int: Predicted class label.
        """
        # Compute distances between x and all training samples
        distances = [euclidean(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Determine the most common label among the k neighbors
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def kneighbors(self, X):
        """
        Find k-nearest neighbors for each data point in X.
        
        Returns:
        - distances: Distances to the k-nearest neighbors.
        """
        distances = []
        for x in X:
            dists = [euclidean(x, x_train) for x_train in self.X_train]
            dists_sorted = sorted(dists)[:self.k]  # Take k smallest distances
            distances.append(dists_sorted[-1])  # Distance to the k-th neighbor
        return np.array(distances)
