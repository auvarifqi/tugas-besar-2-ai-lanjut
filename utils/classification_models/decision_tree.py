import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10):
        """
        Decision Tree Classifier using ID3 algorithm.
        Parameters:
        - max_depth: int, maximum depth of the tree.
        """
        self.max_depth = max_depth
        self.tree = None

    def _entropy(self, y):
        """Compute entropy of labels y."""
        labels, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _information_gain(self, X_column, y, threshold):
        """Calculate information gain by splitting on a threshold."""
        # Parent entropy
        parent_entropy = self._entropy(y)
        
        # Split into left and right children
        left_indices = X_column <= threshold
        right_indices = X_column > threshold
        left_y, right_y = y[left_indices], y[right_indices]
        
        # Calculate child entropy
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        child_entropy = (n_left / n) * self._entropy(left_y) + (n_right / n) * self._entropy(right_y)
        
        # Return information gain
        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        """Find the best split for a node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # Loop through all features
        for feature_idx in range(X.shape[1]):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            
            # Test all thresholds for splitting
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, depth):
        """Recursive function to build the tree."""
        # Base case: max depth reached or pure node
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            most_common_label = np.bincount(y).argmax()
            return most_common_label

        # Find the best split
        feature_idx, threshold, gain = self._best_split(X, y)
        
        # If no gain, return majority label
        if gain == -1:
            most_common_label = np.bincount(y).argmax()
            return most_common_label

        # Split into left and right nodes
        left_indices = X[:, feature_idx] <= threshold
        right_indices = X[:, feature_idx] > threshold

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return the tree as a dictionary
        return {
            "feature_idx": feature_idx,
            "threshold": threshold,
            "left": left_subtree,
            "right": right_subtree,
        }

    def fit(self, X, y):
        """Fit the Decision Tree model."""
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_sample(self, sample, tree):
        """Predict class for a single sample."""
        # If leaf node, return label
        if not isinstance(tree, dict):
            return tree

        # Traverse the tree
        feature_idx = tree["feature_idx"]
        threshold = tree["threshold"]
        if sample[feature_idx] <= threshold:
            return self._predict_sample(sample, tree["left"])
        else:
            return self._predict_sample(sample, tree["right"])

    def predict(self, X):
        """Predict classes for samples in X."""
        return np.array([self._predict_sample(sample, self.tree) for sample in X])
