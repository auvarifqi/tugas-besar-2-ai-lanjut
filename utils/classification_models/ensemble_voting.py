import numpy as np
from collections import Counter

class EnsembleVoter:
    def __init__(self, models, tie_break_model=None):
        """
        Initialize the ensemble voter with a list of models.
        Args:
            models: List of models to be used in the ensemble.
            tie_break_model: Model to use in case of a tie in voting. If None, no preference is given.
        """
        self.models = models
        self.tie_break_model = tie_break_model

    def fit(self, X_train, y_train):
        """
        Fit all models in the ensemble with the training data.
        """
        # Handle different input types (numpy array vs pandas DataFrame)
        X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_values = y_train.values.ravel() if hasattr(y_train, 'values') else y_train

        # Fit all models
        for model in self.models:
            model.fit(X_train_values, y_train_values)
        
        return self

    def predict(self, X_test):
        """
        Make predictions using all models and combine them using hard voting.
        """
        # Handle different input types
        X_test_values = X_test.values if hasattr(X_test, 'values') else X_test

        # Get predictions from all models
        all_predictions = np.array([model.predict(X_test_values) for model in self.models])

        # Perform majority voting
        final_predictions = []
        for i in range(all_predictions.shape[1]):
            # Get predictions for the i-th sample from all models
            sample_predictions = all_predictions[:, i]
            vote_count = Counter(sample_predictions)
            
            # Find the class with the maximum votes
            max_votes = max(vote_count.values())
            candidates = [key for key, val in vote_count.items() if val == max_votes]

            # Resolve ties using the tie_break_model
            if len(candidates) == 1:
                # Unique winner
                final_predictions.append(candidates[0])
            else:
                # Tie occurred, use tie_break_model if provided
                if self.tie_break_model is not None:
                    tie_break_prediction = self.tie_break_model.predict(X_test_values[i].reshape(1, -1))[0]
                    if tie_break_prediction in candidates:
                        final_predictions.append(tie_break_prediction)
                    else:
                        final_predictions.append(candidates[0])  # Default to the first candidate
                else:
                    # No tie_break_model, default to the first candidate
                    final_predictions.append(candidates[0])
        
        return np.array(final_predictions)
