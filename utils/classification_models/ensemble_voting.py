import numpy as np

class EnsembleVoter:
    def __init__(self, model1, model2, prefer_model2=True):
        """
        Initialize the ensemble voter with two models.
        Args:
            model1: First model (LogisticRegression in this case)
            model2: Second model (KNN in this case)
            prefer_model2: If True, use model2's prediction when models disagree
        """
        self.model1 = model1
        self.model2 = model2
        self.prefer_model2 = prefer_model2
    
    def fit(self, X_train, y_train):
        """
        Fit both models with the training data
        """
        # Handle different input types (numpy array vs pandas DataFrame)
        X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
        y_train_values = y_train.values.ravel() if hasattr(y_train, 'values') else y_train
        
        # Fit both models
        self.model1.fit(X_train, y_train_values)
        self.model2.fit(X_train_values, y_train_values)
        
        return self
    
    def predict(self, X_test):
        """
        Make predictions using both models and combine them using hard voting
        """
        # Handle different input types
        X_test_values = X_test.values if hasattr(X_test, 'values') else X_test
        
        # Get predictions from both models
        pred1 = self.model1.predict(X_test)
        pred2 = self.model2.predict(X_test_values)
        
        # Initialize final predictions array
        final_predictions = np.zeros_like(pred1)
        
        # Implement hard voting with preference for model2 (KNN) when there's disagreement
        for i in range(len(pred1)):
            if pred1[i] == pred2[i]:
                final_predictions[i] = pred1[i]
            else:
                # When models disagree, use model2's prediction if prefer_model2 is True
                final_predictions[i] = pred2[i] if self.prefer_model2 else pred1[i]
        
        return final_predictions