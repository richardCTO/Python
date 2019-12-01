#%%
import numpy as np
# K nearest neighbor
class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self, x, y):
        self.X_train = X 
        self.y_train = y
        
    
    def predict(self, x):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x):
        pass