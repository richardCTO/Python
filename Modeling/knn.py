#%%
import numpy as np
from collections import Counter

# calculation for kkn using euclidean
def euclidean_distance(x1, x2):
    np.sqrt(np.sum((x1-x2)**2))
    
# K nearest neighbor from scratch 
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
        # compute distances
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get k nearest samples, labels
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indicies]
        
        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        