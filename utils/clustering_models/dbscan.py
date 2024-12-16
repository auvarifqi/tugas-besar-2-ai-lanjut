import numpy as np
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from utils.classification_models.k_nearest_neighbor import KNearestNeighbors

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        """
        Initialize DBSCAN clustering.
        
        Parameters:
        - eps (float): Maximum distance between two points to be considered neighbors.
        - min_samples (int): Minimum number of points required to form a dense region.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def fit(self, X):
        """
        Fit the DBSCAN model to the data.
        
        Parameters:
        - X (np.ndarray): Input data.
        """
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # Initialize all labels as noise (-1)
        cluster_id = 0
        
        for i in range(n_samples):
            if self.labels_[i] != -1:  # Skip already visited points
                continue
            
            # Find neighbors of point i
            neighbors = self._region_query(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1  # Mark as noise
            else:
                # Expand the cluster
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
    
    def _region_query(self, X, point_idx):
        """
        Find neighbors within eps distance.
        """
        neighbors = []
        for i, point in enumerate(X):
            if euclidean(X[point_idx], point) <= self.eps:
                neighbors.append(i)
        return neighbors
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        """
        Expand the cluster using neighbors.
        """
        self.labels_[point_idx] = cluster_id
        
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            
            if self.labels_[neighbor_idx] == -1:  # If it is noise, mark as border point
                self.labels_[neighbor_idx] = cluster_id
            elif self.labels_[neighbor_idx] == -1:  # If unvisited
                self.labels_[neighbor_idx] = cluster_id
                new_neighbors = self._region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors.extend(new_neighbors)  # Add to cluster
            
            i += 1
    
    def predict(self):
        """
        Return the cluster labels.
        """
        return self.labels_


def find_optimal_eps(X, k=4):
    """
    Find the optimal eps for DBSCAN using k-nearest neighbors distance plot.
    
    Parameters:
    - X (np.ndarray): Input data.
    - k (int): Number of neighbors to consider.
    """
    # Step 1: Fit NearestNeighbors model
    nn = KNearestNeighbors(k=k)
    nn.fit(X)
    
    # Step 2: Compute distances to k-th nearest neighbor
    k_distances = nn.kneighbors(X)
    
    # Step 3: Sort distances in ascending order
    k_distances = np.sort(k_distances)
    
    # Step 4: Plot the distances
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances)), k_distances, marker='o', linestyle='-', color='b')
    plt.title(f"k-Nearest Neighbors Distance Plot (k={k})")
    plt.xlabel("Data Points Sorted by Distance")
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor")
    plt.grid(True)
    plt.show()
    
    print("**Interpretasi:** Nilai eps optimal dapat dilihat sebagai titik elbow (tekukan) pada grafik di atas.")