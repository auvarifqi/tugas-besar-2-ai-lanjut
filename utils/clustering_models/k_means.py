import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, K=3, max_iters=100, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.random_state = random_state

        # List untuk menyimpan indeks sampel pada tiap cluster
        self.clusters = [[] for _ in range(self.K)]

        # List untuk menyimpan centroid (mean) dari tiap cluster
        self.centroids = []

    def fit(self, X):
        """
        Fit the KMeans model to the data.
        Parameters:
        - X: np.ndarray
            The input data to cluster.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # init centroid secara acak dari sampel dalam dataset
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples ke centroid terdekat
            self.clusters = self._create_clusters(self.centroids)

            # hitung centroid baru berdasarkan cluster
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # Periksa konvergensi (apakah centroid tidak berubah)
            if self._is_converged(centroids_old, self.centroids):
                break

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        Parameters:
        - X: np.ndarray
            The input data to predict.
        Returns:
        - labels: np.ndarray
            Cluster labels for each sample in X.
        """
        labels = np.empty(X.shape[0])

        for idx, sample in enumerate(X):
            distances = [euclidean(sample, centroid) for centroid in self.centroids]
            labels[idx] = np.argmin(distances)

        return labels.astype(int)

    def fit_predict(self, X):
        """
        Fit the KMeans model and return cluster labels.
        Parameters:
        - X: np.ndarray
            The input data to cluster.
        Returns:
        - labels: np.ndarray
            Cluster labels for each sample in X.
        """
        self.fit(X)
        return self.predict(X)

    # Mendapatkan label cluster untuk tiap sampel berdasarkan assignment cluster.
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    # Membuat cluster dengan mengassign sampel ke centroid terdekat.
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # Menentukan centroid terdekat untuk sebuah sampel.
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    # Mendapatkan centroid baru berdasarkan rata-rata nilai sampel dalam cluster.
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    # Periksa konvergensi dengan membandingkan centroid baru dan centroid lama.
    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0
    

def elbow_method_kmeans(df, max_k=10):
    """
    Perform elbow method analysis and visualize results with optimal k highlighted.
    """

    def find_optimal_k(sse_values):
        """
        Find the optimal k using the elbow method by identifying the point of maximum curvature
        """
        # Calculate the differences and rate of change
        differences = np.diff(sse_values)
        differences_rate = np.diff(differences)
        
        # Find the elbow point (point of maximum curvature)
        elbow_index = np.argmax(np.abs(differences_rate)) + 2
        return elbow_index

    # Standardize the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df)
    
    # Calculate SSE for different values of k
    sse = []
    K = range(1, max_k + 1)
    
    for k in K:
        kmeans = KMeans(K=k, max_iters=100, random_state=42)
        
        # Use fit instead of predict to ensure centroids are initialized
        kmeans.fit(scaled_features)
        
        # Calculate SSE
        current_sse = 0
        for i, cluster in enumerate(kmeans.clusters):
            if len(cluster) > 0:  # Check if cluster is not empty
                cluster_points = scaled_features[cluster]
                centroid = kmeans.centroids[i]
                current_sse += np.sum((cluster_points - centroid) ** 2)
        
        sse.append(current_sse)
    
    # Find optimal k
    optimal_k = find_optimal_k(sse)
    
    # Create the visualization
    plt.figure(figsize=(12, 7))
    
    # Plot the elbow curve
    plt.plot(K, sse, 'bo-', linewidth=2, markersize=8, label='SSE curve')
    
    # Add vertical line at optimal k
    plt.axvline(x=optimal_k, color='red', linestyle='--', 
                label=f'Optimal k = {optimal_k}')
    
    # Highlight the optimal point
    plt.plot(optimal_k, sse[optimal_k - 1], 'ro', markersize=12)
    
    # Customize the plot
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=12)
    plt.title('Elbow Method Analysis with Optimal k', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Add text annotation for optimal k
    plt.annotate(f'Optimal k = {optimal_k}',
                 xy=(optimal_k, sse[optimal_k - 1]),
                 xytext=(optimal_k + 0.5, sse[optimal_k - 1]),
                 fontsize=10,
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    plt.show()
    
    return optimal_k