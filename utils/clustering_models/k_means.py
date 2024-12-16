import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KMeans:
    def __init__(self, K=3, max_iters=100, random_state=42):
        self.K = K
        self.max_iters = max_iters
        self.random_state = random_state
        
        # List untuk menyimpan indeks sampel pada tiap cluster
        self.clusters = [[] for _ in range(self.K)]
        
        # List untuk menyimpan centroid (mean) dari tiap cluster
        self.centroids = []
        
        # Flag untuk mengecek apakah model sudah di-fit
        self.is_fitted = False

    def fit(self, X):
        """
        Melatih model KMeans dengan data training.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data training untuk fitting model
        
        Returns:
        --------
        self : object
            Returns self
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # Inisialisasi centroid secara acak dari sampel dalam dataset
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples ke centroid terdekat
            self.clusters = self._create_clusters(self.centroids)
            
            # Hitung centroid baru berdasarkan cluster
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # Periksa konvergensi
            if self._is_converged(centroids_old, self.centroids):
                break
        
        self.is_fitted = True
        return self

    def predict(self, X):
        """
        Memprediksi cluster untuk data X.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data baru untuk diprediksi cluster-nya
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Label cluster untuk setiap data point
        """
        if not self.is_fitted:
            raise Exception("Model belum di-fit! Panggil fit() terlebih dahulu.")
            
        # Untuk setiap sample dalam X, tentukan centroid terdekat
        labels = np.empty(X.shape[0])
        for i, sample in enumerate(X):
            closest_centroid = self._closest_centroid(sample, self.centroids)
            labels[i] = closest_centroid
            
        return labels

    def fit_predict(self, X):
        """
        Melakukan fit model dan langsung memprediksi cluster untuk data training.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Data training
            
        Returns:
        --------
        labels : array, shape (n_samples,)
            Label cluster untuk data training
        """
        self.fit(X)
        return self._get_cluster_labels(self.clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) > 0:  # Hindari division by zero
                cluster_mean = np.mean(self.X[cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        return centroids

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0