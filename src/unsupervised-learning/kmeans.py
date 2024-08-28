import numpy as np

class KMeansFromScratch:
    def __init__(self, n_clusters=3, max_iter=300, init='random'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Inisialisasi centroids
        if self.init == 'random':
            self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        elif self.init == 'kmeans++':
            self.centroids = self._kmeans_plus_plus_init(X)
        
        for _ in range(self.max_iter):
            self.labels = self._assign_clusters(X)
            
            # Update centroid
            new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids

    def predict(self, X):
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        return np.argmin(distances, axis=0)

    def _kmeans_plus_plus_init(self, X):
        n_samples = X.shape[0]
        centroids = []
        # Pilih centroid pertama secara acak
        centroids.append(X[np.random.choice(n_samples)])
        
        for _ in range(1, self.n_clusters):
            # Hitung jarak antara setiap titik dan centroid terdekat
            distances = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
            probs = distances**2 / np.sum(distances**2)
            next_centroid = X[np.random.choice(n_samples, p=probs)]
            centroids.append(next_centroid)
        
        return np.array(centroids)