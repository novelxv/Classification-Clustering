import numpy as np
from scipy.spatial import distance

class DBSCANFromScratch:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1)  # Semua titik awalnya dilabeli sebagai noise (-1)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._find_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                # Tetap noise
                self.labels[i] = -1
            else:
                # Expand cluster
                self._expand_cluster(X, i, neighbors, cluster_id, visited)
                cluster_id += 1

    def _find_neighbors(self, X, point_idx):
        # Mengembalikan indeks tetangga yang berada dalam jarak epsilon dari point_idx
        distances = self._calculate_distances(X, point_idx)
        return np.where(distances < self.eps)[0]

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, visited):
        # Menambahkan titik ke cluster dan mencari tetangga baru untuk dimasukkan
        self.labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]

            if not visited[neighbor]:
                visited[neighbor] = True
                new_neighbors = self._find_neighbors(X, neighbor)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.append(neighbors, new_neighbors)

            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id

            i += 1

    def _calculate_distances(self, X, point_idx):
        if self.metric == 'euclidean':
            return np.linalg.norm(X - X[point_idx], axis=1)
        elif self.metric == 'manhattan':
            return distance.cdist(X, [X[point_idx]], metric='cityblock').reshape(-1)
        elif self.metric == 'minkowski':
            p = 3  
            return distance.cdist(X, [X[point_idx]], metric='minkowski', p=p).reshape(-1)