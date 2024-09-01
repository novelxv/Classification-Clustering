import numpy as np

class PCAFromScratch:
    def __init__(self, n_components):

        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X):
        # Standarisasi data (mean = 0, variance = 1)
        X = X - np.mean(X, axis=0)
        
        # Hitung matriks kovarian
        cov_matrix = np.cov(X, rowvar=False)
        
        # Hitung eigenvalues dan eigenvectors dari matriks kovarian
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Urutkan eigenvalues dan eigenvectors dari besar ke kecil
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Normalisasi tanda eigenvectors
        for i in range(eigenvectors.shape[1]):
            if eigenvectors[:, i].max() < np.abs(eigenvectors[:, i].min()):
                eigenvectors[:, i] = -eigenvectors[:, i]
        
        # Simpan n_components pertama
        self.components_ = eigenvectors[:, :self.n_components]
        
        # Explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = [(ev / total_variance) for ev in eigenvalues[:self.n_components]]

    def transform(self, X):
        X = X - np.mean(X, axis=0)
        return np.dot(X, self.components_)
