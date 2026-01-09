import numpy as np

class PCAEngine:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # 1. Mean Centering (Standardization)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Covariance Matrix Calculation
        # Mengukur redundansi antar pengukuran
        n_samples = X.shape[0]
        covariance_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # 3. Eigen Decomposition
        # Mencari arah varians maksimum
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # 4. Sortir eigenvectors berdasarkan eigenvalues terbesar
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, idxs[0:self.n_components]]

    def transform(self, X):
        # Proyeksi data ke basis baru (Principal Components)
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)