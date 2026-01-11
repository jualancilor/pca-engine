"""
Mathematical correctness tests for PCAEngine.

Tests focus on verifying mathematical properties and correctness:
- Eigenvalue/eigenvector properties
- Variance preservation
- Orthogonality constraints
- Comparison with known analytical solutions
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pca import PCAEngine


class TestMathematicalProperties:
    """Test mathematical properties of PCA."""

    def test_components_are_eigenvectors(self):
        """Verify that components are eigenvectors of covariance matrix."""
        np.random.seed(42)
        X = np.random.randn(100, 4)

        pca = PCAEngine(n_components=2)
        pca.fit(X)

        # Calculate covariance matrix
        X_centered = X - pca.mean
        cov_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

        # Check that components are eigenvectors
        for i in range(pca.n_components):
            component = pca.components[:, i]
            # Cov * v should be parallel to v (i.e., Cov * v = λ * v)
            result = np.dot(cov_matrix, component)
            # Check if result is parallel to component
            ratio = result / component
            # All elements of ratio should be approximately equal (the eigenvalue)
            assert np.allclose(ratio, ratio[0], rtol=1e-10)

    def test_eigenvalue_ordering(self):
        """Test that components are ordered by decreasing eigenvalues."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        pca = PCAEngine(n_components=4)
        pca.fit(X)

        # Transform and calculate variance of each component
        X_transformed = pca.transform(X)
        variances = np.var(X_transformed, axis=0)

        # Variance should be in descending order
        for i in range(len(variances) - 1):
            assert variances[i] >= variances[i + 1], \
                f"Variance not descending: {variances[i]} < {variances[i+1]}"

    def test_total_variance_preservation(self):
        """Test that total variance is preserved (for full PCA)."""
        np.random.seed(42)
        n_features = 5
        X = np.random.randn(100, n_features)

        # Full PCA (all components)
        pca = PCAEngine(n_components=n_features)
        pca.fit(X)
        X_transformed = pca.transform(X)

        # Calculate total variance
        original_var = np.sum(np.var(X - pca.mean, axis=0))
        transformed_var = np.sum(np.var(X_transformed, axis=0))

        # Should be approximately equal
        np.testing.assert_almost_equal(
            original_var, transformed_var, decimal=10,
            err_msg="Total variance not preserved"
        )

    def test_components_form_orthonormal_basis(self):
        """Test that all components form an orthonormal set."""
        np.random.seed(42)
        X = np.random.randn(100, 5)

        pca = PCAEngine(n_components=4)
        pca.fit(X)

        # Calculate gram matrix (should be identity)
        gram_matrix = np.dot(pca.components.T, pca.components)
        identity = np.eye(pca.n_components)

        np.testing.assert_array_almost_equal(
            gram_matrix, identity, decimal=10,
            err_msg="Components do not form orthonormal basis"
        )

    def test_reconstruction_error_decreases_with_components(self):
        """Test that reconstruction error decreases as we add components."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        errors = []
        for n_comp in range(1, 6):
            pca = PCAEngine(n_components=n_comp)
            pca.fit(X)
            X_transformed = pca.transform(X)
            # Reconstruct
            X_reconstructed = np.dot(X_transformed, pca.components.T) + pca.mean
            # Calculate error
            error = np.mean((X - X_reconstructed) ** 2)
            errors.append(error)

        # Error should be monotonically decreasing
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1], \
                f"Reconstruction error increased: {errors[i]} < {errors[i+1]}"


class TestKnownSolutions:
    """Test against known analytical solutions."""

    def test_diagonal_covariance_simple(self):
        """Test PCA on data with diagonal covariance (uncorrelated features)."""
        np.random.seed(42)
        # Create data with known variance in each dimension
        n_samples = 1000
        X = np.zeros((n_samples, 3))
        X[:, 0] = np.random.randn(n_samples) * 3.0  # variance = 9
        X[:, 1] = np.random.randn(n_samples) * 2.0  # variance = 4
        X[:, 2] = np.random.randn(n_samples) * 1.0  # variance = 1

        pca = PCAEngine(n_components=3)
        pca.fit(X)
        X_transformed = pca.transform(X)

        # First component should capture most variance
        variances = np.var(X_transformed, axis=0)

        # Check ordering (with some tolerance for randomness)
        assert variances[0] > variances[1] > variances[2]

    def test_perfectly_correlated_2d(self):
        """Test PCA on perfectly correlated 2D data."""
        # Create data on a line y = 2x
        n_samples = 100
        x = np.linspace(-5, 5, n_samples)
        X = np.column_stack([x, 2 * x])

        pca = PCAEngine(n_components=2)
        pca.fit(X)
        X_transformed = pca.transform(X)

        # Second component should have near-zero variance
        variances = np.var(X_transformed, axis=0)
        assert variances[1] < 1e-10, \
            f"Expected near-zero variance in 2nd component, got {variances[1]}"

    def test_identity_transformation_with_centered_data(self):
        """Test that PCA on already-principal data maintains properties."""
        np.random.seed(42)
        # Create data already in principal component space
        n_samples = 100
        X_pc = np.zeros((n_samples, 3))
        X_pc[:, 0] = np.random.randn(n_samples) * 3.0
        X_pc[:, 1] = np.random.randn(n_samples) * 2.0
        X_pc[:, 2] = np.random.randn(n_samples) * 1.0

        pca = PCAEngine(n_components=3)
        pca.fit(X_pc)

        # Components should be close to identity (or permutation/sign flip)
        # Check that components form orthonormal basis aligned with axes
        abs_components = np.abs(pca.components)

        # Each component should be mostly aligned with one axis
        for i in range(3):
            max_val = np.max(abs_components[:, i])
            assert max_val > 0.9, \
                f"Component {i} not aligned with any axis"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_component(self):
        """Test PCA with single component."""
        np.random.seed(42)
        X = np.random.randn(50, 10)

        pca = PCAEngine(n_components=1)
        pca.fit(X)
        X_transformed = pca.transform(X)

        assert X_transformed.shape == (50, 1)
        assert pca.components.shape == (10, 1)

    def test_all_components(self):
        """Test PCA with all possible components."""
        np.random.seed(42)
        n_features = 5
        X = np.random.randn(100, n_features)

        pca = PCAEngine(n_components=n_features)
        pca.fit(X)
        X_transformed = pca.transform(X)

        assert X_transformed.shape == (100, n_features)
        assert pca.components.shape == (n_features, n_features)

    def test_constant_feature(self):
        """Test PCA with one constant feature."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 2] = 5.0  # Constant feature

        pca = PCAEngine(n_components=2)
        pca.fit(X)

        # Should still work, constant feature contributes no variance
        assert pca.components.shape == (3, 2)

    def test_zero_centered_data(self):
        """Test PCA on data already centered at zero."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X_centered = X - np.mean(X, axis=0)

        pca = PCAEngine(n_components=3)
        pca.fit(X_centered)

        # Mean should be close to zero
        np.testing.assert_array_almost_equal(
            pca.mean, np.zeros(5), decimal=10
        )


class TestNumericalStability:
    """Test numerical stability of the implementation."""

    def test_large_values(self):
        """Test PCA with large magnitude values."""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 1e6

        pca = PCAEngine(n_components=3)
        pca.fit(X)
        X_transformed = pca.transform(X)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))

    def test_small_values(self):
        """Test PCA with small magnitude values."""
        np.random.seed(42)
        X = np.random.randn(100, 5) * 1e-6

        pca = PCAEngine(n_components=3)
        pca.fit(X)
        X_transformed = pca.transform(X)

        # Should not produce NaN or Inf
        assert not np.any(np.isnan(X_transformed))
        assert not np.any(np.isinf(X_transformed))

    def test_repeated_values(self):
        """Test PCA with many repeated values."""
        np.random.seed(42)
        X = np.random.choice([1.0, 2.0, 3.0], size=(100, 5))

        pca = PCAEngine(n_components=3)
        pca.fit(X)

        # Should complete without error
        assert pca.components is not None
        assert not np.any(np.isnan(pca.components))
