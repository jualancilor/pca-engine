"""
End-to-end integration tests for PCAEngine.

Tests complete workflows and real-world usage scenarios including:
- Data preprocessing pipelines
- Multiple fit/transform cycles
- Edge cases in production scenarios
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pca import PCAEngine


class TestDataPipeline:
    """Test complete data processing pipelines."""

    def test_train_test_split_workflow(self):
        """Test typical train/test split workflow."""
        np.random.seed(42)
        # Generate full dataset
        X_full = np.random.randn(200, 10)

        # Split into train/test
        train_size = 150
        X_train = X_full[:train_size]
        X_test = X_full[train_size:]

        # Fit on training data
        pca = PCAEngine(n_components=5)
        pca.fit(X_train)

        # Transform both train and test
        X_train_transformed = pca.transform(X_train)
        X_test_transformed = pca.transform(X_test)

        # Verify shapes
        assert X_train_transformed.shape == (train_size, 5)
        assert X_test_transformed.shape == (50, 5)

        # Test data should use same mean and components
        assert pca.mean is not None
        assert pca.components is not None

    def test_cross_validation_workflow(self):
        """Test k-fold cross-validation workflow."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        n_folds = 5
        fold_size = 20

        results = []
        for i in range(n_folds):
            # Create train/val split
            val_indices = slice(i * fold_size, (i + 1) * fold_size)
            X_val = X[val_indices]
            X_train = np.vstack([X[:i * fold_size], X[(i + 1) * fold_size:]])

            # Fit on train, transform val
            pca = PCAEngine(n_components=3)
            pca.fit(X_train)
            X_val_transformed = pca.transform(X_val)

            results.append(X_val_transformed)

        # All folds should produce valid results
        assert len(results) == n_folds
        for result in results:
            assert result.shape == (fold_size, 3)
            assert not np.any(np.isnan(result))

    def test_incremental_dimensionality_reduction(self):
        """Test reducing dimensions in steps."""
        np.random.seed(42)
        X = np.random.randn(100, 20)

        # First reduction: 20 -> 10
        pca1 = PCAEngine(n_components=10)
        pca1.fit(X)
        X_reduced1 = pca1.transform(X)

        # Second reduction: 10 -> 5
        pca2 = PCAEngine(n_components=5)
        pca2.fit(X_reduced1)
        X_reduced2 = pca2.transform(X_reduced1)

        # Final shape should be (100, 5)
        assert X_reduced2.shape == (100, 5)

        # Should be different from single-step reduction
        pca_direct = PCAEngine(n_components=5)
        pca_direct.fit(X)
        X_direct = pca_direct.transform(X)

        # Two-step will generally differ from one-step
        # but both should be valid
        assert not np.allclose(X_reduced2, X_direct)

    def test_standardization_pipeline(self):
        """Test PCA with manual standardization."""
        np.random.seed(42)
        # Data with different scales
        X = np.column_stack([
            np.random.randn(100) * 100,    # Large scale
            np.random.randn(100) * 0.01,   # Small scale
            np.random.randn(100) * 1.0,    # Medium scale
        ])

        # Apply standardization (z-score normalization)
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_standardized = (X - X_mean) / X_std

        # Apply PCA
        pca = PCAEngine(n_components=2)
        pca.fit(X_standardized)
        X_transformed = pca.transform(X_standardized)

        assert X_transformed.shape == (100, 2)
        assert not np.any(np.isnan(X_transformed))


class TestRepeatedOperations:
    """Test repeated and sequential operations."""

    def test_repeated_fit_same_data(self):
        """Test fitting multiple times on same data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        pca = PCAEngine(n_components=5)

        # Fit three times
        pca.fit(X)
        components1 = pca.components.copy()

        pca.fit(X)
        components2 = pca.components.copy()

        pca.fit(X)
        components3 = pca.components.copy()

        # All should produce same components
        np.testing.assert_array_almost_equal(components1, components2)
        np.testing.assert_array_almost_equal(components2, components3)

    def test_fit_different_data_sequences(self):
        """Test fitting on different datasets sequentially."""
        np.random.seed(42)
        X1 = np.random.randn(100, 10)
        X2 = np.random.randn(150, 10)
        X3 = np.random.randn(80, 10)

        pca = PCAEngine(n_components=5)

        # Fit on different datasets
        pca.fit(X1)
        components1 = pca.components.copy()

        pca.fit(X2)
        components2 = pca.components.copy()

        pca.fit(X3)
        components3 = pca.components.copy()

        # Components should be different
        assert not np.allclose(components1, components2)
        assert not np.allclose(components2, components3)

    def test_multiple_transforms_same_model(self):
        """Test transforming different batches with same model."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)

        pca = PCAEngine(n_components=5)
        pca.fit(X_train)

        # Transform multiple different batches
        batches = [np.random.randn(20, 10) for _ in range(5)]
        results = [pca.transform(batch) for batch in batches]

        # All should have correct shape
        for result in results:
            assert result.shape == (20, 5)
            assert not np.any(np.isnan(result))


class TestRealWorldScenarios:
    """Test scenarios mimicking real-world applications."""

    def test_dimensionality_reduction_for_visualization(self):
        """Test reducing high-dimensional data to 2D for visualization."""
        np.random.seed(42)
        # Simulate high-dimensional data (e.g., from sensors or features)
        n_samples = 200
        n_features = 50
        X = np.random.randn(n_samples, n_features)

        # Reduce to 2D for visualization
        pca = PCAEngine(n_components=2)
        pca.fit(X)
        X_2d = pca.transform(X)

        assert X_2d.shape == (n_samples, 2)

        # Check that data is well-distributed (not collapsed)
        assert np.std(X_2d[:, 0]) > 0.1
        assert np.std(X_2d[:, 1]) > 0.1

    def test_feature_extraction_pipeline(self):
        """Test using PCA for feature extraction."""
        np.random.seed(42)
        # Original high-dimensional features
        X_train = np.random.randn(1000, 100)
        X_test = np.random.randn(200, 100)

        # Extract top 20 principal components as features
        pca = PCAEngine(n_components=20)
        pca.fit(X_train)

        X_train_features = pca.transform(X_train)
        X_test_features = pca.transform(X_test)

        # Verify feature shapes
        assert X_train_features.shape == (1000, 20)
        assert X_test_features.shape == (200, 20)

        # Features should be uncorrelated (decorrelated)
        corr_matrix = np.corrcoef(X_train_features.T)
        # Off-diagonal elements should be small
        off_diagonal = corr_matrix - np.eye(20)
        assert np.max(np.abs(off_diagonal)) < 0.1

    def test_noise_filtering(self):
        """Test using PCA to filter noise from data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 20

        # Create signal with low-rank structure
        signal = np.dot(
            np.random.randn(n_samples, 3),
            np.random.randn(3, n_features)
        )
        # Add noise
        noise = np.random.randn(n_samples, n_features) * 0.5
        X_noisy = signal + noise

        # Use PCA to extract main components (filter noise)
        pca = PCAEngine(n_components=3)
        pca.fit(X_noisy)
        X_denoised_components = pca.transform(X_noisy)

        # Reconstruct from top components
        X_denoised = np.dot(X_denoised_components, pca.components.T) + pca.mean

        # Denoised data should be closer to original signal than noisy data
        noise_level_original = np.mean((X_noisy - signal) ** 2)
        noise_level_denoised = np.mean((X_denoised - signal) ** 2)

        # Denoising should reduce noise
        assert noise_level_denoised < noise_level_original

    def test_data_compression_reconstruction(self):
        """Test data compression and reconstruction workflow."""
        np.random.seed(42)
        X = np.random.randn(100, 50)

        # Compress to lower dimensions
        n_compressed = 10
        pca = PCAEngine(n_components=n_compressed)
        pca.fit(X)
        X_compressed = pca.transform(X)

        # Reconstruct from compressed representation
        X_reconstructed = np.dot(X_compressed, pca.components.T) + pca.mean

        # Check reconstruction quality
        assert X_reconstructed.shape == X.shape

        # Calculate reconstruction error
        reconstruction_error = np.mean((X - X_reconstructed) ** 2)

        # Error should be reasonable (not too large)
        original_variance = np.mean(np.var(X - pca.mean, axis=0))
        assert reconstruction_error < original_variance


class TestEdgeCasesIntegration:
    """Test edge cases in integrated workflows."""

    def test_single_sample_transform(self):
        """Test transforming a single sample after training."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)

        pca = PCAEngine(n_components=5)
        pca.fit(X_train)

        # Transform single sample
        single_sample = np.random.randn(1, 10)
        result = pca.transform(single_sample)

        assert result.shape == (1, 5)

    def test_transform_same_shape_as_training(self):
        """Test transforming data with same shape as training."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        X_new = np.random.randn(100, 10)

        pca = PCAEngine(n_components=5)
        pca.fit(X_train)

        result = pca.transform(X_new)
        assert result.shape == (100, 5)

    def test_transform_different_samples_same_features(self):
        """Test transforming different number of samples."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)

        pca = PCAEngine(n_components=5)
        pca.fit(X_train)

        # Transform different sizes
        for n_samples in [1, 10, 50, 200]:
            X_test = np.random.randn(n_samples, 10)
            result = pca.transform(X_test)
            assert result.shape == (n_samples, 5)

    def test_full_workflow_with_explained_variance(self):
        """Test complete workflow with variance analysis."""
        np.random.seed(42)
        X = np.random.randn(100, 20)

        # Fit PCA
        pca = PCAEngine(n_components=10)
        pca.fit(X)

        # Transform
        X_transformed = pca.transform(X)

        # Calculate explained variance for each component
        explained_variances = np.var(X_transformed, axis=0)

        # Should be in descending order
        assert np.all(explained_variances[:-1] >= explained_variances[1:])

        # Total explained variance should be less than original
        original_total_var = np.sum(np.var(X - pca.mean, axis=0))
        explained_total_var = np.sum(explained_variances)

        assert explained_total_var <= original_total_var

    def test_chained_pca_operations(self):
        """Test chaining multiple PCA operations."""
        np.random.seed(42)
        X = np.random.randn(100, 30)

        # First PCA
        pca1 = PCAEngine(n_components=20)
        pca1.fit(X)
        X1 = pca1.transform(X)

        # Second PCA on transformed data
        pca2 = PCAEngine(n_components=10)
        pca2.fit(X1)
        X2 = pca2.transform(X1)

        # Third PCA
        pca3 = PCAEngine(n_components=5)
        pca3.fit(X2)
        X3 = pca3.transform(X2)

        # Final result
        assert X3.shape == (100, 5)

        # Verify no NaN or Inf values
        assert not np.any(np.isnan(X3))
        assert not np.any(np.isinf(X3))
