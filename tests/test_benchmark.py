"""
Performance benchmark tests comparing PCAEngine with scikit-learn.

These tests run as part of the pytest suite but are marked with @pytest.mark.benchmark
so they can be run separately with: pytest -v -m benchmark
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.decomposition import PCA as SklearnPCA

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pca import PCAEngine


@pytest.fixture
def small_dataset():
    """Small dataset for quick benchmarks."""
    np.random.seed(42)
    return np.random.randn(100, 10)


@pytest.fixture
def medium_dataset():
    """Medium dataset for realistic benchmarks."""
    np.random.seed(42)
    return np.random.randn(1000, 50)


@pytest.fixture
def large_dataset():
    """Large dataset for stress testing."""
    np.random.seed(42)
    return np.random.randn(5000, 100)


class TestPerformanceComparison:
    """Compare performance with scikit-learn implementation."""

    @pytest.mark.benchmark
    def test_fit_performance_small(self, small_dataset, benchmark):
        """Benchmark fit on small dataset."""
        pca = PCAEngine(n_components=5)
        benchmark(pca.fit, small_dataset)

    @pytest.mark.benchmark
    def test_fit_performance_medium(self, medium_dataset, benchmark):
        """Benchmark fit on medium dataset."""
        pca = PCAEngine(n_components=10)
        benchmark(pca.fit, medium_dataset)

    @pytest.mark.benchmark
    def test_transform_performance_small(self, small_dataset, benchmark):
        """Benchmark transform on small dataset."""
        pca = PCAEngine(n_components=5)
        pca.fit(small_dataset)
        benchmark(pca.transform, small_dataset)

    @pytest.mark.benchmark
    def test_transform_performance_medium(self, medium_dataset, benchmark):
        """Benchmark transform on medium dataset."""
        pca = PCAEngine(n_components=10)
        pca.fit(medium_dataset)
        benchmark(pca.transform, medium_dataset)


class TestAccuracyVsSklearn:
    """Test numerical accuracy compared to scikit-learn."""

    def test_components_match_sklearn_small(self, small_dataset):
        """Test that components match sklearn on small dataset."""
        custom_pca = PCAEngine(n_components=5)
        sklearn_pca = SklearnPCA(n_components=5)

        custom_pca.fit(small_dataset)
        sklearn_pca.fit(small_dataset)

        # Components should match (up to sign flip)
        for i in range(5):
            custom_comp = custom_pca.components[:, i]
            sklearn_comp = sklearn_pca.components_[i, :]

            # Check if they match (same or opposite direction)
            dot_product = np.abs(np.dot(custom_comp, sklearn_comp))
            assert dot_product > 0.999, \
                f"Component {i} does not match sklearn (dot product: {dot_product})"

    def test_transform_matches_sklearn_small(self, small_dataset):
        """Test that transform matches sklearn on small dataset."""
        custom_pca = PCAEngine(n_components=5)
        sklearn_pca = SklearnPCA(n_components=5)

        custom_pca.fit(small_dataset)
        sklearn_pca.fit(small_dataset)

        custom_result = custom_pca.transform(small_dataset)
        sklearn_result = sklearn_pca.transform(small_dataset)

        # Results should be close (accounting for sign flips)
        abs_diff = np.abs(np.abs(custom_result) - np.abs(sklearn_result))
        max_diff = np.max(abs_diff)

        assert max_diff < 1e-10, \
            f"Transform results differ too much from sklearn (max diff: {max_diff})"

    def test_variance_explained_matches_sklearn(self, medium_dataset):
        """Test that explained variance matches sklearn."""
        n_components = 10
        custom_pca = PCAEngine(n_components=n_components)
        sklearn_pca = SklearnPCA(n_components=n_components)

        custom_pca.fit(medium_dataset)
        sklearn_pca.fit(medium_dataset)

        # Calculate variance explained for custom implementation
        custom_transformed = custom_pca.transform(medium_dataset)
        custom_var = np.var(custom_transformed, axis=0)

        # Get variance from sklearn
        sklearn_var = sklearn_pca.explained_variance_

        # Compare variances (they should be very close)
        np.testing.assert_allclose(
            custom_var, sklearn_var, rtol=1e-10,
            err_msg="Explained variance does not match sklearn"
        )

    @pytest.mark.benchmark
    def test_mean_matches_sklearn(self, small_dataset):
        """Test that mean calculation matches sklearn."""
        custom_pca = PCAEngine(n_components=5)
        sklearn_pca = SklearnPCA(n_components=5)

        custom_pca.fit(small_dataset)
        sklearn_pca.fit(small_dataset)

        np.testing.assert_allclose(
            custom_pca.mean, sklearn_pca.mean_, rtol=1e-10,
            err_msg="Mean calculation does not match sklearn"
        )


class TestScalability:
    """Test performance scalability with data size."""

    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_samples,n_features,n_components", [
        (100, 10, 5),
        (500, 20, 10),
        (1000, 50, 15),
        (2000, 100, 20),
    ])
    def test_scalability_fit(self, n_samples, n_features, n_components):
        """Test fit scalability with increasing dataset size."""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)

        pca = PCAEngine(n_components=n_components)
        pca.fit(X)

        # Just verify it completes successfully
        assert pca.components is not None
        assert pca.components.shape == (n_features, n_components)

    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_components", [1, 5, 10, 20])
    def test_scalability_components(self, medium_dataset, n_components):
        """Test scalability with different numbers of components."""
        pca = PCAEngine(n_components=n_components)
        pca.fit(medium_dataset)

        assert pca.components.shape[1] == n_components


class TestMemoryEfficiency:
    """Test memory usage patterns."""

    @pytest.mark.benchmark
    def test_no_memory_leaks_repeated_fits(self, small_dataset):
        """Test that repeated fits don't cause memory issues."""
        pca = PCAEngine(n_components=5)

        # Fit multiple times
        for _ in range(100):
            pca.fit(small_dataset)

        # Should complete without memory errors
        assert pca.components is not None

    @pytest.mark.benchmark
    def test_transform_multiple_batches(self, medium_dataset):
        """Test transforming multiple batches efficiently."""
        pca = PCAEngine(n_components=10)
        pca.fit(medium_dataset)

        # Transform multiple small batches
        n_batches = 10
        batch_size = 100
        for i in range(n_batches):
            batch = medium_dataset[i*batch_size:(i+1)*batch_size]
            result = pca.transform(batch)
            assert result.shape == (batch_size, 10)


class TestRobustness:
    """Test robustness of implementation."""

    def test_different_data_distributions(self):
        """Test PCA on different data distributions."""
        np.random.seed(42)
        n_components = 3

        # Uniform distribution
        X_uniform = np.random.uniform(-10, 10, size=(100, 10))
        pca_uniform = PCAEngine(n_components=n_components)
        pca_uniform.fit(X_uniform)
        assert not np.any(np.isnan(pca_uniform.components))

        # Exponential distribution
        X_exp = np.random.exponential(scale=2.0, size=(100, 10))
        pca_exp = PCAEngine(n_components=n_components)
        pca_exp.fit(X_exp)
        assert not np.any(np.isnan(pca_exp.components))

        # Mixed distribution
        X_mixed = np.column_stack([
            np.random.randn(100, 5),
            np.random.uniform(-5, 5, size=(100, 5))
        ])
        pca_mixed = PCAEngine(n_components=n_components)
        pca_mixed.fit(X_mixed)
        assert not np.any(np.isnan(pca_mixed.components))

    def test_correlated_features(self):
        """Test PCA on highly correlated features."""
        np.random.seed(42)
        n_samples = 100

        # Create correlated features
        base = np.random.randn(n_samples, 1)
        X = np.hstack([
            base,
            base + np.random.randn(n_samples, 1) * 0.1,
            base * 2 + np.random.randn(n_samples, 1) * 0.1,
            np.random.randn(n_samples, 2)
        ])

        pca = PCAEngine(n_components=3)
        pca.fit(X)

        # First component should capture most variance
        X_transformed = pca.transform(X)
        variances = np.var(X_transformed, axis=0)
        assert variances[0] > variances[1] * 2, \
            "First component should dominate for correlated data"

    def test_sparse_like_data(self):
        """Test PCA on data with many zeros."""
        np.random.seed(42)
        X = np.random.randn(100, 20)
        # Make 70% of values zero
        mask = np.random.random((100, 20)) > 0.3
        X = X * mask

        pca = PCAEngine(n_components=5)
        pca.fit(X)

        assert not np.any(np.isnan(pca.components))
        assert not np.any(np.isinf(pca.components))
