import numpy as np
import pytest

from pca import PCAEngine


class TestPCAEngineInit:
    """Tests for PCAEngine initialization."""

    def test_init_n_components(self):
        """Test that n_components is set correctly."""
        pca = PCAEngine(n_components=3)
        assert pca.n_components == 3

    def test_init_components_none(self):
        """Test that components is None before fit."""
        pca = PCAEngine(n_components=2)
        assert pca.components is None

    def test_init_mean_none(self):
        """Test that mean is None before fit."""
        pca = PCAEngine(n_components=2)
        assert pca.mean is None


class TestPCAEngineFit:
    """Tests for PCAEngine fit method."""

    def test_fit_sets_mean(self, simple_data, pca_engine):
        """Test that fit calculates and stores the mean."""
        pca_engine.fit(simple_data)
        expected_mean = np.mean(simple_data, axis=0)
        np.testing.assert_array_almost_equal(pca_engine.mean, expected_mean)

    def test_fit_sets_components(self, simple_data, pca_engine):
        """Test that fit calculates and stores the components."""
        pca_engine.fit(simple_data)
        assert pca_engine.components is not None
        assert pca_engine.components.shape == (2, 2)

    def test_fit_components_orthogonal(self, simple_data, pca_engine):
        """Test that principal components are orthogonal."""
        pca_engine.fit(simple_data)
        # Dot product of orthogonal vectors should be close to 0
        dot_product = np.dot(
            pca_engine.components[:, 0],
            pca_engine.components[:, 1]
        )
        np.testing.assert_almost_equal(dot_product, 0, decimal=10)

    def test_fit_components_unit_length(self, simple_data, pca_engine):
        """Test that principal components have unit length."""
        pca_engine.fit(simple_data)
        for i in range(pca_engine.n_components):
            norm = np.linalg.norm(pca_engine.components[:, i])
            np.testing.assert_almost_equal(norm, 1.0, decimal=10)

    def test_fit_with_more_features(self, sample_data):
        """Test fit with higher dimensional data."""
        pca = PCAEngine(n_components=3)
        pca.fit(sample_data)
        assert pca.components.shape == (5, 3)


class TestPCAEngineTransform:
    """Tests for PCAEngine transform method."""

    def test_transform_reduces_dimensions(self, simple_data):
        """Test that transform reduces dimensions correctly."""
        pca = PCAEngine(n_components=1)
        pca.fit(simple_data)
        transformed = pca.transform(simple_data)
        assert transformed.shape == (10, 1)

    def test_transform_preserves_samples(self, simple_data, pca_engine):
        """Test that transform preserves number of samples."""
        pca_engine.fit(simple_data)
        transformed = pca_engine.transform(simple_data)
        assert transformed.shape[0] == simple_data.shape[0]

    def test_transform_output_shape(self, sample_data):
        """Test transform output shape matches n_components."""
        pca = PCAEngine(n_components=3)
        pca.fit(sample_data)
        transformed = pca.transform(sample_data)
        assert transformed.shape == (100, 3)

    def test_transform_centered_data(self, simple_data, pca_engine):
        """Test that transformed data is centered around origin."""
        pca_engine.fit(simple_data)
        transformed = pca_engine.transform(simple_data)
        mean_transformed = np.mean(transformed, axis=0)
        np.testing.assert_array_almost_equal(
            mean_transformed,
            np.zeros(pca_engine.n_components),
            decimal=10
        )


class TestPCAEngineIntegration:
    """Integration tests for PCAEngine."""

    def test_fit_transform_workflow(self, sample_data):
        """Test complete fit-transform workflow."""
        pca = PCAEngine(n_components=2)
        pca.fit(sample_data)
        transformed = pca.transform(sample_data)

        assert transformed.shape == (100, 2)
        assert pca.mean is not None
        assert pca.components is not None

    def test_variance_preserved_in_order(self, sample_data):
        """Test that first component captures more variance than second."""
        pca = PCAEngine(n_components=2)
        pca.fit(sample_data)
        transformed = pca.transform(sample_data)

        var_first = np.var(transformed[:, 0])
        var_second = np.var(transformed[:, 1])
        assert var_first >= var_second

    def test_transform_new_data(self, simple_data, pca_engine):
        """Test transforming new data after fitting."""
        pca_engine.fit(simple_data)

        new_data = np.array([[2.0, 2.0], [1.5, 1.5]])
        transformed = pca_engine.transform(new_data)

        assert transformed.shape == (2, 2)
