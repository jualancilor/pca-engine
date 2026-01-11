"""
Physics-specific test cases for PCAEngine.

Tests demonstrate PCA applications in various physics domains:
- Particle physics (multi-detector analysis)
- Quantum mechanics (spectroscopy)
- Thermodynamics (phase transitions)
- Experimental physics (sensor fusion)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pca import PCAEngine


@pytest.mark.physics
class TestParticlePhysics:
    """Test PCA applications in particle physics."""

    def test_multi_detector_signal_extraction(self):
        """Simulate extracting particle track from multi-detector data."""
        np.random.seed(42)

        # Simulate particle detector array
        n_events = 500
        n_detectors = 30

        # True particle parameters: energy, angle, position
        true_params = np.random.randn(n_events, 3)

        # Detector response matrix (how each param affects each detector)
        response_matrix = np.random.randn(3, n_detectors)

        # Generate detector signals
        detector_signals = np.dot(true_params, response_matrix)

        # Add measurement noise
        noise_level = 0.3
        detector_signals += np.random.randn(n_events, n_detectors) * noise_level

        # Apply PCA to extract principal components
        pca = PCAEngine(n_components=3)
        pca.fit(detector_signals)
        extracted_params = pca.transform(detector_signals)

        # Verify dimensionality reduction
        assert extracted_params.shape == (n_events, 3)

        # Verify first component captures most variance
        variances = np.var(extracted_params, axis=0)
        assert variances[0] > variances[1] > variances[2]

        # Total variance should be significant
        total_original_var = np.sum(np.var(detector_signals - pca.mean, axis=0))
        total_extracted_var = np.sum(variances)
        variance_ratio = total_extracted_var / total_original_var

        # Should capture >80% of variance with 3 components
        assert variance_ratio > 0.8, f"Only captured {variance_ratio*100:.1f}% variance"

    def test_signal_background_separation(self):
        """Test PCA for separating signal from background events."""
        np.random.seed(42)

        n_signal = 200
        n_background = 300
        n_features = 20

        # Signal events: correlated features (e.g., energy deposits)
        signal_base = np.random.randn(n_signal, 3)
        signal_features = np.dot(signal_base, np.random.randn(3, n_features))
        signal_features += np.random.randn(n_signal, n_features) * 0.2

        # Background events: uncorrelated random features
        background_features = np.random.randn(n_background, n_features)

        # Combine
        all_features = np.vstack([signal_features, background_features])

        # Apply PCA
        pca = PCAEngine(n_components=3)
        pca.fit(all_features)
        transformed = pca.transform(all_features)

        # Signal events should cluster differently than background
        signal_transformed = transformed[:n_signal]
        background_transformed = transformed[n_signal:]

        # Signal should have higher variance in first component
        signal_var = np.var(signal_transformed[:, 0])
        background_var = np.var(background_transformed[:, 0])

        assert signal_var > background_var * 0.5  # Signal more structured


@pytest.mark.physics
class TestQuantumMechanics:
    """Test PCA applications in quantum mechanics."""

    def test_spectroscopy_line_identification(self):
        """Simulate identifying spectral lines from multiple spectra."""
        np.random.seed(42)

        n_spectra = 100
        n_wavelengths = 200

        # Create spectral lines at specific wavelengths
        # Three dominant lines (representing atomic transitions)
        line_positions = [50, 100, 150]
        line_widths = [3, 4, 3]

        spectra = np.zeros((n_spectra, n_wavelengths))

        for i in range(n_spectra):
            # Each spectrum has varying line intensities
            intensities = np.abs(np.random.randn(3)) + 1.0

            for line_pos, width, intensity in zip(line_positions, line_widths, intensities):
                # Add Gaussian line profile
                x = np.arange(n_wavelengths)
                line = intensity * np.exp(-((x - line_pos) ** 2) / (2 * width ** 2))
                spectra[i] += line

            # Add noise
            spectra[i] += np.random.randn(n_wavelengths) * 0.1

        # Apply PCA
        pca = PCAEngine(n_components=3)
        pca.fit(spectra)

        # First 3 components should capture line variations
        # Components should have peaks near line positions
        for comp_idx in range(3):
            component = np.abs(pca.components[:, comp_idx])

            # Find peaks in component
            # Should correspond to spectral line positions
            max_positions = np.argsort(component)[-5:]  # Top 5 positions

            # At least one peak should be near a line position
            near_line = False
            for max_pos in max_positions:
                for line_pos in line_positions:
                    if abs(max_pos - line_pos) < 10:
                        near_line = True
                        break

            assert near_line, f"Component {comp_idx} doesn't capture spectral lines"

    def test_quantum_state_dimensionality_reduction(self):
        """Test reducing dimensionality of quantum state representations."""
        np.random.seed(42)

        # Simulate quantum state measurements
        # Hilbert space dimension reduced to classical measurement space
        n_measurements = 500
        n_observables = 50

        # True quantum state has low-rank structure
        quantum_dim = 5
        state_amplitudes = np.random.randn(n_measurements, quantum_dim)

        # Observable measurements (Born rule: |<ψ|φ>|²)
        observable_basis = np.random.randn(quantum_dim, n_observables)
        measurements = np.dot(state_amplitudes, observable_basis)

        # Add quantum/measurement noise
        measurements += np.random.randn(n_measurements, n_observables) * 0.5

        # Apply PCA
        pca = PCAEngine(n_components=quantum_dim)
        pca.fit(measurements)
        reduced_state = pca.transform(measurements)

        # Should recover low-dimensional structure
        assert reduced_state.shape == (n_measurements, quantum_dim)

        # Check variance preservation
        variances = np.var(reduced_state, axis=0)
        assert variances[0] > variances[-1] * 2  # Clear hierarchy


@pytest.mark.physics
class TestThermodynamics:
    """Test PCA applications in thermodynamics and statistical physics."""

    def test_phase_transition_detection(self):
        """Simulate detecting phase transition using PCA."""
        np.random.seed(42)

        # Simulate Ising-like model with phase transition
        n_temps = 100  # Temperature steps
        n_configs = 50  # Configurations per temperature

        # High temperature (disordered phase)
        high_T_spins = np.random.choice([-1, 1], size=(n_configs, n_temps // 2))

        # Low temperature (ordered phase) - more correlated
        low_T_base = np.random.choice([-1, 1], size=(n_configs, 1))
        low_T_spins = np.repeat(low_T_base, n_temps // 2, axis=1)
        # Add small fluctuations
        low_T_spins = low_T_spins * (1 + np.random.randn(n_configs, n_temps // 2) * 0.3)

        # Combine
        all_configs = np.hstack([high_T_spins, low_T_spins])

        # Apply PCA
        pca = PCAEngine(n_components=5)
        pca.fit(all_configs)
        transformed = pca.transform(all_configs)

        # First component should distinguish ordered vs disordered
        # Low-T (ordered) should have different PC1 distribution
        assert not np.allclose(
            np.mean(transformed[:, 0]),
            0.0,
            atol=0.1
        )

    def test_molecular_dynamics_mode_analysis(self):
        """Test extracting vibrational modes from molecular trajectories."""
        np.random.seed(42)

        # Simulate molecular system with few dominant vibrational modes
        n_timesteps = 200
        n_atoms = 30
        n_coords = n_atoms * 3  # x, y, z for each atom

        # True vibrational modes (normal modes)
        n_modes = 5
        mode_amplitudes = np.random.randn(n_timesteps, n_modes)
        mode_vectors = np.random.randn(n_modes, n_coords)

        # Generate trajectories
        trajectory = np.dot(mode_amplitudes, mode_vectors)

        # Add thermal noise
        trajectory += np.random.randn(n_timesteps, n_coords) * 0.3

        # Apply PCA to extract dominant modes
        pca = PCAEngine(n_components=n_modes)
        pca.fit(trajectory)
        extracted_modes = pca.transform(trajectory)

        # Verify extraction
        assert extracted_modes.shape == (n_timesteps, n_modes)

        # Modes should be ordered by variance (frequency importance)
        variances = np.var(extracted_modes, axis=0)
        assert np.all(variances[:-1] >= variances[1:])


@pytest.mark.physics
class TestExperimentalPhysics:
    """Test PCA applications in experimental physics."""

    def test_sensor_fusion_multi_instrument(self):
        """Test combining data from multiple experimental sensors."""
        np.random.seed(42)

        # Simulate multi-sensor experiment
        n_measurements = 300
        n_sensors = 25

        # True physical quantity being measured (e.g., magnetic field)
        true_signal = np.random.randn(n_measurements, 2)  # 2D field

        # Sensor response (different sensitivities and orientations)
        sensor_matrix = np.random.randn(2, n_sensors)

        # Sensor readings
        sensor_data = np.dot(true_signal, sensor_matrix)

        # Add sensor-specific noise
        sensor_noise = np.random.randn(n_measurements, n_sensors) * 0.4

        # Add systematic errors (calibration drift)
        systematic_errors = np.random.randn(1, n_sensors) * 0.2
        sensor_data = sensor_data + sensor_noise + systematic_errors

        # Apply PCA for sensor fusion
        pca = PCAEngine(n_components=2)
        pca.fit(sensor_data)
        fused_signal = pca.transform(sensor_data)

        # Should extract 2D signal structure
        assert fused_signal.shape == (n_measurements, 2)

        # Check that fused signal captures main variations
        correlation_with_true = np.corrcoef(
            true_signal[:, 0],
            fused_signal[:, 0]
        )[0, 1]

        # Should have reasonable correlation (accounting for noise)
        assert abs(correlation_with_true) > 0.3

    def test_noise_reduction_experimental_data(self):
        """Test using PCA for noise filtering in experimental data."""
        np.random.seed(42)

        n_samples = 200
        n_channels = 40

        # True signal (low-rank structure)
        signal_components = 3
        true_signal = np.dot(
            np.random.randn(n_samples, signal_components),
            np.random.randn(signal_components, n_channels)
        )

        # High-frequency noise
        noise = np.random.randn(n_samples, n_channels) * 1.0

        # Measured data
        measured = true_signal + noise

        # Apply PCA denoising
        pca = PCAEngine(n_components=signal_components)
        pca.fit(measured)
        denoised_components = pca.transform(measured)

        # Reconstruct from principal components
        denoised = np.dot(denoised_components, pca.components.T) + pca.mean

        # Calculate SNR improvement
        noise_original = np.var(measured - true_signal)
        noise_denoised = np.var(denoised - true_signal)

        snr_improvement = noise_original / noise_denoised

        # Should improve SNR
        assert snr_improvement > 1.2, f"SNR improvement: {snr_improvement:.2f}x"

    def test_calibration_systematic_error_detection(self):
        """Test identifying systematic errors across measurement channels."""
        np.random.seed(42)

        n_measurements = 150
        n_channels = 20

        # True values
        true_values = np.random.randn(n_measurements, 3)

        # Channel responses
        channel_response = np.random.randn(3, n_channels)
        measurements = np.dot(true_values, channel_response)

        # Add systematic errors to some channels
        systematic_channels = [5, 10, 15]
        for ch in systematic_channels:
            measurements[:, ch] += 2.0  # Offset
            measurements[:, ch] *= 1.5  # Gain error

        # Add random noise
        measurements += np.random.randn(n_measurements, n_channels) * 0.3

        # Apply PCA
        pca = PCAEngine(n_components=5)
        pca.fit(measurements)

        # Components should reveal channel correlations
        # Channels with systematic errors should have different loadings
        assert pca.components is not None

        # Check that components capture structure
        total_var = np.sum(np.var(measurements - pca.mean, axis=0))
        captured_var = np.sum(np.var(pca.transform(measurements), axis=0))

        assert captured_var / total_var > 0.7


@pytest.mark.physics
class TestAstrophysics:
    """Test PCA applications in astrophysics."""

    def test_stellar_spectra_classification(self):
        """Test classifying stellar types from spectra."""
        np.random.seed(42)

        n_stars = 200
        n_wavelengths = 300

        # Three stellar types with different spectral features
        # Type A: Strong hydrogen lines
        # Type B: Strong helium lines
        # Type C: Strong metal lines

        spectra = []
        for i in range(n_stars):
            star_type = i % 3
            spectrum = np.zeros(n_wavelengths)

            if star_type == 0:  # Type A
                spectrum[100:120] = 5.0 + np.random.randn(20) * 0.5
            elif star_type == 1:  # Type B
                spectrum[150:170] = 4.0 + np.random.randn(20) * 0.5
            else:  # Type C
                spectrum[200:220] = 3.0 + np.random.randn(20) * 0.5

            # Add continuum
            spectrum += 1.0 + np.random.randn(n_wavelengths) * 0.2

            spectra.append(spectrum)

        spectra = np.array(spectra)

        # Apply PCA
        pca = PCAEngine(n_components=3)
        pca.fit(spectra)
        transformed = pca.transform(spectra)

        # Different stellar types should cluster in PC space
        type_a = transformed[0::3]
        type_b = transformed[1::3]
        type_c = transformed[2::3]

        # Check separation (means should differ)
        mean_a = np.mean(type_a[:, 0])
        mean_b = np.mean(type_b[:, 0])
        mean_c = np.mean(type_c[:, 0])

        # At least two types should be well-separated
        separations = [
            abs(mean_a - mean_b),
            abs(mean_b - mean_c),
            abs(mean_a - mean_c)
        ]

        assert max(separations) > 0.5, "Stellar types not separated in PC space"

    def test_galaxy_morphology_features(self):
        """Test extracting morphological features from galaxy images."""
        np.random.seed(42)

        n_galaxies = 150
        image_size = 20
        n_pixels = image_size * image_size

        # Simulate galaxy images (flattened)
        galaxies = []
        for i in range(n_galaxies):
            # Create simple galaxy models
            x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
            center_x, center_y = image_size // 2, image_size // 2

            # Distance from center
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

            # Different morphologies
            if i % 2 == 0:  # Elliptical
                galaxy = np.exp(-(r ** 2) / 20)
            else:  # Spiral (add angular structure)
                theta = np.arctan2(y - center_y, x - center_x)
                galaxy = np.exp(-(r ** 2) / 20) * (1 + 0.5 * np.cos(3 * theta))

            # Add noise
            galaxy += np.random.randn(image_size, image_size) * 0.1

            galaxies.append(galaxy.flatten())

        galaxies = np.array(galaxies)

        # Apply PCA
        pca = PCAEngine(n_components=5)
        pca.fit(galaxies)
        morphology_features = pca.transform(galaxies)

        # Should extract morphological features
        assert morphology_features.shape == (n_galaxies, 5)

        # First component should distinguish elliptical vs spiral
        elliptical_pc1 = morphology_features[0::2, 0]
        spiral_pc1 = morphology_features[1::2, 0]

        mean_diff = abs(np.mean(elliptical_pc1) - np.mean(spiral_pc1))
        assert mean_diff > 0.1, "Failed to distinguish galaxy morphologies"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "physics"])
