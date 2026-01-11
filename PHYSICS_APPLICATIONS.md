# PCA Engine: Physics Applications Guide

This document explains potential physics applications for the PCA Engine and how to test them.

## Overview

Principal Component Analysis (PCA) is a powerful dimensionality reduction technique widely used in physics to:
- Extract meaningful features from high-dimensional data
- Identify correlations and patterns
- Filter noise from experimental measurements
- Reduce computational complexity
- Visualize complex physical systems

## Physics Domains & Applications

### 1. Particle Physics

#### 1.1 Multi-Detector Signal Extraction
**Problem**: Particle detectors generate high-dimensional data (e.g., 30+ channels). The true particle parameters (energy, angle, position) are only 3-4 dimensions.

**PCA Solution**:
- Reduces detector signals to principal components
- Extracts underlying particle parameters
- Filters detector noise and electronic artifacts

**Test Case**: `test_multi_detector_signal_extraction()`
```python
# Simulates 500 events, 30 detectors → 3 principal components
# Verifies >80% variance capture with 3 components
```

**Real-world analogs**:
- CERN Large Hadron Collider calorimeter data
- Neutrino detector arrays
- Cosmic ray observatories

#### 1.2 Signal-Background Separation
**Problem**: Distinguish rare signal events from background noise in particle collision data.

**PCA Solution**:
- Signal events have correlated features (physics-driven)
- Background events are more random
- PCA separates these in principal component space

**Test Case**: `test_signal_background_separation()`
```python
# 200 signal events (correlated features)
# 300 background events (random features)
# PCA distinguishes by variance structure
```

**Real-world analogs**:
- Higgs boson discovery at LHC
- Dark matter detection experiments
- Rare decay search

---

### 2. Quantum Mechanics

#### 2.1 Spectroscopy Line Identification
**Problem**: Identify atomic/molecular transitions from spectral data with multiple overlapping lines.

**PCA Solution**:
- Principal components reveal dominant spectral features
- Component peaks correspond to spectral line positions
- Separates overlapping signals

**Test Case**: `test_spectroscopy_line_identification()`
```python
# 100 spectra with 3 dominant lines at wavelengths [50, 100, 150]
# PCA components show peaks near line positions
```

**Real-world analogs**:
- Atomic emission spectroscopy
- Molecular vibrational spectra (IR, Raman)
- Exoplanet atmospheric composition

#### 2.2 Quantum State Dimensionality Reduction
**Problem**: Quantum systems live in high-dimensional Hilbert spaces but measurements collapse to classical observables.

**PCA Solution**:
- Extracts low-rank structure from measurement data
- Reveals effective dimensionality of quantum state
- Identifies entanglement patterns

**Test Case**: `test_quantum_state_dimensionality_reduction()`
```python
# 500 measurements of 50 observables
# True quantum state: 5 dimensions
# PCA recovers low-dimensional structure
```

**Real-world analogs**:
- Quantum tomography
- Quantum computing benchmarking
- Many-body quantum systems

---

### 3. Thermodynamics & Statistical Physics

#### 3.1 Phase Transition Detection
**Problem**: Identify phase transitions (e.g., ferromagnetic, liquid-gas) from simulation or experimental data.

**PCA Solution**:
- Order parameter emerges as principal component
- Distinguishes ordered vs. disordered phases
- Reveals critical behavior

**Test Case**: `test_phase_transition_detection()`
```python
# Ising-like model: high-T (disordered) vs. low-T (ordered)
# First PC distinguishes phases
```

**Real-world analogs**:
- Ising model ferromagnetic transition
- Liquid-gas phase separation
- Superconducting transitions

#### 3.2 Molecular Dynamics Mode Analysis
**Problem**: Extract dominant vibrational modes from molecular simulation trajectories.

**PCA Solution**:
- Principal components = normal modes
- Identifies collective motions
- Reduces 3N coordinates to few relevant modes

**Test Case**: `test_molecular_dynamics_mode_analysis()`
```python
# 30 atoms × 3 coords = 90 dimensions
# Extract 5 dominant vibrational modes
```

**Real-world analogs**:
- Protein folding dynamics
- Crystal lattice vibrations (phonons)
- Chemical reaction pathways

---

### 4. Experimental Physics

#### 4.1 Multi-Sensor Data Fusion
**Problem**: Combine measurements from multiple sensors with different sensitivities and calibrations.

**PCA Solution**:
- Fuses redundant sensor information
- Extracts true underlying signal
- Reduces sensor-specific noise

**Test Case**: `test_sensor_fusion_multi_instrument()`
```python
# 25 sensors measuring 2D physical quantity
# PCA extracts 2D signal from noisy sensor data
```

**Real-world analogs**:
- Multi-detector telescope arrays
- Magnetometer arrays
- Seismic sensor networks

#### 4.2 Noise Reduction & Filtering
**Problem**: Experimental data contains high-frequency noise obscuring true signal.

**PCA Solution**:
- Low-rank signal structure captured by top components
- Noise distributed across all components
- Reconstruction from top PCs filters noise

**Test Case**: `test_noise_reduction_experimental_data()`
```python
# Signal: 3 components + high-frequency noise
# PCA reconstruction improves SNR by >1.2×
```

**Real-world analogs**:
- Oscilloscope trace denoising
- Voltage/current measurements
- Time-series sensor data

#### 4.3 Systematic Error Detection
**Problem**: Identify calibration drifts and systematic errors across measurement channels.

**PCA Solution**:
- Reveals correlated errors across channels
- Identifies outlier channels
- Guides calibration procedures

**Test Case**: `test_calibration_systematic_error_detection()`
```python
# 20 channels, some with offset/gain errors
# PCA components reveal systematic patterns
```

**Real-world analogs**:
- Multi-channel ADC calibration
- Detector response uniformity
- Sensor array calibration

---

### 5. Astrophysics

#### 5.1 Stellar Spectroscopy Classification
**Problem**: Classify stars by type (O, B, A, F, G, K, M) from spectral data.

**PCA Solution**:
- Principal components capture spectral features
- Different stellar types cluster in PC space
- Automated classification pipeline

**Test Case**: `test_stellar_spectra_classification()`
```python
# 3 stellar types with different spectral lines
# PCA separates types in component space
```

**Real-world analogs**:
- Sloan Digital Sky Survey (SDSS)
- Gaia stellar catalog
- Exoplanet host star characterization

#### 5.2 Galaxy Morphology Analysis
**Problem**: Classify galaxy shapes (elliptical, spiral, irregular) from imaging data.

**PCA Solution**:
- Extracts morphological features from pixel data
- Reduces image to compact representation
- Distinguishes structural patterns

**Test Case**: `test_galaxy_morphology_features()`
```python
# 150 galaxy images (20×20 pixels)
# Elliptical vs. spiral galaxies
# PCA extracts morphological features
```

**Real-world analogs**:
- Galaxy Zoo classification
- Hubble Space Telescope surveys
- Large Synoptic Survey Telescope (LSST)

---

## Running Physics Tests

### Run All Physics Tests
```bash
python -m pytest tests/test_physics.py -v
```

### Run Specific Physics Domain
```bash
# Particle physics tests
pytest tests/test_physics.py::TestParticlePhysics -v

# Quantum mechanics tests
pytest tests/test_physics.py::TestQuantumMechanics -v

# Thermodynamics tests
pytest tests/test_physics.py::TestThermodynamics -v

# Experimental physics tests
pytest tests/test_physics.py::TestExperimentalPhysics -v

# Astrophysics tests
pytest tests/test_physics.py::TestAstrophysics -v
```

### Run Only Physics-Marked Tests
```bash
pytest -v -m physics
```

---

## Extending with Real Data

To test PCA on real physics data:

### 1. Particle Physics Data
```python
# Example: CERN Open Data
from src.pca import PCAEngine
import numpy as np

# Load real detector data (replace with actual data)
detector_data = np.load('data/particle_events.npy')  # Shape: (n_events, n_channels)

pca = PCAEngine(n_components=10)
pca.fit(detector_data)
reduced_events = pca.transform(detector_data)

# Analyze variance explained
variances = np.var(reduced_events, axis=0)
print(f"Top 3 components explain: {np.sum(variances[:3])/np.sum(variances)*100:.1f}%")
```

### 2. Spectroscopy Data
```python
# Example: Astronomical spectra
spectra = np.load('data/stellar_spectra.npy')  # Shape: (n_stars, n_wavelengths)

pca = PCAEngine(n_components=5)
pca.fit(spectra)

# Visualize first component (mean spectrum)
import matplotlib.pyplot as plt
plt.plot(pca.components[:, 0])
plt.xlabel('Wavelength')
plt.ylabel('First PC Loading')
plt.title('Dominant Spectral Feature')
```

### 3. Molecular Dynamics
```python
# Example: MD trajectory
trajectory = np.load('data/md_trajectory.npy')  # Shape: (n_frames, n_atoms*3)

pca = PCAEngine(n_components=10)
pca.fit(trajectory)

# Extract dominant motion modes
principal_modes = pca.transform(trajectory)

# First mode often = largest collective motion
plt.plot(principal_modes[:, 0])
plt.xlabel('Time Frame')
plt.ylabel('First PC Amplitude')
plt.title('Dominant Vibrational Mode')
```

---

## Theoretical Background

### Why PCA Works for Physics

1. **Physical systems often have low intrinsic dimensionality**
   - Many measured variables are correlated
   - True degrees of freedom << measured dimensions
   - Example: 30 detectors measuring 3 particle parameters

2. **Noise is typically uncorrelated**
   - Signal: low-rank, structured
   - Noise: high-rank, random
   - PCA separates signal from noise

3. **Physical laws create correlations**
   - Conservation laws link variables
   - Symmetries reduce effective dimensions
   - Response functions create linear relationships

### Mathematical Properties

For physics applications, PCA provides:

1. **Orthogonal basis**: Components are uncorrelated
   - Useful for independent variable analysis
   - Simplifies statistical models

2. **Variance ordering**: Most important features first
   - Natural feature ranking
   - Guides model selection

3. **Reversibility**: Can reconstruct original data
   - Compression-decompression cycle
   - Quantifies information loss

---

## Best Practices for Physics Applications

### 1. Data Preprocessing
```python
# Standardize if variables have different units
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_standardized = (X - X_mean) / X_std

# Then apply PCA
pca = PCAEngine(n_components=k)
pca.fit(X_standardized)
```

### 2. Choosing Number of Components
```python
# Scree plot: variance vs. component number
pca_full = PCAEngine(n_components=min(X.shape))
pca_full.fit(X)
X_transformed = pca_full.transform(X)

variances = np.var(X_transformed, axis=0)
plt.plot(np.cumsum(variances) / np.sum(variances))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%')
```

### 3. Physical Interpretation
```python
# Examine component loadings for physical meaning
for i in range(pca.n_components):
    component = pca.components[:, i]
    top_features = np.argsort(np.abs(component))[-5:]
    print(f"Component {i} dominated by features: {top_features}")
```

### 4. Validation
```python
# Always validate on independent test set
pca.fit(X_train)
X_test_transformed = pca.transform(X_test)

# Check reconstruction error
X_test_reconstructed = np.dot(X_test_transformed, pca.components.T) + pca.mean
mse = np.mean((X_test - X_test_reconstructed) ** 2)
print(f"Test set reconstruction error: {mse}")
```

---

## Literature & References

### Key Papers on PCA in Physics

1. **Particle Physics**
   - Dutta et al., "Neural Networks for Signal/Background Classification" (2019)
   - Uses PCA preprocessing for particle identification

2. **Quantum Mechanics**
   - McGibbon et al., "Variational Cross-Validation of Slow Dynamical Modes" (2017)
   - PCA for quantum molecular dynamics

3. **Statistical Physics**
   - Wetzel, "Unsupervised Learning and Phase Transitions" (2017)
   - PCA for identifying phase transitions in Ising model

4. **Astrophysics**
   - Yip et al., "Spectral Classification of Quasars in SDSS" (2004)
   - PCA-based stellar/quasar classification

5. **Experimental Physics**
   - Joliffe, "Principal Component Analysis in Meteorology and Oceanography" (1990)
   - Classic review of PCA in physical sciences

### Books
- Shlens, J. "A Tutorial on Principal Component Analysis" (arXiv:1404.1100)
- Jolliffe, I.T. "Principal Component Analysis" (Springer, 2002)

---

## Contributing Physics Examples

To add new physics test cases:

1. Identify the physical problem
2. Create synthetic data mimicking the problem
3. Apply PCA and verify expected behavior
4. Add test to `tests/test_physics.py`
5. Document in this guide

Example template:
```python
@pytest.mark.physics
def test_new_physics_application(self):
    """Test PCA for [specific physics problem]."""
    np.random.seed(42)

    # Generate synthetic physics data
    # ... (problem-specific data generation)

    # Apply PCA
    pca = PCAEngine(n_components=k)
    pca.fit(data)
    transformed = pca.transform(data)

    # Verify physics-motivated expectation
    assert [some physical property holds]
```

---

## Conclusion

PCA is a versatile tool for physics data analysis. The key is understanding:
1. What physical structure exists in your data
2. How PCA can extract that structure
3. How to interpret components physically

The test suite in `test_physics.py` provides concrete examples across major physics domains. Use these as templates for your own applications.

For questions or suggestions, please open an issue on GitHub.
