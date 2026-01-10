import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pca import PCAEngine


@pytest.fixture
def sample_data():
    """Generate sample 2D data for PCA testing."""
    np.random.seed(42)
    # Create correlated data
    X = np.random.randn(100, 5)
    return X


@pytest.fixture
def simple_data():
    """Simple 2D data for basic testing."""
    return np.array([
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ])


@pytest.fixture
def pca_engine():
    """Create a PCAEngine instance with 2 components."""
    return PCAEngine(n_components=2)
