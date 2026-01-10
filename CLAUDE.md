# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PCA (Principal Component Analysis) engine implementation from scratch using NumPy. The project implements dimensionality reduction without relying on scikit-learn or other ML libraries.

## Commands

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Run all tests
python -m pytest tests/

# Run tests with verbose output
python -m pytest tests/ -v

# Run a specific test class
python -m pytest tests/test_pca.py::TestPCAEngineFit -v

# Run a specific test
python -m pytest tests/test_pca.py::TestPCAEngineFit::test_fit_sets_mean -v
```

## Architecture

The codebase follows a simple structure:
- `src/pca.py` - Contains the `PCAEngine` class implementing PCA from scratch
- `src/__init__.py` - Module exports
- `tests/` - Test files for the PCA implementation
- `tests/conftest.py` - Shared pytest fixtures
- `tests/test_pca.py` - Unit tests for PCAEngine
- `data/` - Directory for datasets (currently empty)
- `pyproject.toml` - Project configuration and pytest settings

### PCAEngine Class (src/pca.py)

The `PCAEngine` class implements the four core steps of PCA:
1. **Mean centering** - Subtracts the mean from each feature
2. **Covariance matrix calculation** - Computes feature covariance
3. **Eigen decomposition** - Finds eigenvalues and eigenvectors using `np.linalg.eigh`
4. **Component selection** - Sorts and selects top n principal components

Key methods:
- `fit(X)` - Learns the principal components from training data
- `transform(X)` - Projects data onto the learned principal components

## Dependencies

- NumPy (for matrix operations and linear algebra)
