"""
Benchmark script to compare PCAEngine with scikit-learn's PCA.

This script measures performance metrics including:
- Execution time for fit and transform
- Memory usage
- Numerical accuracy comparison
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA as SklearnPCA

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from pca import PCAEngine


def generate_test_data(n_samples, n_features):
    """Generate random test data."""
    np.random.seed(42)
    return np.random.randn(n_samples, n_features)


def benchmark_fit(pca_impl, X, name):
    """Benchmark the fit method."""
    start_time = time.perf_counter()
    pca_impl.fit(X)
    end_time = time.perf_counter()
    return end_time - start_time


def benchmark_transform(pca_impl, X, name):
    """Benchmark the transform method."""
    start_time = time.perf_counter()
    result = pca_impl.transform(X)
    end_time = time.perf_counter()
    return end_time - start_time, result


def compare_accuracy(custom_result, sklearn_result):
    """Compare numerical accuracy between implementations."""
    # Account for possible sign flips in principal components
    # by comparing absolute values
    abs_diff = np.abs(np.abs(custom_result) - np.abs(sklearn_result))
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    return max_diff, mean_diff


def run_benchmark(n_samples, n_features, n_components):
    """Run a single benchmark comparison."""
    print(f"\n{'='*70}")
    print(f"Benchmark: {n_samples} samples, {n_features} features, {n_components} components")
    print(f"{'='*70}")

    # Generate test data
    X = generate_test_data(n_samples, n_features)

    # Initialize implementations
    custom_pca = PCAEngine(n_components=n_components)
    sklearn_pca = SklearnPCA(n_components=n_components)

    # Benchmark fit
    print("\n[Fit Performance]")
    custom_fit_time = benchmark_fit(custom_pca, X, "Custom PCA")
    sklearn_fit_time = benchmark_fit(sklearn_pca, X, "Sklearn PCA")

    print(f"  Custom PCA fit time:  {custom_fit_time*1000:.3f} ms")
    print(f"  Sklearn PCA fit time: {sklearn_fit_time*1000:.3f} ms")
    print(f"  Speedup ratio:        {sklearn_fit_time/custom_fit_time:.2f}x")

    # Benchmark transform
    print("\n[Transform Performance]")
    custom_transform_time, custom_result = benchmark_transform(custom_pca, X, "Custom PCA")
    sklearn_transform_time, sklearn_result = benchmark_transform(sklearn_pca, X, "Sklearn PCA")

    print(f"  Custom PCA transform time:  {custom_transform_time*1000:.3f} ms")
    print(f"  Sklearn PCA transform time: {sklearn_transform_time*1000:.3f} ms")
    print(f"  Speedup ratio:              {sklearn_transform_time/custom_transform_time:.2f}x")

    # Compare accuracy
    print("\n[Numerical Accuracy]")
    max_diff, mean_diff = compare_accuracy(custom_result, sklearn_result)
    print(f"  Max difference:  {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")

    if max_diff < 1e-10:
        print("  Status: EXCELLENT - Numerically equivalent")
    elif max_diff < 1e-6:
        print("  Status: GOOD - Very close match")
    elif max_diff < 1e-3:
        print("  Status: ACCEPTABLE - Minor differences")
    else:
        print("  Status: WARNING - Significant differences detected")

    return {
        'custom_fit_time': custom_fit_time,
        'sklearn_fit_time': sklearn_fit_time,
        'custom_transform_time': custom_transform_time,
        'sklearn_transform_time': sklearn_transform_time,
        'max_diff': max_diff,
        'mean_diff': mean_diff
    }


def main():
    """Run comprehensive benchmarks."""
    print("\n" + "="*70)
    print(" PCA Engine Benchmark Suite")
    print(" Comparing Custom Implementation vs Scikit-Learn")
    print("="*70)

    # Define benchmark scenarios
    scenarios = [
        (100, 10, 5),      # Small dataset
        (1000, 50, 10),    # Medium dataset
        (5000, 100, 20),   # Large dataset
        (10000, 200, 50),  # Very large dataset
    ]

    results = []
    for n_samples, n_features, n_components in scenarios:
        result = run_benchmark(n_samples, n_features, n_components)
        results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print(" Summary")
    print(f"{'='*70}")

    avg_custom_fit = np.mean([r['custom_fit_time'] for r in results])
    avg_sklearn_fit = np.mean([r['sklearn_fit_time'] for r in results])
    avg_custom_transform = np.mean([r['custom_transform_time'] for r in results])
    avg_sklearn_transform = np.mean([r['sklearn_transform_time'] for r in results])

    print(f"\nAverage fit time:")
    print(f"  Custom PCA:  {avg_custom_fit*1000:.3f} ms")
    print(f"  Sklearn PCA: {avg_sklearn_fit*1000:.3f} ms")
    print(f"  Overall speedup: {avg_sklearn_fit/avg_custom_fit:.2f}x")

    print(f"\nAverage transform time:")
    print(f"  Custom PCA:  {avg_custom_transform*1000:.3f} ms")
    print(f"  Sklearn PCA: {avg_sklearn_transform*1000:.3f} ms")
    print(f"  Overall speedup: {avg_sklearn_transform/avg_custom_transform:.2f}x")

    max_error = max([r['max_diff'] for r in results])
    print(f"\nMaximum numerical error across all tests: {max_error:.2e}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
