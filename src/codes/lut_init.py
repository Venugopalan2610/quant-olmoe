"""Initialize the HYB LUT via K-means on 2D iid Gaussian samples.

This produces the Lloyd-Max-optimal placement of 2^Q centroids for an
iid 2D Gaussian source, which is HYB's starting point before fine-tuning.
"""
import numpy as np
from scipy.cluster.vq import kmeans2


def init_hyb_lut(Q=9, n_samples=1_000_000, seed=0):
    """Run K-means on 2D Gaussian samples to initialize the HYB LUT.

    The LUT is then renormalized so that uniform sampling over the 2^Q
    centroids has empirical variance = 1.0 per component. This matters
    because in the trellis, bitshift state -> LUT index produces a
    near-uniform distribution over centroids (NOT a Gaussian-weighted
    distribution), so without renormalization the decoded values have
    variance > 1.

    Returns a (2^Q, 2) float32 array of centroids.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(size=(n_samples, 2)).astype(np.float32)

    centroids, _labels = kmeans2(
        samples,
        k=2 ** Q,
        minit='++',
        seed=seed,
        iter=50,
    )
    centroids = centroids.astype(np.float32)

    # Renormalize so uniform sampling has unit variance per component
    uniform_var = float(np.mean(centroids ** 2))  # mean is ~0 by symmetry
    scale = 1.0 / np.sqrt(uniform_var)
    centroids = centroids * scale

    return centroids

def lut_mse(lut, n_samples=100_000, seed=42):
    """Empirical MSE of the LUT against a 2D Gaussian source.

    For each sample, find nearest centroid (brute force, slow but correct),
    return mean squared distance.
    """
    rng = np.random.default_rng(seed)
    samples = rng.standard_normal(size=(n_samples, 2)).astype(np.float32)

    # Brute-force nearest neighbor — fine for n=100k, k=512
    dists_sq = ((samples[:, None, :] - lut[None, :, :]) ** 2).sum(axis=-1)
    nearest = dists_sq.min(axis=-1)
    return float(nearest.mean())