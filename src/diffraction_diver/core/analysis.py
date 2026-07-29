"""Unsupervised decomposition of a sliding-FFT stack.

Ported from the notebook's `do_PCA` / `do_ICA` / `do_NMF`, with plotting
stripped out — these return arrays only, for the GUI to render.
"""

import numpy as np
from sklearn import decomposition

# PCA has no natural "right" number of components the way ICA/NMF are given
# one explicitly - the scree plot (singular values) is what the GUI shows to
# help pick it. This is just the initial spinbox value, matching the count
# the original notebook happened to plot.
PCA_DEFAULT_COMPONENTS = 9


def run_pca(fft_mat: np.ndarray, num_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via SVD on the real part of the FFT stack.

    `fft_mat` has shape (size_x, size_y, crop, crop).
    Returns the top `num_components` (fewer if the SVD produces less):
    (components[n, crop, crop], loadings[size_x, size_y, n], singular_values[n]).
    """
    size_x, size_y, crop_h, crop_w = fft_mat.shape
    data_mat = np.real(fft_mat.reshape(size_x * size_y, crop_h * crop_w))

    u, s, vt = np.linalg.svd(data_mat, full_matrices=False, compute_uv=True)

    n = min(num_components, s.shape[0])
    components = (s[:n] * vt[:n].T).T.reshape(n, crop_h, crop_w)
    loadings = u[:, :n].reshape(size_x, size_y, n)

    return components, loadings, s[:n]


def run_ica(fft_mat: np.ndarray, num_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent component analysis on the real part of the FFT stack.

    Returns (components[num_components, crop, crop], mixing[size_x, size_y, num_components]).
    """
    size_x, size_y, crop_h, crop_w = fft_mat.shape
    data_mat = np.real(fft_mat.reshape(size_x * size_y, crop_h * crop_w))

    ica = decomposition.FastICA(n_components=num_components, random_state=0)
    mixing = ica.fit_transform(data_mat).reshape(size_x, size_y, -1)
    components = ica.components_.reshape(-1, crop_h, crop_w)

    return components, mixing


def run_nmf(fft_mat: np.ndarray, num_components: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-negative matrix factorization on the magnitude of the FFT stack.

    Returns (components[num_components, crop, crop], abundances[size_x, size_y, num_components]).
    """
    size_x, size_y, crop_h, crop_w = fft_mat.shape
    data_mat = np.abs(fft_mat.reshape(size_x * size_y, crop_h * crop_w)).astype(np.float32)

    model = decomposition.NMF(
        n_components=num_components, init="random", random_state=0, max_iter=400
    )
    abundances = model.fit_transform(data_mat).reshape(size_x, size_y, num_components)
    components = model.components_.reshape(num_components, crop_h, crop_w)

    return components, abundances
