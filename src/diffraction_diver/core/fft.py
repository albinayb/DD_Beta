"""Sliding-window 2D FFT over a raw image.

Ported from the original notebook's `ApplyHamming` / `GenerateXYPos` /
`MakeWindow` / `Do_Sliding_FFT_discrete` / `zoom_discrete`, with globals
removed and plotting stripped out.
"""

from collections.abc import Callable

import numpy as np
from scipy import fftpack


def apply_hamming(patch: np.ndarray) -> np.ndarray:
    """Apply a 2D Hamming window to a square image patch."""
    window_size = patch.shape[0]
    bw2d = np.outer(np.hamming(window_size), np.ones(window_size))
    bw2d = np.sqrt(bw2d * bw2d.T)
    return patch * bw2d


def generate_xy_positions(
    window_size: int, window_step: int, image_width: int, image_height: int
) -> tuple[np.ndarray, int, int]:
    """Generate the (x, y) top-left corners of every window position.

    Returns (pos_mat, num_steps_x, num_steps_y).
    """
    xpos_vec = np.arange(0, image_width - window_size, window_step)
    ypos_vec = np.arange(0, image_height - window_size, window_step)
    num_steps_x = len(xpos_vec)
    num_steps_y = len(ypos_vec)

    pos_mat = np.zeros((num_steps_x * num_steps_y, 2), dtype=np.int64)
    for i in range(num_steps_x):
        for j in range(num_steps_y):
            pos_mat[i * num_steps_y + j, :] = (xpos_vec[i], ypos_vec[j])

    return pos_mat, num_steps_x, num_steps_y


def max_window_position(window_size: int, image_width: int, image_height: int) -> tuple[int, int]:
    """Largest valid (xpos, ypos) so that a window of `window_size` fits in the image."""
    return (
        max(0, image_width - window_size - 1),
        max(0, image_height - window_size - 1),
    )


def window_fft(
    raw_image: np.ndarray,
    xpos: int,
    ypos: int,
    window_size: int,
    hamming: bool,
    crop_size: int,
) -> np.ndarray:
    """Compute the (optionally Hamming-windowed) FFT of a single patch.

    Crops `raw_image` to a `window_size` x `window_size` patch at (xpos, ypos),
    optionally applies a Hamming window, takes the 2D FFT, shifts zero-frequency
    to the center, then center-crops the frequency-domain result to `crop_size`.
    """
    patch = raw_image[xpos : xpos + window_size, ypos : ypos + window_size]
    patch = apply_hamming(patch) if hamming else patch

    spectrum = fftpack.fftshift(fftpack.fft2(patch))

    if crop_size >= window_size:
        return spectrum

    lo = (window_size - crop_size) // 2
    hi = lo + crop_size
    return spectrum[lo:hi, lo:hi]


def sliding_fft(
    raw_image: np.ndarray,
    window_size: int,
    window_step: int,
    hamming: bool,
    crop_size: int,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[np.ndarray, int, int]:
    """Run `window_fft` over every window position in `raw_image`.

    Returns (fft_stack, size_x, size_y) where fft_stack has shape
    (size_x, size_y, crop_size, crop_size) and is complex-valued.
    """
    pos_mat, size_x, size_y = generate_xy_positions(
        window_size, window_step, raw_image.shape[0], raw_image.shape[1]
    )
    total = size_x * size_y
    if total == 0:
        raise ValueError(
            "No window positions fit in this image with the given window_size/window_step."
        )

    fft_stack = np.zeros((total, crop_size, crop_size), dtype=np.complex128)
    for i in range(total):
        xpos, ypos = pos_mat[i]
        fft_stack[i] = window_fft(raw_image, xpos, ypos, window_size, hamming, crop_size)
        if progress_cb is not None and (i % 100 == 0 or i == total - 1):
            progress_cb(i + 1, total)

    return fft_stack.reshape(size_x, size_y, crop_size, crop_size), size_x, size_y
