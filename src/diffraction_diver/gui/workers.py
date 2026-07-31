"""QThread workers so the sliding FFT and decompositions don't block the UI."""

import traceback

import numpy as np
from PySide6.QtCore import QObject, Signal

from diffraction_diver.core import analysis, carbontools, fft, radialprofile, structuremaps


class FFTWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object, int, int)
    error = Signal(str)

    def __init__(self, raw_image: np.ndarray, window_size: int, window_step: int, hamming: bool, crop_size: int):
        super().__init__()
        self.raw_image = raw_image
        self.window_size = window_size
        self.window_step = window_step
        self.hamming = hamming
        self.crop_size = crop_size

    def run(self) -> None:
        try:
            fft_stack, size_x, size_y = fft.sliding_fft(
                self.raw_image,
                self.window_size,
                self.window_step,
                self.hamming,
                self.crop_size,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.finished.emit(fft_stack, size_x, size_y)


class AnalysisWorker(QObject):
    # method name is carried in the signal so slots can be real bound methods
    # (not lambdas) - Qt needs a QObject receiver to know to queue the call
    # onto the main thread; a lambda has no thread affinity of its own and
    # would run in this worker thread instead, corrupting Qt's object tree.
    finished = Signal(str, object)
    error = Signal(str, str)

    def __init__(self, method: str, fft_stack: np.ndarray, num_components: int):
        super().__init__()
        self.method = method
        self.fft_stack = fft_stack
        self.num_components = num_components

    def run(self) -> None:
        try:
            if self.method == "pca":
                result = analysis.run_pca(self.fft_stack, self.num_components)
            elif self.method == "ica":
                result = analysis.run_ica(self.fft_stack, self.num_components)
            elif self.method == "nmf":
                result = analysis.run_nmf(self.fft_stack, self.num_components)
            else:
                raise ValueError(f"Unknown analysis method: {self.method}")
        except Exception:
            self.error.emit(self.method, traceback.format_exc())
            return
        self.finished.emit(self.method, result)


class StructureMapWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object, object, object)
    error = Signal(str)

    def __init__(
        self,
        raw_image: np.ndarray,
        window_size: int,
        window_step: int,
        hamming: bool,
        crop_size: int,
        peak_row: int,
        peak_col: int,
        fit_rad: int,
        pixel_size_nm: float | None,
        min_peak_ratio: float,
        highpass_radius: float,
    ):
        super().__init__()
        self.raw_image = raw_image
        self.window_size = window_size
        self.window_step = window_step
        self.hamming = hamming
        self.crop_size = crop_size
        self.peak_row = peak_row
        self.peak_col = peak_col
        self.fit_rad = fit_rad
        self.pixel_size_nm = pixel_size_nm
        self.min_peak_ratio = min_peak_ratio
        self.highpass_radius = highpass_radius

    def run(self) -> None:
        try:
            spacing_map, angle_map, intensity_map = structuremaps.compute_structure_maps(
                self.raw_image,
                self.window_size,
                self.window_step,
                self.hamming,
                self.crop_size,
                self.peak_row,
                self.peak_col,
                self.fit_rad,
                self.pixel_size_nm,
                self.min_peak_ratio,
                self.highpass_radius,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.finished.emit(spacing_map, angle_map, intensity_map)


class OrientationMapWorker(QObject):
    """Per-patch auto-find batch worker for Carbon Tools - see
    `core.carbontools.compute_orientation_maps`. Distinct from
    `StructureMapWorker`, which tracks one reflection from a fixed position."""

    progress = Signal(int, int)
    finished = Signal(object, object, object)
    error = Signal(str)

    def __init__(
        self,
        raw_image: np.ndarray,
        window_size: int,
        window_step: int,
        hamming: bool,
        crop_size: int,
        r_min: float,
        r_max: float,
        fit_rad: int,
        pixel_size_nm: float | None,
        min_peak_ratio: float,
        highpass_radius: float,
        mask: np.ndarray | None = None,
    ):
        super().__init__()
        self.raw_image = raw_image
        self.window_size = window_size
        self.window_step = window_step
        self.hamming = hamming
        self.crop_size = crop_size
        self.r_min = r_min
        self.r_max = r_max
        self.fit_rad = fit_rad
        self.pixel_size_nm = pixel_size_nm
        self.min_peak_ratio = min_peak_ratio
        self.highpass_radius = highpass_radius
        self.mask = mask

    def run(self) -> None:
        try:
            spacing_map, angle_map, intensity_map = carbontools.compute_orientation_maps(
                self.raw_image,
                self.window_size,
                self.window_step,
                self.hamming,
                self.crop_size,
                self.r_min,
                self.r_max,
                self.fit_rad,
                self.pixel_size_nm,
                self.min_peak_ratio,
                self.highpass_radius,
                mask=self.mask,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.finished.emit(spacing_map, angle_map, intensity_map)


class RadialProfileWorker(QObject):
    progress = Signal(int, int)
    finished = Signal(object, int, int)
    error = Signal(str)

    def __init__(self, raw_image: np.ndarray, window_size: int, window_step: int, hamming: bool, crop_size: int):
        super().__init__()
        self.raw_image = raw_image
        self.window_size = window_size
        self.window_step = window_step
        self.hamming = hamming
        self.crop_size = crop_size

    def run(self) -> None:
        try:
            profile_stack, size_x, size_y = radialprofile.compute_radial_profile_stack(
                self.raw_image,
                self.window_size,
                self.window_step,
                self.hamming,
                self.crop_size,
                progress_cb=lambda done, total: self.progress.emit(done, total),
            )
        except Exception:
            self.error.emit(traceback.format_exc())
            return
        self.finished.emit(profile_stack, size_x, size_y)
