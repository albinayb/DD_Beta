"""DiffractionDiver launcher: pick a module, each opening its own independent
top-level window with its own image loading - no state is shared between
modules."""

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from diffraction_diver.gui.peak_fit_window import PeakFitWindow
from diffraction_diver.gui.radial_profile_window import RadialProfileWindow
from diffraction_diver.gui.sliding_fft_window import SlidingFFTWindow


class LauncherWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DiffractionDiver")
        self.setFixedWidth(340)

        self._sliding_fft_window: SlidingFFTWindow | None = None
        self._peak_fit_window: PeakFitWindow | None = None
        self._radial_profile_window: RadialProfileWindow | None = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<b>DiffractionDiver</b>"))
        layout.addWidget(QLabel("Choose a module:"))

        sliding_fft_btn = QPushButton("Sliding FFT / Array Analysis")
        sliding_fft_btn.clicked.connect(self.open_sliding_fft)
        layout.addWidget(sliding_fft_btn)

        peak_fit_btn = QPushButton("Diffractogram Peak Fit")
        peak_fit_btn.clicked.connect(self.open_peak_fit)
        layout.addWidget(peak_fit_btn)

        radial_profile_btn = QPushButton("Radial Profile Analysis")
        radial_profile_btn.clicked.connect(self.open_radial_profile)
        layout.addWidget(radial_profile_btn)

    def open_sliding_fft(self) -> None:
        if self._sliding_fft_window is None:
            self._sliding_fft_window = SlidingFFTWindow()
        self._raise_window(self._sliding_fft_window)

    def open_peak_fit(self) -> None:
        if self._peak_fit_window is None:
            self._peak_fit_window = PeakFitWindow()
        self._raise_window(self._peak_fit_window)

    def open_radial_profile(self) -> None:
        if self._radial_profile_window is None:
            self._radial_profile_window = RadialProfileWindow()
        self._raise_window(self._radial_profile_window)

    @staticmethod
    def _raise_window(window) -> None:
        window.show()
        window.raise_()
        window.activateWindow()
