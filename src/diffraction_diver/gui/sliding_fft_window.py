"""Sliding-FFT array analysis window: slide an FFT window across the whole
raw image, then run PCA/ICA/NMF and inspect/save the results."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from diffraction_diver.core import analysis
from diffraction_diver.gui import plotting
from diffraction_diver.gui.calibration_widget import CalibrationWidget
from diffraction_diver.gui.canvas_utils import embed_figure
from diffraction_diver.gui.patch_selector import PatchSelectorWidget
from diffraction_diver.gui.reflection_picker import ReflectionPicker
from diffraction_diver.gui.workers import AnalysisWorker, FFTWorker

ANALYSIS_LABELS = {"pca": "PCA", "ica": "ICA", "nmf": "NMF"}


class SlidingFFTWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Sliding FFT / Array Analysis")
        self.resize(1400, 900)

        self.fft_stack: np.ndarray | None = None

        # Keep worker/thread references alive for the duration of the run.
        self._fft_thread: QThread | None = None
        self._fft_worker: FFTWorker | None = None
        self._analysis_runs: dict[str, tuple[QThread, AnalysisWorker]] = {}
        self._results_run_count = 0

        self._build_ui()

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        self.patch_selector = PatchSelectorWidget(show_patch_preview=True)
        self.patch_selector.changed.connect(self._on_patch_changed)
        self.patch_selector.image_loaded.connect(self._on_image_loaded)
        controls_layout.addWidget(self.patch_selector)
        controls_layout.addWidget(self._build_fft_run_group())
        self.analysis_group = self._build_analysis_group()
        controls_layout.addWidget(self.analysis_group)
        self.calibration = CalibrationWidget()
        self.calibration.changed.connect(self._on_calibration_changed)
        controls_layout.addWidget(self.calibration)
        controls_layout.addStretch(1)

        # Click a peak in the patch preview's FFT panel to calibrate from a
        # reflection of known spacing; its status text sits right under the
        # preview it refers to.
        self.reflection_picker = ReflectionPicker(
            self.patch_selector.patch_canvas,
            1,
            self.patch_selector,
            self.calibration,
            self.patch_selector.set_spectrum_selections,
        )
        controls_layout.insertWidget(1, self.reflection_picker.status_label)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls)
        controls_scroll.setFixedWidth(460)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        root.addWidget(controls_scroll)

        self.results_tabs = QTabWidget()
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.tabCloseRequested.connect(self.results_tabs.removeTab)
        root.addWidget(self.results_tabs, stretch=1)

    def _build_fft_run_group(self) -> QGroupBox:
        group = QGroupBox("Sliding FFT")
        v = QVBoxLayout(group)

        form = QFormLayout()
        self.window_step_spin = QSpinBox()
        self.window_step_spin.setRange(1, 4096)
        self.window_step_spin.setValue(4)
        form.addRow("Window step", self.window_step_spin)
        v.addLayout(form)

        self.run_fft_btn = QPushButton("Run Sliding FFT")
        self.run_fft_btn.setEnabled(False)
        self.run_fft_btn.clicked.connect(self.on_run_fft_clicked)
        v.addWidget(self.run_fft_btn)

        self.fft_progress = QProgressBar()
        v.addWidget(self.fft_progress)

        return group

    def _build_analysis_group(self) -> QGroupBox:
        group = QGroupBox("Decomposition")
        group.setEnabled(False)
        v = QVBoxLayout(group)

        tabs = QTabWidget()
        v.addWidget(tabs)

        pca_tab = QWidget()
        pca_layout = QFormLayout(pca_tab)
        pca_layout.addRow(QLabel("Components are ranked by singular value (see scree plot)."))
        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(2, 48)
        self.pca_components_spin.setValue(analysis.PCA_DEFAULT_COMPONENTS)
        pca_layout.addRow("Components to show", self.pca_components_spin)
        self.pca_run_btn = QPushButton("Run PCA")
        self.pca_run_btn.clicked.connect(lambda: self.on_run_analysis("pca"))
        pca_layout.addRow(self.pca_run_btn)
        tabs.addTab(pca_tab, "PCA")

        ica_tab = QWidget()
        ica_layout = QFormLayout(ica_tab)
        self.ica_components_spin = QSpinBox()
        self.ica_components_spin.setRange(2, 24)
        self.ica_components_spin.setValue(6)
        ica_layout.addRow("Components", self.ica_components_spin)
        self.ica_run_btn = QPushButton("Run ICA")
        self.ica_run_btn.clicked.connect(lambda: self.on_run_analysis("ica"))
        ica_layout.addRow(self.ica_run_btn)
        tabs.addTab(ica_tab, "ICA")

        nmf_tab = QWidget()
        nmf_layout = QFormLayout(nmf_tab)
        self.nmf_components_spin = QSpinBox()
        self.nmf_components_spin.setRange(2, 48)
        self.nmf_components_spin.setValue(12)
        nmf_layout.addRow("Components", self.nmf_components_spin)
        self.nmf_run_btn = QPushButton("Run NMF")
        self.nmf_run_btn.clicked.connect(lambda: self.on_run_analysis("nmf"))
        nmf_layout.addRow(self.nmf_run_btn)
        tabs.addTab(nmf_tab, "NMF")

        self._analysis_buttons = {"pca": self.pca_run_btn, "ica": self.ica_run_btn, "nmf": self.nmf_run_btn}

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 0)
        self.analysis_progress.setVisible(False)
        v.addWidget(self.analysis_progress)

        return group

    # --------------------------------------------------------- patch state

    def _on_patch_changed(self) -> None:
        # A previous sliding-FFT result no longer matches the current patch
        # parameters (window/crop/hamming/position may have changed since it
        # ran), so it must be invalidated rather than left available for
        # analysis under a new, mismatched label.
        self.fft_stack = None
        self.analysis_group.setEnabled(False)
        # The spectrum itself changed, so any picked reflection no longer applies.
        self.reflection_picker.clear()
        raw_image = self.patch_selector.raw_image
        self.run_fft_btn.setEnabled(raw_image is not None)
        self.calibration.set_image_reference(raw_image.shape[0] if raw_image is not None else None)

    def _on_image_loaded(self) -> None:
        self.calibration.reset()

    def _on_calibration_changed(self) -> None:
        self.patch_selector.set_pixel_size(self.calibration.pixel_size_nm)

    # --------------------------------------------------------- sliding FFT

    def on_run_fft_clicked(self) -> None:
        raw_image = self.patch_selector.raw_image
        if raw_image is None:
            return

        self.run_fft_btn.setEnabled(False)
        # Analysis buttons must stay disabled for the duration of this run -
        # otherwise a click on "Run PCA/ICA/NMF" before this finishes would
        # compute on the previous (stale) self.fft_stack, e.g. still at the
        # old crop size.
        self.analysis_group.setEnabled(False)
        self.fft_progress.setRange(0, 1)
        self.fft_progress.setValue(0)

        self._fft_thread = QThread()
        self._fft_worker = FFTWorker(
            raw_image,
            self.patch_selector.window_size,
            self.window_step_spin.value(),
            self.patch_selector.hamming,
            self.patch_selector.crop_size,
        )
        self._fft_worker.moveToThread(self._fft_thread)
        self._fft_thread.started.connect(self._fft_worker.run)
        self._fft_worker.progress.connect(self._on_fft_progress)
        self._fft_worker.finished.connect(self._on_fft_finished)
        self._fft_worker.error.connect(self._on_fft_error)
        self._fft_worker.finished.connect(self._fft_thread.quit)
        self._fft_worker.error.connect(self._fft_thread.quit)
        self._fft_thread.finished.connect(self._fft_thread.deleteLater)
        self._fft_thread.start()

    def _on_fft_progress(self, done: int, total: int) -> None:
        self.fft_progress.setRange(0, total)
        self.fft_progress.setValue(done)

    def _on_fft_finished(self, fft_stack: np.ndarray, size_x: int, size_y: int) -> None:
        self.fft_stack = fft_stack
        self.run_fft_btn.setEnabled(True)
        self.analysis_group.setEnabled(True)

    def _on_fft_error(self, message: str) -> None:
        self.run_fft_btn.setEnabled(True)
        # A prior successful run's stack (if any) is still valid and usable.
        self.analysis_group.setEnabled(self.fft_stack is not None)
        QMessageBox.critical(self, "Sliding FFT failed", message)

    # --------------------------------------------------------- analysis

    def on_run_analysis(self, method: str) -> None:
        if self.fft_stack is None:
            return

        num_components = {
            "pca": self.pca_components_spin,
            "ica": self.ica_components_spin,
            "nmf": self.nmf_components_spin,
        }[method].value()

        self._analysis_buttons[method].setEnabled(False)
        self.analysis_progress.setVisible(True)

        thread = QThread()
        worker = AnalysisWorker(method, self.fft_stack, num_components)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_analysis_finished)
        worker.error.connect(self._on_analysis_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self._analysis_runs[method] = (thread, worker)
        thread.start()

    def _on_analysis_finished(self, method: str, result) -> None:
        self.analysis_progress.setVisible(False)
        self._analysis_buttons[method].setEnabled(True)
        self._add_results_tab(method, result)

    def _on_analysis_error(self, method: str, message: str) -> None:
        self.analysis_progress.setVisible(False)
        self._analysis_buttons[method].setEnabled(True)
        QMessageBox.critical(self, f"{ANALYSIS_LABELS[method]} failed", message)

    # --------------------------------------------------------------- results

    def _add_results_tab(self, method: str, result) -> None:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        # Without this, Qt can leave stale pixels un-repainted in any empty
        # margin beside a narrower-than-the-viewport figure when scrolling -
        # a partial-repaint artifact, not a data/plotting bug.
        content.setAutoFillBackground(True)
        content_layout = QVBoxLayout(content)
        scroll.setWidget(content)
        tab_layout.addWidget(scroll)

        save_btn = QPushButton("Save results…")
        tab_layout.addWidget(save_btn)

        # Real-space grids (Loadings/Mixing/Abundance - one value per patch
        # position) can carry a scale bar; the Components grids (FFT-domain
        # patterns) can't - a spatial scale is meaningless in reciprocal space.
        map_scale_bar = None
        if self.calibration.pixel_size_nm is not None:
            map_pixel_size = self.window_step_spin.value() * self.calibration.pixel_size_nm
            map_scale_bar = {"pixel_size": map_pixel_size, "units": "nm"}

        if method == "pca":
            components, loadings, singular_values = result
            n = components.shape[0]
            comp_fig = plotting.make_grid_figure(
                [np.abs(c) for c in components], [f"PC {i + 1}" for i in range(n)], suptitle="PCA Components"
            )
            load_fig = plotting.make_grid_figure(
                plotting.map_images(loadings, n),
                [f"Loading {i + 1}" for i in range(n)],
                suptitle="PCA Loadings",
                scale_bar=map_scale_bar,
            )
            content_layout.addWidget(embed_figure(comp_fig))
            content_layout.addWidget(embed_figure(load_fig))
            scree_canvas = embed_figure(plotting.scree_figure(singular_values), fixed_width=True)
            content_layout.addWidget(scree_canvas, alignment=Qt.AlignLeft)
            save_btn.clicked.connect(lambda: self._save_pca(components, loadings, singular_values))
        else:
            components, maps = result
            n = components.shape[0]
            label = ANALYSIS_LABELS[method]
            map_label = "Mixing" if method == "ica" else "Abundance"
            comp_fig = plotting.make_grid_figure(
                list(components), [f"{label} {i + 1}" for i in range(n)], suptitle=f"{label} Components"
            )
            map_fig = plotting.make_grid_figure(
                plotting.map_images(maps, n),
                [f"{map_label} {i + 1}" for i in range(n)],
                suptitle=f"{label} {map_label} Coefficients",
                scale_bar=map_scale_bar,
            )
            content_layout.addWidget(embed_figure(comp_fig))
            content_layout.addWidget(embed_figure(map_fig))
            save_btn.clicked.connect(lambda: self._save_components_and_maps(method, components, maps))

        content_layout.addStretch(1)
        self._results_run_count += 1
        index = self.results_tabs.addTab(tab, f"{ANALYSIS_LABELS[method]} #{self._results_run_count}")
        self.results_tabs.setCurrentIndex(index)

    def _choose_output_dir_and_prefix(self, default_prefix: str) -> tuple[Path, str] | None:
        directory = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if not directory:
            return None
        prefix, ok = QInputDialog.getText(self, "Filename prefix", "Prefix for saved files:", text=default_prefix)
        if not ok:
            return None
        return Path(directory), (prefix.strip() or default_prefix)

    def _save_components_and_maps(self, method: str, components: np.ndarray, maps: np.ndarray) -> None:
        choice = self._choose_output_dir_and_prefix(method)
        if choice is None:
            return
        out, prefix = choice
        n = components.shape[0]
        np.save(out / f"{prefix}_components.npy", components)
        for i in range(n):
            np.save(out / f"{prefix}_component_{i + 1}.npy", components[i])
            np.save(out / f"{prefix}_map_{i + 1}.npy", maps[:, :, i])
        QMessageBox.information(self, "Saved", f"Saved {ANALYSIS_LABELS[method]} results to {out}")

    def _save_pca(self, components: np.ndarray, loadings: np.ndarray, singular_values: np.ndarray) -> None:
        choice = self._choose_output_dir_and_prefix("pca")
        if choice is None:
            return
        out, prefix = choice
        n = components.shape[0]
        np.save(out / f"{prefix}_components.npy", components)
        np.save(out / f"{prefix}_singular_values.npy", singular_values)
        for i in range(n):
            np.save(out / f"{prefix}_component_{i + 1}.npy", components[i])
            np.save(out / f"{prefix}_loading_{i + 1}.npy", loadings[:, :, i])
        QMessageBox.information(self, "Saved", f"Saved PCA results to {out}")
