import sys

from PySide6.QtWidgets import QApplication

from diffraction_diver.gui.launcher_window import LauncherWindow


def run() -> None:
    app = QApplication(sys.argv)
    launcher = LauncherWindow()
    launcher.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
