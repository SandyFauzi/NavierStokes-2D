"""Entry point GUI."""

import sys
from PyQt6.QtWidgets import QApplication
from src.gui import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Navier-Stokes 2D Solver")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
