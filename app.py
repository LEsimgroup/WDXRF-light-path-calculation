"""
xrf_sim/ui/app.py
=================
UI 入口。

    python -m xrf_sim.ui
"""

import sys
from PySide6.QtWidgets import QApplication
from .main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
