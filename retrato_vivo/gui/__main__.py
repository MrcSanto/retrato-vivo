"""Ponto de entrada da interface gráfica: ``python -m retrato_vivo.gui [foto]``."""

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.show()
if len(sys.argv) > 1:
    window.open_photo(sys.argv[1])
sys.exit(app.exec())
