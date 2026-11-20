"""Ponto de entrada da interface gráfica: ``python -m retrato_vivo.gui [foto]``.

Sem foto, abre a primeira de ``imagens/`` (se a pasta existir)."""

import sys

from PySide6.QtWidgets import QApplication

from .folder_nav import first_photo
from .main_window import MainWindow

app = QApplication(sys.argv)
window = MainWindow()
window.show()
start = sys.argv[1] if len(sys.argv) > 1 else first_photo()
if start:
    window.open_photo(start)
sys.exit(app.exec())
