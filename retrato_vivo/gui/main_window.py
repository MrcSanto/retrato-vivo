"""Janela principal: junta as abas, os controles e a barra de status."""

from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Retrato Vivo")
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Abra uma foto para começar.")
