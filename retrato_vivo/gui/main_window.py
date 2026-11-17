"""Janela principal: junta as abas, os controles e a barra de status."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QTabWidget

from ..io_utils import SUPPORTED_EXTENSIONS, load_image
from .photo_view import PhotoView

# "Todos os arquivos" existe porque load_image também abre formatos fora da lista
# (AVIF, HEIC) e fotos salvas com a extensão errada.
OPEN_FILTER = ("Imagens (" + " ".join(f"*{ext}" for ext in SUPPORTED_EXTENSIONS)
               + ");;Todos os arquivos (*)")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Retrato Vivo")
        self.resize(1200, 800)
        self.setAcceptDrops(True)

        self.image: np.ndarray | None = None      # a foto aberta, como load_image devolveu
        self.photo_path: Path | None = None

        toolbar = self.addToolBar("Principal")
        toolbar.setMovable(False)
        open_action = toolbar.addAction("Abrir", self.choose_photo)
        open_action.setShortcut(QKeySequence.StandardKey.Open)

        self.photo_view = PhotoView()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.photo_view, "Antes e depois")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage("Abra uma foto ou arraste o arquivo para a janela.")

    def choose_photo(self) -> None:
        """Pergunta qual arquivo abrir. Cancelar o diálogo não muda nada."""
        path, _ = QFileDialog.getOpenFileName(self, "Abrir foto", "", OPEN_FILTER)
        if path:
            self.open_photo(path)

    def open_photo(self, path: str | Path) -> None:
        """Abre a foto e a mostra. Se o arquivo não abrir, avisa e mantém a foto atual."""
        path = Path(path)
        try:
            image = load_image(path)
        except (ValueError, OSError) as error:     # formato desconhecido, arquivo ausente ou sem permissão
            QMessageBox.warning(self, "Não foi possível abrir a foto", str(error))
            return
        self.image, self.photo_path = image, path
        self.photo_view.set_photo(image)
        height, width = image.shape[:2]
        self.statusBar().showMessage(f"{path.name} · {width} × {height} px")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """O Qt só entrega o ``dropEvent`` se o arrasto for aceito aqui."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            self.open_photo(files[0])
