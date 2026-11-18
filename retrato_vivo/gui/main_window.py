"""Janela principal: junta as abas, os controles e a barra de status."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QProgressBar, QTabWidget

from ..io_utils import SUPPORTED_EXTENSIONS, load_image
from ..pipeline import RestorationResult
from .compare_view import CompareView
from .worker import RestoreWorker

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

        self.image: np.ndarray | None = None            # a foto aberta, como load_image devolveu
        self.photo_path: Path | None = None
        self.result: RestorationResult | None = None    # a última restauração da foto aberta
        self.worker: RestoreWorker | None = None

        self.toolbar = self.addToolBar("Principal")
        self.toolbar.setMovable(False)
        open_action = self.toolbar.addAction("Abrir", self.choose_photo)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.restore_action = self.toolbar.addAction("Restaurar", self.restore_photo)
        self.restore_action.setEnabled(False)           # só faz sentido com uma foto aberta

        self.compare_view = CompareView()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.compare_view, "Antes e depois")
        self.setCentralWidget(self.tabs)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)                    # sem faixa, a barra só anima: "ocupado"
        self.progress.setMaximumWidth(160)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Abra uma foto ou arraste o arquivo para a janela.")

    # ------------------------------------------------------------ abrir a foto

    def choose_photo(self) -> None:
        """Pergunta qual arquivo abrir. Cancelar o diálogo não muda nada."""
        path, _ = QFileDialog.getOpenFileName(self, "Abrir foto", "", OPEN_FILTER)
        if path:
            self.open_photo(path)

    def open_photo(self, path: str | Path) -> None:
        """Abre a foto, mostra e já manda restaurar. Se o arquivo não abrir, avisa e mantém a foto atual."""
        path = Path(path)
        try:
            image = load_image(path)
        except (ValueError, OSError) as error:     # formato desconhecido, arquivo ausente ou sem permissão
            QMessageBox.warning(self, "Não foi possível abrir a foto", str(error))
            return
        self.image, self.photo_path, self.result = image, path, None
        self.compare_view.set_photo(image)
        self.restore_action.setEnabled(True)
        self.restore_photo()

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """O Qt só entrega o ``dropEvent`` se o arrasto for aceito aqui."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if files:
            self.open_photo(files[0])

    # --------------------------------------------------------------- restaurar

    def restore_photo(self) -> None:
        """Restaura a foto aberta em segundo plano. O resultado chega em ``_show_result``."""
        self._wait_for_worker()
        self.worker = RestoreWorker(self.image)
        self.worker.done.connect(self._show_result)
        self.worker.failed.connect(self._show_failure)
        self._set_busy(True)
        self.statusBar().showMessage(f"{self._photo_label()} · restaurando…")
        self.worker.start()

    def _show_result(self, result: RestorationResult) -> None:
        self.result = result
        self.compare_view.set_comparison(result.original, result.image)
        self._set_busy(False)
        self.statusBar().showMessage(f"{self._photo_label()} · restaurada")

    def _show_failure(self, message: str) -> None:
        self._set_busy(False)
        self.statusBar().showMessage(f"{self._photo_label()} · a restauração falhou")
        QMessageBox.warning(self, "Não foi possível restaurar a foto", message)

    def _set_busy(self, busy: bool) -> None:
        """Enquanto uma restauração roda, a janela não aceita outra foto nem outro pedido.

        O pipeline guarda os modelos em variáveis globais, sem trava: duas
        restaurações ao mesmo tempo poderiam se atrapalhar.
        """
        self.toolbar.setEnabled(not busy)
        self.setAcceptDrops(not busy)
        self.progress.setVisible(busy)

    def _wait_for_worker(self) -> None:
        """O Qt aborta o programa se um ``QThread`` for destruído enquanto ainda roda."""
        if self.worker is not None:
            self.worker.wait()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._wait_for_worker()
        super().closeEvent(event)

    def _photo_label(self) -> str:
        height, width = self.image.shape[:2]
        return f"{self.photo_path.name} · {width} × {height} px"
