"""Aba "Bordas": Canny (PI clássico) × HED (Deep Learning), antes e depois da restauração."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..ai import edges
from ..io_utils import save_image
from ..pipeline import RestorationResult
from .extras import TaskWorker, _intro
from .photo_view import PhotoView


class EdgesView(QWidget):
    """Mostra o painel 2×2 de ``ai.edges.painel_bordas`` para a última restauração.

    O painel só é calculado quando a aba aparece: o HED leva cerca de um segundo e não
    vale a pena gastá-lo a cada ajuste nos controles se ninguém está olhando.
    """

    def __init__(self) -> None:
        super().__init__()
        self.before: np.ndarray | None = None
        self.after: np.ndarray | None = None
        self.panel: np.ndarray | None = None
        self.worker: TaskWorker | None = None
        self._generation = 0          # muda a cada restauração: painel de uma restauração antiga é descartado

        self.save_button = QPushButton("Salvar painel")
        self.save_button.clicked.connect(self._save)
        self.status = QLabel()
        self.view = PhotoView()
        self.view.placeholder = "Restaure uma foto para comparar as bordas"

        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(_intro("Bordas da foto antes e depois da restauração. Em cima, o Canny (método "
                                "clássico) marca qualquer variação de brilho, inclusive textura e grão. "
                                "Embaixo, o HED (rede neural treinada com contornos desenhados por "
                                "pessoas) marca só os contornos dos objetos."))
        layout.addLayout(buttons)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status)
        self.set_result(None)

    def set_result(self, result: RestorationResult | None) -> None:
        """Nova restauração: o painel anterior deixa de valer."""
        self._generation += 1
        self.before, self.after = (result.original, result.image) if result is not None else (None, None)
        self.panel = None
        self.view.clear()
        self.save_button.setEnabled(False)
        self.status.setText("" if edges.is_available() else
                            "Modelo HED não baixado: só o Canny aparece. Rode: python -m retrato_vivo modelos")
        if result is not None and self.isVisible():
            self.update_panel()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self.panel is None and self.before is not None:
            self.update_panel()

    def update_panel(self) -> None:
        """Calcula o painel em segundo plano, se ainda não estiver calculando."""
        if self.before is None or (self.worker is not None and self.worker.isRunning()):
            return
        before, after, generation = self.before, self.after, self._generation

        def task():
            started = time.monotonic()
            return generation, edges.painel_bordas(before, after), time.monotonic() - started

        self.worker = TaskWorker(task)
        self.worker.done.connect(self._show_panel)
        self.worker.failed.connect(lambda msg: self.status.setText(f"Não foi possível calcular as bordas: {msg}"))
        self.status.setText("Calculando as bordas…")
        self.worker.start()

    def _show_panel(self, outcome: tuple[int, np.ndarray, float]) -> None:
        generation, panel, seconds = outcome
        if generation != self._generation:      # chegou uma restauração nova enquanto calculava
            self.worker.wait()                  # o sinal sai antes de a thread terminar de vez
            self.update_panel()
            return
        self.panel = panel
        self.view.set_photo(panel)
        self.save_button.setEnabled(True)
        self.status.setText(f"Bordas calculadas em {seconds:.1f} s." if edges.is_available() else
                            "Modelo HED não baixado: só o Canny aparece. Rode: python -m retrato_vivo modelos")

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salvar painel de bordas", "bordas.jpg",
                                              "JPEG (*.jpg);;PNG (*.png)")
        if path:
            self.status.setText(f"Salvo em {save_image(path, self.panel)}")

    def shutdown(self) -> None:
        """O Qt aborta o programa se um ``QThread`` for destruído enquanto ainda roda."""
        if self.worker is not None:
            self.worker.wait()
