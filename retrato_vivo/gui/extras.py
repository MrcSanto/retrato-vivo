"""Abas do diferencial do projeto: Retrato Vivo e Máquina do tempo."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PySide6.QtCore import QThread, QTimer, Signal
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from .. import living_portrait as lp
from ..io_utils import save_image
from ..pipeline import restore
from ..time_machine import ERAS, travel
from ..visualize import side_by_side
from .photo_view import PhotoView


class TaskWorker(QThread):
    """Roda uma função demorada fora da thread da interface."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[[], object]) -> None:
        super().__init__()
        self._task = task

    def run(self) -> None:
        try:
            result = self._task()
        except Exception as error:          # uma exceção solta mataria a thread em silêncio
            self.failed.emit(f"{type(error).__name__}: {error}")
            return
        self.done.emit(result)


def _intro(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    return label


# ================================================================ Retrato Vivo

class LivingPortraitView(QWidget):
    """Gera e toca o Retrato Vivo da foto restaurada."""

    def __init__(self) -> None:
        super().__init__()
        self.source: np.ndarray | None = None       # a foto restaurada (vem da janela)
        self.portrait: lp.LivingPortrait | None = None
        self.worker: TaskWorker | None = None
        self._frame = 0

        self.create_button = QPushButton("Criar Retrato Vivo")
        self.create_button.clicked.connect(self.create)
        self.show_depth = QCheckBox("Ver profundidade")
        self.show_depth.setToolTip("O que a IA acha que está perto (claro) e longe (escuro).")
        self.show_depth.toggled.connect(self._show_current)
        self.save_video_button = QPushButton("Salvar vídeo")
        self.save_video_button.clicked.connect(lambda: self._save("vídeo MP4 (*.mp4)", ".mp4", lp.save_video))
        self.save_gif_button = QPushButton("Salvar GIF")
        self.save_gif_button.clicked.connect(lambda: self._save("GIF animado (*.gif)", ".gif", lp.save_gif))
        self.status = QLabel()

        self.view = PhotoView()
        self.view.placeholder = "Restaure uma foto e clique em “Criar Retrato Vivo”"
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)

        buttons = QHBoxLayout()
        for widget in (self.create_button, self.show_depth, self.save_video_button, self.save_gif_button):
            buttons.addWidget(widget)
        buttons.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(_intro("A foto restaurada ganha vida: uma câmera virtual passeia diante dela e a "
                                "pessoa se destaca do fundo em 3D. A IA estima a distância de cada ponto "
                                "(Depth Anything) e reconstrói o fundo escondido atrás da pessoa (LaMa)."))
        layout.addLayout(buttons)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status)
        if not lp.is_available():
            self.status.setText("Modelo de profundidade não baixado. Rode: python -m retrato_vivo modelos")
        self.set_source(None)          # começa sem foto: nada a criar nem a salvar

    def set_source(self, image: np.ndarray | None) -> None:
        """Nova foto restaurada: o retrato anterior deixa de valer."""
        self.timer.stop()
        self.source, self.portrait = image, None
        self.view.clear()
        self._enable_saving(False)
        self.create_button.setEnabled(image is not None and lp.is_available())

    def create(self) -> None:
        if self.source is None:
            return
        self._wait()
        source = self.source
        self.worker = TaskWorker(lambda: lp.make_living_portrait(source))
        self.worker.done.connect(self._show_portrait)
        self.worker.failed.connect(lambda msg: (self._busy(False),
                                                QMessageBox.warning(self, "Retrato Vivo", msg)))
        self._busy(True)
        self.status.setText("Criando o Retrato Vivo… (alguns segundos)")
        self.worker.start()

    def _show_portrait(self, portrait: lp.LivingPortrait) -> None:
        self.portrait, self._frame = portrait, 0
        self._busy(False)
        self._enable_saving(True)
        seconds = sum(portrait.timings_ms.values()) / 1000
        self.status.setText(f"{len(portrait.frames)} quadros criados em {seconds:.1f} s ({portrait.device}).")
        self.timer.start(int(1000 / portrait.fps))
        self._show_current()

    def _next_frame(self) -> None:
        if self.portrait is None or self.show_depth.isChecked():
            return
        self._frame = (self._frame + 1) % len(self.portrait.frames)
        self.view.set_photo(self.portrait.frames[self._frame])

    def _show_current(self) -> None:
        if self.portrait is None:
            return
        if self.show_depth.isChecked():
            self.view.set_photo(lp.depth_colormap(self.portrait.depth))
        else:
            self.view.set_photo(self.portrait.frames[self._frame])

    def _save(self, file_filter: str, suffix: str, writer) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salvar Retrato Vivo", "retrato_vivo" + suffix, file_filter)
        if not path:
            return
        try:
            saved = writer(self.portrait, path)
        except Exception as error:
            QMessageBox.warning(self, "Não foi possível salvar", str(error))
            return
        self.status.setText(f"Salvo em {saved}")

    def _busy(self, busy: bool) -> None:
        self.create_button.setEnabled(not busy and self.source is not None)
        if busy:
            self.timer.stop()

    def _enable_saving(self, enabled: bool) -> None:
        self.save_video_button.setEnabled(enabled)
        self.save_gif_button.setEnabled(enabled)
        self.show_depth.setEnabled(enabled)

    def _wait(self) -> None:
        if self.worker is not None:
            self.worker.wait()

    def shutdown(self) -> None:
        self.timer.stop()
        self._wait()


# ============================================================ Máquina do tempo

class TimeMachineView(QWidget):
    """Mostra a foto aberta como se fosse de outra época (o inverso da restauração)."""

    def __init__(self) -> None:
        super().__init__()
        self.source: np.ndarray | None = None       # a foto aberta, como chegou
        self.aged: np.ndarray | None = None
        self.shown: np.ndarray | None = None
        self.era_buttons: dict[str, QPushButton] = {}

        eras = QHBoxLayout()
        for key, era in ERAS.items():
            button = QPushButton(key)
            button.setToolTip(f"{era.name}: {era.description}")
            button.clicked.connect(lambda _=False, k=key: self.travel_to(k))
            self.era_buttons[key] = button
            eras.addWidget(button)
        eras.addStretch()
        self.restore_back = QPushButton("Restaurar de volta")
        self.restore_back.setToolTip("Passa a foto envelhecida pela restauração: antes | depois.")
        self.restore_back.clicked.connect(self.round_trip)
        self.save_button = QPushButton("Salvar")
        self.save_button.clicked.connect(self._save)
        eras.addWidget(self.restore_back)
        eras.addWidget(self.save_button)

        self.view = PhotoView()
        self.view.placeholder = "Abra uma foto e escolha uma época"
        self.status = QLabel()
        layout = QVBoxLayout(self)
        layout.addWidget(_intro("O inverso da restauração: veja como a foto aberta ficaria se tivesse sido "
                                "tirada em outra época. Depois, “Restaurar de volta” mostra a "
                                "restauração desfazendo o envelhecimento."))
        layout.addLayout(eras)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status)
        self.set_source(None)

    def set_source(self, image: np.ndarray | None) -> None:
        self.source, self.aged, self.shown = image, None, None
        self.view.clear()
        for button in self.era_buttons.values():
            button.setEnabled(image is not None)
        self.restore_back.setEnabled(False)
        self.save_button.setEnabled(False)
        self.status.clear()

    def travel_to(self, era: str) -> None:
        if self.source is None:
            return
        self.aged = travel(self.source, era)
        self._show(self.aged)
        self.restore_back.setEnabled(True)
        self.status.setText(ERAS[era].name + ": " + ERAS[era].description)

    def round_trip(self) -> None:
        if self.aged is None:
            return
        restored = restore(self.aged).image
        self._show(side_by_side(self.aged, restored))
        self.status.setText("Esquerda: envelhecida. Direita: restaurada pelo Retrato Vivo.")

    def _show(self, image: np.ndarray) -> None:
        self.shown = image
        self.view.set_photo(image)
        self.save_button.setEnabled(True)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Salvar", "maquina_do_tempo.png",
                                              "PNG (*.png);;JPEG (*.jpg)")
        if path:
            self.status.setText(f"Salvo em {save_image(path, self.shown)}")
