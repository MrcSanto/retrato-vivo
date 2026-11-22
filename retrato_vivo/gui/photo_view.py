"""Área que mostra uma foto encaixada no espaço disponível."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage, QPainter, QPaintEvent, QPalette
from PySide6.QtWidgets import QWidget

from .convert import fit_rect, to_qimage

PLACEHOLDER_EXTRA_POINTS = 4    # o texto de área vazia é um pouco maior que o resto da interface


class PhotoView(QWidget):
    """Desenha a foto inteira, centralizada e na proporção original."""

    def __init__(self) -> None:
        super().__init__()
        self._photo: QImage | None = None
        self.placeholder = ""     # texto mostrado no meio da área enquanto não há foto

    def set_photo(self, img: np.ndarray) -> None:
        self._photo = to_qimage(img)
        self.update()     # pede ao Qt para chamar paintEvent de novo

    def clear(self) -> None:
        self._photo = None
        self.update()

    def photo_rect(self) -> QRect:
        """Onde a foto fica dentro do widget. Vazio enquanto não há foto."""
        if self._photo is None:
            return QRect()
        return fit_rect(self._photo.size(), self.rect())

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        if self._photo is None:
            self._draw_placeholder(painter)
            return
        # Sem isto o Qt reduz a foto pelo vizinho mais próximo e ela fica serrilhada.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.photo_rect(), self._photo)

    def _draw_placeholder(self, painter: QPainter) -> None:
        font = painter.font()
        font.setPointSize(font.pointSize() + PLACEHOLDER_EXTRA_POINTS)
        painter.setFont(font)
        painter.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))    # o cinza de "campo vazio" do tema
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)
