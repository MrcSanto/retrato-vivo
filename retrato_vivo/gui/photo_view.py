"""Área que mostra uma foto encaixada no espaço disponível."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect
from PySide6.QtGui import QImage, QPainter, QPaintEvent
from PySide6.QtWidgets import QWidget

from .convert import fit_rect, to_qimage


class PhotoView(QWidget):
    """Desenha a foto inteira, centralizada e na proporção original."""

    def __init__(self) -> None:
        super().__init__()
        self._photo: QImage | None = None

    def set_photo(self, img: np.ndarray) -> None:
        self._photo = to_qimage(img)
        self.update()     # pede ao Qt para chamar paintEvent de novo

    def photo_rect(self) -> QRect:
        """Onde a foto fica dentro do widget. Vazio enquanto não há foto."""
        if self._photo is None:
            return QRect()
        return fit_rect(self._photo.size(), self.rect())

    def paintEvent(self, event: QPaintEvent) -> None:
        if self._photo is None:
            return
        painter = QPainter(self)
        # Sem isto o Qt reduz a foto pelo vizinho mais próximo e ela fica serrilhada.
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(self.photo_rect(), self._photo)
