"""Comparação antes e depois: um divisor arrastável sobre a foto."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPaintEvent, QPen

from .convert import to_qimage
from .photo_view import PhotoView

DIVIDER_WIDTH = 2   # em pixels da tela


class CompareView(PhotoView):
    """Mostra a foto original à esquerda do divisor e a restaurada à direita."""

    def __init__(self) -> None:
        super().__init__()
        self._before: QImage | None = None
        self.split = 0.5     # posição do divisor: 0 é a borda esquerda da foto, 1 é a direita
        self.setCursor(Qt.CursorShape.SplitHCursor)

    def set_photo(self, img: np.ndarray) -> None:
        """Mostra uma foto só, sem comparação (a foto recém-aberta, ainda não restaurada)."""
        self._before = None
        super().set_photo(img)

    def set_comparison(self, before: np.ndarray, after: np.ndarray) -> None:
        self.set_photo(after)
        self._before = to_qimage(before)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)     # a foto restaurada, inteira
        if self._before is None:
            return
        photo = self.photo_rect()
        divider_x = photo.left() + round(self.split * photo.width())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # A original é desenhada por cima da restaurada, mas o recorte só deixa passar
        # o que está à esquerda do divisor.
        painter.setClipRect(QRect(photo.left(), photo.top(), divider_x - photo.left(), photo.height()))
        painter.drawImage(photo, self._before)
        painter.setClipping(False)
        painter.setPen(QPen(Qt.GlobalColor.white, DIVIDER_WIDTH))
        painter.drawLine(divider_x, photo.top(), divider_x, photo.bottom())

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._move_divider_to(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # O Qt só envia este evento com um botão apertado, ou seja, durante o arrasto.
        self._move_divider_to(event.position().x())

    def _move_divider_to(self, x: float) -> None:
        photo = self.photo_rect()
        if photo.isEmpty():
            return
        self.split = min(1.0, max(0.0, (x - photo.left()) / photo.width()))
        self.update()
