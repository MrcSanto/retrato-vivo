"""Comparação antes e depois: um divisor arrastável sobre a foto."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen

from .convert import to_qimage
from .photo_view import PhotoView

# Medidas em pixels da tela.
DIVIDER_WIDTH = 2               # o traço branco
DIVIDER_OUTLINE_WIDTH = 4       # o traço escuro por baixo, que faz o divisor aparecer em área clara
LABEL_MARGIN = 10               # do rótulo até a borda do lado da foto a que ele se refere
LABEL_PADDING = 6               # do texto até a borda do fundo do rótulo
LABEL_BACKGROUND = QColor(0, 0, 0, 150)     # preto meio transparente


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
        before_side = QRect(photo.left(), photo.top(), divider_x - photo.left(), photo.height())
        after_side = QRect(divider_x, photo.top(), photo.right() - divider_x + 1, photo.height())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        # A original é desenhada por cima da restaurada, mas o recorte só deixa passar
        # o que está à esquerda do divisor.
        painter.setClipRect(before_side)
        painter.drawImage(photo, self._before)
        painter.setClipping(False)

        for color, width in ((Qt.GlobalColor.black, DIVIDER_OUTLINE_WIDTH), (Qt.GlobalColor.white, DIVIDER_WIDTH)):
            painter.setPen(QPen(color, width))
            painter.drawLine(divider_x, photo.top(), divider_x, photo.bottom())
        self._draw_label(painter, "Antes", before_side, Qt.AlignmentFlag.AlignLeft)
        self._draw_label(painter, "Depois", after_side, Qt.AlignmentFlag.AlignRight)

    def _draw_label(self, painter: QPainter, text: str, side: QRect, corner: Qt.AlignmentFlag) -> None:
        """Escreve ``text`` no canto de cima de ``side``. Se o lado ficou estreito demais, não escreve."""
        metrics = painter.fontMetrics()
        box = QRect(0, 0, metrics.horizontalAdvance(text) + 2 * LABEL_PADDING,
                    metrics.height() + 2 * LABEL_PADDING)
        if box.width() + 2 * LABEL_MARGIN > side.width():
            return
        if corner == Qt.AlignmentFlag.AlignLeft:
            box.moveTopLeft(side.topLeft() + QPoint(LABEL_MARGIN, LABEL_MARGIN))
        else:
            box.moveTopRight(side.topRight() + QPoint(-LABEL_MARGIN, LABEL_MARGIN))
        painter.fillRect(box, LABEL_BACKGROUND)
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

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
