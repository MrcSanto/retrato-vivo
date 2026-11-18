"""Aba "Máscara": o usuário pinta por cima da foto o que a detecção automática errou.

- Pincel: marca um dano que a detecção não pegou (``mask_add``).
- Borracha: desmarca o que foi marcado por engano (``mask_remove``).

As duas máscaras são ``uint8`` do tamanho de ``result.original``, que é o formato que
``restore(mask_add=..., mask_remove=...)`` espera.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QRadioButton, QSlider, QVBoxLayout, QWidget

from .photo_view import PhotoView

ADD_COLOR = (0, 255, 255)        # RGB ciano: a cor do "pintado pelo usuário" no mapa de danos
REMOVE_COLOR = (0, 0, 255)       # RGB azul: não aparece no mapa de danos, então não se confunde
OVERLAY_ALPHA = 150              # de 0 (invisível) a 255 (cobre a foto)
DEFAULT_BRUSH_SIZE = 16          # diâmetro do pincel, em pixels da tela
MIN_BRUSH_SIZE, MAX_BRUSH_SIZE = 2, 80
RGBA_CHANNELS = 4


def overlay_image(mask_add: np.ndarray, mask_remove: np.ndarray) -> QImage:
    """As duas máscaras numa imagem transparente, para desenhar por cima da foto."""
    height, width = mask_add.shape
    rgba = np.zeros((height, width, RGBA_CHANNELS), np.uint8)       # alfa 0: tudo transparente
    rgba[mask_add > 0] = (*ADD_COLOR, OVERLAY_ALPHA)
    rgba[mask_remove > 0] = (*REMOVE_COLOR, OVERLAY_ALPHA)
    # A cópia mantém a imagem válida depois que o array ``rgba`` deixar de existir.
    return QImage(rgba.data, width, height, RGBA_CHANNELS * width, QImage.Format.Format_RGBA8888).copy()


class MaskCanvas(PhotoView):
    """A foto com as duas máscaras por cima. Arrastar o mouse pinta na máscara da ferramenta ativa."""

    def __init__(self) -> None:
        super().__init__()
        self.mask_add = np.zeros((0, 0), np.uint8)
        self.mask_remove = np.zeros((0, 0), np.uint8)
        self.erasing = False                                  # False: pincel. True: borracha.
        self.brush_size = DEFAULT_BRUSH_SIZE
        self._last_point: tuple[int, int] | None = None       # onde o traço em andamento parou, na foto
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_photo(self, img: np.ndarray) -> None:
        """Troca a foto de fundo. As máscaras só são mantidas se a foto tiver o mesmo tamanho."""
        super().set_photo(img)
        if self.mask_add.shape != img.shape[:2]:
            self.mask_add = np.zeros(img.shape[:2], np.uint8)
            self.mask_remove = np.zeros(img.shape[:2], np.uint8)

    def set_erasing(self, erasing: bool) -> None:
        self.erasing = erasing

    def set_brush_size(self, size: int) -> None:
        self.brush_size = size

    def clear_masks(self) -> None:
        self.mask_add[:] = 0
        self.mask_remove[:] = 0
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)      # a foto
        if self.photo_rect().isEmpty():
            return
        QPainter(self).drawImage(self.photo_rect(), overlay_image(self.mask_add, self.mask_remove))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.photo_rect().isEmpty():     # sem foto não há onde pintar
            return
        self._last_point = self._photo_point(event.position())
        self._paint_to(self._last_point)    # um clique sem arrastar já deixa um ponto

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_point is not None:
            self._paint_to(self._photo_point(event.position()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_point = None

    def _photo_scale(self) -> float:
        """Quantos pixels da foto cabem em um pixel da tela."""
        return self._photo.width() / self.photo_rect().width()

    def _photo_point(self, position: QPointF) -> tuple[int, int]:
        """Converte a posição do mouse, em pixels da tela, em coordenadas (x, y) da foto."""
        rect, scale = self.photo_rect(), self._photo_scale()
        return round((position.x() - rect.left()) * scale), round((position.y() - rect.top()) * scale)

    def _paint_to(self, point: tuple[int, int]) -> None:
        """Risca do último ponto até ``point`` na máscara da ferramenta ativa e apaga o mesmo traço da outra."""
        painted, other = (self.mask_remove, self.mask_add) if self.erasing else (self.mask_add, self.mask_remove)
        # O pincel é medido na tela: assim ele parece do mesmo tamanho em fotos de qualquer resolução.
        thickness = max(1, round(self.brush_size * self._photo_scale()))
        cv2.line(painted, self._last_point, point, 255, thickness)
        cv2.line(other, self._last_point, point, 0, thickness)
        self._last_point = point
        self.update()


class MaskEditor(QWidget):
    """As ferramentas em cima (pincel, borracha, tamanho, limpar) e a área de pintura embaixo."""

    def __init__(self) -> None:
        super().__init__()
        self.canvas = MaskCanvas()
        self.brush_button = QRadioButton("Pincel: marcar dano que faltou")
        self.brush_button.setChecked(True)
        self.eraser_button = QRadioButton("Borracha: desmarcar o que não é dano")
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self.size_slider.setValue(self.canvas.brush_size)
        self.clear_button = QPushButton("Limpar")

        tools = QHBoxLayout()
        tools.addWidget(self.brush_button)
        tools.addWidget(self.eraser_button)
        tools.addWidget(QLabel("Tamanho"))
        tools.addWidget(self.size_slider, stretch=1)
        tools.addWidget(self.clear_button)
        layout = QVBoxLayout(self)
        layout.addLayout(tools)
        layout.addWidget(self.canvas, stretch=1)

        self.eraser_button.toggled.connect(self.canvas.set_erasing)
        self.size_slider.valueChanged.connect(self.canvas.set_brush_size)
        self.clear_button.clicked.connect(self.canvas.clear_masks)
