"""Ponte entre o OpenCV e o Qt: converte a imagem e encaixa a foto na área de desenho."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QImage

BYTES_PER_PIXEL = 3   # B, G e R, um byte cada


def to_qimage(img: np.ndarray) -> QImage:
    """Converte uma imagem do projeto (BGR ``uint8``) em ``QImage``, para o Qt desenhar."""
    # Um recorte de outra imagem não ocupa memória contínua; o Qt precisa das linhas em sequência.
    img = np.ascontiguousarray(img)
    height, width = img.shape[:2]
    borrowed = QImage(img.data, width, height, BYTES_PER_PIXEL * width, QImage.Format.Format_BGR888)
    # O QImage acima só aponta para a memória do NumPy. A cópia continua válida
    # depois que o array for alterado ou liberado.
    return borrowed.copy()


def fit_rect(image_size: QSize, area: QRect) -> QRect:
    """O maior retângulo com a proporção da imagem que cabe em ``area``, centralizado."""
    size = image_size.scaled(area.size(), Qt.AspectRatioMode.KeepAspectRatio)
    left = area.x() + (area.width() - size.width()) // 2
    top = area.y() + (area.height() - size.height()) // 2
    return QRect(QPoint(left, top), size)
