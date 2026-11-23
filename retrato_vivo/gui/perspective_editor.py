"""Aba "Perspectiva": os 4 cantos do papel por cima da foto aberta, para o usuário arrastar.

A correção de perspectiva ("Foto tirada do celular") acha o papel sozinha, mas desiste quando
ele se confunde com o fundo: papel de margem branca sobre mesa clara, por exemplo. Aqui o
usuário leva cada canto até a ponta da foto impressa. Ao soltar, a janela restaura de novo,
entregando os cantos ao pipeline em ``RestorationConfig.perspective_corners``.

Os cantos são sempre (x, y) em pixels da foto como foi aberta, que é o que o pipeline espera.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent, QPen, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..pipeline import RestorationConfig, RestorationResult
from .photo_view import PhotoView

# Medidas em pixels da tela.
HANDLE_RADIUS = 8                   # o círculo desenhado em cada canto
GRAB_RADIUS = 24                    # até esta distância do canto, o clique pega o canto
OUTLINE_WIDTH = 2                   # o traço verde do contorno do papel
OUTLINE_SHADOW_WIDTH = 4            # o traço escuro por baixo, que faz o contorno aparecer em área clara
PAPER_COLOR = QColor(0, 200, 0)     # o mesmo verde da etapa "Papel encontrado"
STARTING_INSET = 0.1                # sem papel encontrado, os cantos começam a esta fração da borda da foto

OFF_HINT = "Marque “Foto tirada do celular” nos controles para desentortar a foto."
FOUND_HINT = "O contorno verde é o papel encontrado. Se um canto estiver fora do lugar, arraste-o."
NOT_FOUND_HINT = "Não achei o papel. Arraste os 4 cantos até as pontas da foto impressa."
MARKED_HINT = "Cantos marcados à mão."
REFUSED_HINT = "Os cantos marcados não cercam uma área da foto. Arraste-os de novo."


def same_corners(first: np.ndarray | None, second: np.ndarray | None) -> bool:
    """Os dois pedidos de cantos são o mesmo? ``None`` (o pipeline detecta) só é igual a ``None``."""
    if first is None or second is None:
        return first is second
    return np.array_equal(first, second)


class CornerCanvas(PhotoView):
    """A foto com o contorno do papel por cima. Arrastar um canto com o botão esquerdo muda o contorno."""

    moved = Signal()     # o usuário soltou um canto em outro lugar

    def __init__(self) -> None:
        super().__init__()
        self.corners: np.ndarray | None = None      # 4×2, (x, y) na foto; None: não há contorno para mostrar
        self._dragged: int | None = None            # qual canto o mouse está arrastando
        self._grip = QPointF()                      # do ponto clicado até o centro do canto, na tela
        self._corner_before_drag = np.zeros(2, np.float32)

    def show_corners(self, corners: np.ndarray | None) -> None:
        """Troca o contorno mostrado.

        Durante um arrasto não troca: o canto voltaria para trás na mão do usuário.
        """
        if self._dragged is not None:
            return
        # Cópia: o arrasto altera o array, que não pode ser o de quem chamou.
        self.corners = None if corners is None else np.array(corners, np.float32)
        self.update()

    def clear(self) -> None:
        """Tira a foto e o contorno: é o que acontece quando outra foto é aberta."""
        super().clear()
        self._dragged = None
        self.show_corners(None)

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)      # a foto
        if self.corners is None or self.photo_rect().isEmpty():
            return
        points = [self._on_screen(corner) for corner in self.corners]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for color, width in ((Qt.GlobalColor.black, OUTLINE_SHADOW_WIDTH), (PAPER_COLOR, OUTLINE_WIDTH)):
            painter.setPen(QPen(color, width))
            painter.drawPolygon(QPolygonF(points))
        painter.setPen(QPen(Qt.GlobalColor.black, OUTLINE_WIDTH))
        painter.setBrush(PAPER_COLOR)
        for point in points:
            painter.drawEllipse(point, HANDLE_RADIUS, HANDLE_RADIUS)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.corners is None or self.photo_rect().isEmpty() or event.button() != Qt.MouseButton.LeftButton:
            return
        position = event.position()
        to_each_corner = [self._on_screen(corner) - position for corner in self.corners]
        distances = [math.hypot(offset.x(), offset.y()) for offset in to_each_corner]
        nearest = int(np.argmin(distances))
        if distances[nearest] > GRAB_RADIUS:
            return
        # O canto anda o mesmo que o mouse: pegá-lo fora do centro não o faz pular para o cursor.
        self._dragged, self._grip = nearest, to_each_corner[nearest]
        self._corner_before_drag = self.corners[nearest].copy()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragged is None:
            return
        self.corners[self._dragged] = self._on_photo(event.position() + self._grip)
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._dragged is None:
            return
        was_moved = not np.array_equal(self.corners[self._dragged], self._corner_before_drag)
        self._dragged = None
        if was_moved:       # um clique sem arrastar não muda nada: não há o que avisar
            self.moved.emit()

    def _screen_pixels_per_photo_pixel(self) -> float:
        return self.photo_rect().width() / self._photo.width()

    def _on_screen(self, corner: np.ndarray) -> QPointF:
        """Onde o ponto (x, y) da foto aparece na tela."""
        rect, scale = self.photo_rect(), self._screen_pixels_per_photo_pixel()
        return QPointF(rect.left() + float(corner[0]) * scale, rect.top() + float(corner[1]) * scale)

    def _on_photo(self, position: QPointF) -> tuple[float, float]:
        """O ponto (x, y) da foto que está em ``position`` na tela, sem passar da borda da foto."""
        rect, scale = self.photo_rect(), self._screen_pixels_per_photo_pixel()
        x = (position.x() - rect.left()) / scale
        y = (position.y() - rect.top()) / scale
        return (min(self._photo.width() - 1.0, max(0.0, x)),
                min(self._photo.height() - 1.0, max(0.0, y)))


class PerspectiveEditor(QWidget):
    """A explicação e o botão em cima, a foto com o contorno do papel no meio e a situação embaixo."""

    changed = Signal()      # os cantos marcados à mão mudaram: a janela restaura de novo

    def __init__(self) -> None:
        super().__init__()
        self._marked: np.ndarray | None = None      # os 4 cantos que o usuário marcou; None: o pipeline detecta
        self._photo_size = (0, 0)                   # largura e altura da foto aberta

        self.canvas = CornerCanvas()
        self.canvas.placeholder = "Abra uma foto tirada do celular para conferir os cantos do papel"
        self.detect_button = QPushButton("Detectar de novo")
        self.detect_button.setToolTip("Esquece os cantos marcados à mão e procura o papel de novo.")
        self.status = QLabel()
        intro = QLabel("Para foto de uma foto impressa, tirada com o celular. Se o contorno verde não estiver "
                       "nas bordas do papel, arraste cada canto até a ponta da foto impressa: a restauração "
                       "é refeita com os cantos marcados.")
        intro.setWordWrap(True)

        tools = QHBoxLayout()
        tools.addWidget(self.detect_button)
        tools.addStretch()
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(tools)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.status)

        self.canvas.moved.connect(self._mark_by_hand)
        self.detect_button.clicked.connect(self._detect_again)
        self.set_photo(None)        # começa sem foto: nada a mostrar nem a detectar de novo

    def set_photo(self, image: np.ndarray | None) -> None:
        """Outra foto foi aberta: os cantos marcados eram da anterior."""
        self._marked = None
        self.canvas.clear()
        if image is not None:
            self.canvas.set_photo(image)
            self._photo_size = (image.shape[1], image.shape[0])
        self.detect_button.setEnabled(False)
        self.status.clear()

    def marked_corners(self) -> np.ndarray | None:
        """Os cantos para ``RestorationConfig.perspective_corners``. ``None``: o usuário não marcou nenhum."""
        return self._marked

    def show_result(self, result: RestorationResult, config: RestorationConfig) -> None:
        """Mostra o contorno do papel usado em ``result``, a restauração pedida com ``config``."""
        if not config.correct_perspective:
            self._show(None, OFF_HINT)      # os cantos marcados ficam guardados para quando a caixa voltar
        elif not same_corners(config.perspective_corners, self._marked):
            return      # os cantos mudaram depois deste pedido: a restauração seguinte já vem com os novos
        elif self._marked is not None:
            self._show(self._marked, MARKED_HINT if result.paper_corners is not None else REFUSED_HINT)
        elif result.paper_corners is not None:
            self._show(result.paper_corners, FOUND_HINT)
        else:
            self._show(self._starting_corners(), NOT_FOUND_HINT)
        self.detect_button.setEnabled(config.correct_perspective and self._marked is not None)

    def _show(self, corners: np.ndarray | None, hint: str) -> None:
        self.canvas.show_corners(corners)
        self.status.setText(hint)

    def _starting_corners(self) -> np.ndarray:
        """Um retângulo um pouco para dentro da foto: o ponto de partida quando o papel não foi achado."""
        width, height = self._photo_size
        left, right = STARTING_INSET * width, (1 - STARTING_INSET) * width
        top, bottom = STARTING_INSET * height, (1 - STARTING_INSET) * height
        return np.float32([[left, top], [right, top], [right, bottom], [left, bottom]])

    def _mark_by_hand(self) -> None:
        """O usuário soltou um canto: os 4 da tela passam a valer no lugar da detecção."""
        self._marked = self.canvas.corners.copy()
        self.detect_button.setEnabled(True)
        self.status.setText(MARKED_HINT)
        self.changed.emit()

    def _detect_again(self) -> None:
        self._marked = None
        self.detect_button.setEnabled(False)
        self.status.setText("Procurando o papel…")
        self.changed.emit()
