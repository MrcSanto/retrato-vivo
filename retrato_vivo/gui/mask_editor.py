"""Aba "Máscara": o usuário pinta por cima da foto o que a detecção automática errou.

- Pincel: marca um dano que a detecção não pegou (``mask_add``).
- Borracha: desmarca o que foi marcado por engano (``mask_remove``).
- Roda do mouse: amplia a foto para pintar com precisão. Os outros botões do mouse a arrastam.

As duas máscaras são ``uint8`` do tamanho de ``result.original``, que é o formato que
``restore(mask_add=..., mask_remove=...)`` espera.
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QSize, Qt, Signal
from PySide6.QtGui import (QImage, QKeySequence, QMouseEvent, QPainter, QPaintEvent, QResizeEvent,
                           QWheelEvent)
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSlider,
                               QVBoxLayout, QWidget)

from ..pipeline import RestorationResult
from .photo_view import PhotoView

DAMAGE_MAP_STEP = "mapa_de_danos"   # o nome que o pipeline dá à etapa em result.steps
ADD_COLOR = (0, 255, 255)        # RGB ciano: a cor do "pintado pelo usuário" no mapa de danos
REMOVE_COLOR = (0, 0, 255)       # RGB azul: não aparece no mapa de danos, então não se confunde
OVERLAY_ALPHA = 150              # de 0 (invisível) a 255 (cobre a foto)
DEFAULT_BRUSH_SIZE = 16          # diâmetro do pincel, em pixels da tela
MIN_BRUSH_SIZE, MAX_BRUSH_SIZE = 2, 80
# Cada passo guarda uma cópia das duas máscaras: cerca de 6 MB na maior foto que o pipeline aceita.
MAX_UNDO_STEPS = 10
MAX_ZOOM = 8.0                   # o zoom 1 mostra a foto inteira; o 8 mostra um oitavo da largura dela
ZOOM_PER_NOTCH = 1.25            # quanto cada "clique" da roda do mouse amplia
WHEEL_NOTCH = 120                # o Qt mede o giro da roda em oitavos de grau: 120 por clique
MIN_SLIDER_WIDTH = 80            # em pixels da tela: cerca de 1 pixel por tamanho de pincel
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
    """A foto com as duas máscaras por cima. Arrastar com o botão esquerdo pinta na máscara da ferramenta ativa.

    A roda do mouse amplia a foto e os outros botões a arrastam.
    """

    zoom_changed = Signal(float)     # o zoom novo

    def __init__(self) -> None:
        super().__init__()
        self.mask_add = np.zeros((0, 0), np.uint8)
        self.mask_remove = np.zeros((0, 0), np.uint8)
        self.erasing = False                                  # False: pincel. True: borracha.
        self.brush_size = DEFAULT_BRUSH_SIZE
        self._last_point: tuple[int, int] | None = None       # onde o traço em andamento parou, na foto
        # Para o desfazer: as máscaras como estavam antes de cada traço. Passou do limite, a mais antiga sai.
        self._history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=MAX_UNDO_STEPS)
        self.zoom = 1.0                                       # 1: a foto inteira encaixada na área
        self._pan = QPointF()                                 # quanto a foto ampliada saiu do centro, na tela
        self._drag_from: QPointF | None = None                # onde o arrasto da foto em andamento parou, na tela
        self.setCursor(Qt.CursorShape.CrossCursor)

    def set_photo(self, img: np.ndarray) -> None:
        """Troca a foto de fundo. As máscaras e o zoom só são mantidos se a foto tiver o mesmo tamanho."""
        super().set_photo(img)
        if self.mask_add.shape != img.shape[:2]:
            self._start_over(img.shape[:2])

    def _start_over(self, shape: tuple[int, int]) -> None:
        """Máscaras zeradas, histórico vazio e foto inteira: nada do que foi feito na foto anterior continua."""
        self.mask_add = np.zeros(shape, np.uint8)
        self.mask_remove = np.zeros(shape, np.uint8)
        self._history.clear()
        self.reset_zoom()

    def masks(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """``(mask_add, mask_remove)`` para entregar a ``restore()``. Sem foto, não há máscara: ``(None, None)``.

        São cópias porque a restauração roda em outra thread e o usuário pode continuar pintando.
        """
        if self.photo_rect().isEmpty():
            return None, None
        return self.mask_add.copy(), self.mask_remove.copy()

    def clear(self) -> None:
        """Tira a foto e apaga a pintura: é o que acontece quando outra foto é aberta."""
        super().clear()
        self._start_over(self.mask_add.shape)

    def clear_masks(self) -> None:
        """O botão "Limpar": apaga a pintura, mas dá para voltar atrás com o desfazer."""
        self._remember()
        self.mask_add[:] = 0
        self.mask_remove[:] = 0
        self.update()

    def undo(self) -> None:
        """Volta as máscaras para antes do último traço ou do último "Limpar"."""
        if not self._history:
            return
        self.mask_add, self.mask_remove = self._history.pop()
        self.update()

    def _remember(self) -> None:
        """Guarda uma cópia das máscaras como estão agora, para o desfazer poder voltar a este ponto."""
        self._history.append((self.mask_add.copy(), self.mask_remove.copy()))

    def set_erasing(self, erasing: bool) -> None:
        self.erasing = erasing

    def set_brush_size(self, size: int) -> None:
        self.brush_size = size

    # -------------------------------------------------------------------- zoom

    def photo_rect(self) -> QRect:
        """Onde a foto fica na tela: encaixada na área, ampliada pelo zoom e deslocada pelo arrasto.

        Com zoom, passa dos limites da área; o Qt só desenha a parte que cabe. A pintura e o
        desenho usam este retângulo, por isso acompanham o zoom sem saber dele.
        """
        size = self._zoomed_size()
        # Arredonda para baixo, como fit_rect: assim o zoom 1 dá exatamente a foto encaixada.
        left = math.floor((self.width() - size.width()) / 2 + self._pan.x())
        top = math.floor((self.height() - size.height()) / 2 + self._pan.y())
        return QRect(QPoint(left, top), size)

    def _zoomed_size(self) -> QSize:
        """O tamanho da foto na tela. Sem foto, é vazio."""
        return super().photo_rect().size() * self.zoom

    def _move_photo(self, pan: QPointF) -> None:
        """Põe a foto ampliada a ``pan`` do centro, sem deixar área vazia onde ainda há foto para mostrar."""
        size = self._zoomed_size()
        # Em cada direção a foto anda só o quanto ela passa da área. Se cabe inteira, fica no centro.
        limit_x = max(0, size.width() - self.width()) / 2
        limit_y = max(0, size.height() - self.height()) / 2
        self._pan = QPointF(min(limit_x, max(-limit_x, pan.x())), min(limit_y, max(-limit_y, pan.y())))
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        self._move_photo(self._pan)     # a área mudou de tamanho, e com ela até onde a foto pode ir

    def zoom_at(self, position: QPointF, factor: float) -> None:
        """Multiplica o zoom por ``factor`` sem tirar do lugar o ponto da foto que está em ``position`` na tela."""
        if self.photo_rect().isEmpty():     # sem foto não há o que ampliar
            return
        new_zoom = min(MAX_ZOOM, max(1.0, self.zoom * factor))
        growth = new_zoom / self.zoom
        photo_center = QPointF(self.width() / 2, self.height() / 2) + self._pan
        self.zoom = new_zoom
        # A distância do ponto ao centro da foto cresce junto com a foto. Para o ponto
        # ficar parado, o centro se afasta dele essa diferença.
        self._move_photo(self._pan + (position - photo_center) * (1 - growth))
        self.zoom_changed.emit(self.zoom)

    def reset_zoom(self) -> None:
        """Volta a mostrar a foto inteira."""
        self.zoom = 1.0
        self._move_photo(QPointF())
        self.zoom_changed.emit(self.zoom)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Girar a roda para a frente amplia em volta do cursor; para trás, reduz."""
        notches = event.angleDelta().y() / WHEEL_NOTCH      # o touchpad manda frações de clique
        self.zoom_at(event.position(), ZOOM_PER_NOTCH ** notches)

    # ------------------------------------------------------------------ pintura

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)      # a foto
        if self.photo_rect().isEmpty():
            return
        QPainter(self).drawImage(self.photo_rect(), overlay_image(self.mask_add, self.mask_remove))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.photo_rect().isEmpty():     # sem foto não há onde pintar
            return
        if event.button() != Qt.MouseButton.LeftButton:     # só o esquerdo pinta; os outros arrastam a foto
            self._drag_from = event.position()
            return
        self._remember()                    # cada traço é um passo do desfazer
        self._last_point = self._photo_point(event.position())
        self._paint_to(self._last_point)    # um clique sem arrastar já deixa um ponto

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_from is not None:
            self._move_photo(self._pan + event.position() - self._drag_from)
            self._drag_from = event.position()
        elif self._last_point is not None:
            self._paint_to(self._photo_point(event.position()))

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._last_point = None
        self._drag_from = None

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
    """As ferramentas em cima (pincel, borracha, tamanho, desfazer, limpar) e a área de pintura embaixo."""

    def __init__(self) -> None:
        super().__init__()
        self._result: RestorationResult | None = None
        self.canvas = MaskCanvas()
        # Textos curtos para a linha de ferramentas caber na aba; a explicação fica na dica.
        self.brush_button = QRadioButton("Pincel")
        self.brush_button.setToolTip("Marca um dano que a detecção não pegou. Aparece em ciano.")
        self.brush_button.setChecked(True)
        self.eraser_button = QRadioButton("Borracha")
        self.eraser_button.setToolTip("Desmarca o que foi marcado por engano. Aparece em azul.")
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE)
        self.size_slider.setValue(self.canvas.brush_size)
        self.size_slider.setMinimumWidth(MIN_SLIDER_WIDTH)
        self.undo_button = QPushButton("Desfazer")
        self.undo_button.setShortcut(QKeySequence.StandardKey.Undo)     # Ctrl+Z
        self.clear_button = QPushButton("Limpar")
        # Com o mapa de fundo dá para ver o que já foi detectado: o que falta pintar e o que desmarcar.
        self.damage_checkbox = QCheckBox("Mostrar danos detectados")
        self.damage_checkbox.setChecked(True)
        self.zoom_button = QPushButton()         # o texto é o zoom atual, escrito por _show_zoom
        self.zoom_button.setToolTip("Gire a roda do mouse sobre a foto para ampliar. Para mover a foto ampliada, "
                                    "arraste com o botão direito. Clique aqui para ver a foto inteira.")
        self._show_zoom(self.canvas.zoom)
        self.apply_button = QPushButton("Aplicar correções")
        self.apply_button.setEnabled(False)      # a janela habilita quando há uma foto restaurada

        # Duas linhas, para a aba caber em janela estreita: em cima o que pinta, embaixo o que mostra e aplica.
        painting_tools = QHBoxLayout()
        painting_tools.addWidget(self.brush_button)
        painting_tools.addWidget(self.eraser_button)
        painting_tools.addWidget(QLabel("Tamanho"))
        painting_tools.addWidget(self.size_slider, stretch=1)
        painting_tools.addWidget(self.undo_button)
        painting_tools.addWidget(self.clear_button)
        view_and_apply = QHBoxLayout()
        view_and_apply.addWidget(self.damage_checkbox)
        view_and_apply.addStretch()
        view_and_apply.addWidget(self.zoom_button)
        view_and_apply.addWidget(self.apply_button)
        layout = QVBoxLayout(self)
        layout.addLayout(painting_tools)
        layout.addLayout(view_and_apply)
        layout.addWidget(self.canvas, stretch=1)

        self.eraser_button.toggled.connect(self.canvas.set_erasing)
        self.size_slider.valueChanged.connect(self.canvas.set_brush_size)
        self.undo_button.clicked.connect(self.canvas.undo)
        self.clear_button.clicked.connect(self.canvas.clear_masks)
        self.damage_checkbox.toggled.connect(self._show_background)
        self.canvas.zoom_changed.connect(self._show_zoom)
        self.zoom_button.clicked.connect(self.canvas.reset_zoom)

    def set_result(self, result: RestorationResult) -> None:
        """Põe a foto restaurada na área de pintura. A pintura continua se for a mesma foto."""
        self._result = result
        self._show_background()

    def clear(self) -> None:
        """Esvazia a aba: é o que acontece quando outra foto é aberta."""
        self._result = None
        self.canvas.clear()

    def _show_background(self) -> None:
        """Escolhe o fundo da pintura: o mapa de danos ou a foto original, que têm o mesmo tamanho."""
        if self._result is None:
            return
        damage_map = self._result.steps.get(DAMAGE_MAP_STEP)
        show_damage = self.damage_checkbox.isChecked() and damage_map is not None
        self.canvas.set_photo(damage_map if show_damage else self._result.original)

    def _show_zoom(self, zoom: float) -> None:
        self.zoom_button.setText(f"Zoom {zoom:.0%}")
