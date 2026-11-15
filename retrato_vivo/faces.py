"""Detecção de rostos com Deep Learning (YuNet, uma CNN leve do OpenCV Zoo).

Usada para PROTEGER os rostos: dentro deles a detecção de danos fica mais
rígida, para não "apagar" pupilas, cílios ou o contorno dos lábios. As
caixas dos rostos também ficam disponíveis para as etapas de IA (Marco).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import models
from .io_utils import resize_max_side

Box = tuple[int, int, int, int]  # (x, y, largura, altura)

_DETECTOR = None
_DETECTION_SIDE = 640  # o YuNet roda numa cópia reduzida: rápido e suficiente


def detect_faces(img: np.ndarray, min_score: float = 0.6) -> list[Box]:
    """Retorna as caixas dos rostos, na resolução de ``img``.

    Se o modelo não foi baixado (``python -m retrato_vivo modelos``),
    retorna lista vazia e o resto do pipeline funciona sem a proteção.
    """
    global _DETECTOR
    if not models.is_available("yunet"):
        return []
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    small, k = resize_max_side(img, _DETECTION_SIDE)
    h, w = small.shape[:2]
    if _DETECTOR is None:
        _DETECTOR = cv2.FaceDetectorYN.create(str(models.model_path("yunet")), "",
                                              (w, h), min_score, 0.3, 50)
    _DETECTOR.setInputSize((w, h))
    _DETECTOR.setScoreThreshold(min_score)
    _, found = _DETECTOR.detect(small)
    if found is None:
        return []

    H, W = img.shape[:2]
    boxes: list[Box] = []
    for f in found:
        x, y, bw, bh = (int(v) for v in (f[:4] / k).round())  # volta à escala original
        x, y = max(0, x), max(0, y)
        boxes.append((x, y, min(bw, W - x), min(bh, H - y)))
    return boxes


def faces_region(shape: tuple[int, ...], boxes: list[Box]) -> np.ndarray:
    """Máscara booleana (H, W) com True dentro das caixas dos rostos."""
    region = np.zeros(shape[:2], dtype=bool)
    for x, y, w, h in boxes:
        region[y:y + h, x:x + w] = True
    return region
