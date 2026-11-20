"""Imagens para MOSTRAR o que o sistema fez (GUI, relatório, apresentação).

Nada aqui altera o resultado da restauração.
"""

from __future__ import annotations

import cv2
import numpy as np

from .damage import DamageResult

# Cores (BGR) do mapa de danos
COLOR_SCRATCH = (0, 0, 255)      # vermelho
COLOR_CREASE = (0, 128, 255)     # laranja
COLOR_SPOT = (0, 220, 255)       # amarelo
COLOR_AI = (255, 120, 0)         # azul: riscos achados pela IA
COLOR_LOSS = (255, 0, 255)       # magenta
COLOR_MANUAL = (255, 255, 0)     # ciano: pintado pelo usuário
COLOR_FACE = (0, 200, 0)         # verde: rosto protegido


def damage_map(img: np.ndarray, result: DamageResult, alpha: float = 0.6,
               draw_faces: bool = True) -> np.ndarray:
    """A foto com os danos detectados pintados por cima.

    Azul = risco achado pela IA, vermelho = risco, laranja = vinco, amarelo = mancha, magenta = perda grande, ciano = área
    adicionada manualmente, contorno verde = rosto protegido.
    """
    out = img.copy()

    def paint(region: np.ndarray, color: tuple[int, int, int]) -> None:
        if region.any():
            out[region] = (out[region] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)

    if result.ai_scratches is not None:
        paint(result.ai_scratches, COLOR_AI)    # primeiro: as detecções clássicas ficam por cima
    paint(result.scratches, COLOR_SCRATCH)
    paint(result.creases, COLOR_CREASE)
    paint(result.spots, COLOR_SPOT)
    paint(result.losses, COLOR_LOSS)

    automatic = (result.scratches | result.creases | result.spots | result.losses
                 | (result.ai_scratches if result.ai_scratches is not None else False)).astype(np.uint8)
    near_automatic = cv2.dilate(automatic, np.ones((7, 7), np.uint8)).astype(bool)
    paint((result.mask > 0) & ~near_automatic, COLOR_MANUAL)

    if draw_faces:
        thickness = max(1, int(round(2 * max(img.shape[:2]) / 1000.0)))
        for x, y, w, h in result.faces:
            cv2.rectangle(out, (x, y), (x + w, y + h), COLOR_FACE, thickness)
    return out


def side_by_side(before: np.ndarray, after: np.ndarray, gap: int = 8) -> np.ndarray:
    """Antes | depois, na mesma altura."""
    h = min(before.shape[0], after.shape[0])

    def fit(img: np.ndarray) -> np.ndarray:
        w = round(img.shape[1] * h / img.shape[0])
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    spacer = np.full((h, gap, 3), 255, np.uint8)
    return np.hstack([fit(before), spacer, fit(after)])


def steps_panel(steps: dict[str, np.ndarray], columns: int = 4, cell_height: int = 360) -> np.ndarray:
    """Um painel único com todas as etapas, cada uma com seu nome em cima.

    Os nomes das etapas não têm acento de propósito: as fontes embutidas do
    OpenCV (``cv2.putText``) só desenham ASCII.
    """
    label_h = 30
    cells = []
    for i, (name, img) in enumerate(steps.items()):
        w = round(img.shape[1] * cell_height / img.shape[0])
        cell = np.full((cell_height + label_h, w, 3), 30, np.uint8)
        cell[label_h:] = cv2.resize(img, (w, cell_height), interpolation=cv2.INTER_AREA)
        cv2.putText(cell, f"{i}. {name}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cell)

    rows = []
    for start in range(0, len(cells), columns):
        row = cells[start:start + columns]
        rows.append(np.hstack([cv2.copyMakeBorder(c, 0, 6, 0, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                               for c in row]))
    width = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
            for r in rows]
    return np.vstack(rows)
