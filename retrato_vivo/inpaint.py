"""Preenchimento de danos (inpainting): apaga o que a máscara marcou,
reconstruindo a região a partir dos pixels vizinhos.

Dois métodos clássicos do OpenCV:

- ``telea``: avança da borda para dentro (Fast Marching, Telea 2004).
  Rápido e bom para riscos finos e poeira.
- ``ns``: propaga as linhas de mesmo brilho para dentro do buraco
  (Navier-Stokes, Bertalmio et al. 2001). Melhor em áreas maiores.

Para buracos grandes (rasgos), nenhum dos dois inventa textura. Um
inpainting com Deep Learning pode ser plugado no pipeline (``inpainter=``).
"""

from __future__ import annotations

import cv2
import numpy as np

METHODS = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "telea",
            radius: float = 3.0) -> np.ndarray:
    """Preenche as regiões brancas de ``mask``.

    ``radius`` é o tamanho da vizinhança consultada, na escala de 1000 px
    (ajustado automaticamente à resolução da foto).
    """
    if method not in METHODS:
        raise ValueError(f"método desconhecido: {method!r}. Opções: {', '.join(METHODS)}")
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not mask.any():
        return img.copy()
    scaled_radius = max(1.0, radius * max(img.shape[:2]) / 1000.0)
    return cv2.inpaint(img, mask, scaled_radius, METHODS[method])
