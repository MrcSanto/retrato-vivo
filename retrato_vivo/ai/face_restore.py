"""Restauração de rostos com Deep Learning (GFPGAN).

Modelo: GFPGAN v1.4 (Wang et al., CVPR 2021, "Towards Real-World Blind Face
Restoration with Generative Facial Prior"), uma GAN que aprendeu como rostos
reais são e reconstrói olhos, boca, pele e cabelo a partir de um rosto
borrado, granulado ou danificado. Roda no ONNX Runtime, na GPU quando houver
(``runtime.py``).

Como funciona:

1. O YuNet (``faces.py``) acha cada rosto e 5 pontos: olhos, nariz e cantos da boca.
2. Uma transformação de semelhança (rotação + escala + translação) leva esses
   pontos para as posições padrão do conjunto de treino (FFHQ, 512 x 512).
   É assim que a rede espera receber o rosto.
3. A rede restaura o rosto alinhado.
4. A transformação inversa devolve o rosto ao lugar, com borda suave.

``intensidade`` mistura o rosto restaurado com o original: 1 = só a rede,
0 = original. Abaixo de 1 o rosto fica mais parecido com a pessoa real.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import models, runtime
from ..faces import detect_faces_landmarks
from ..preprocess import is_gray

MODEL = "gfpgan"
SIDE = 512
DEFAULT_STRENGTH = 0.7
MIN_FACE_PX = 24           # rosto menor que isso (largura) não tem detalhe para restaurar

# Posição dos 5 pontos no rosto alinhado do FFHQ (fração de 512 px).
FFHQ_TEMPLATE = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813],
                          [0.50123859, 0.61331904], [0.39308822, 0.72541100],
                          [0.61150205, 0.72490465]], dtype=np.float32) * SIDE


def is_available() -> bool:
    return runtime.is_available() and models.is_available(MODEL)


def _restore_aligned(face: np.ndarray) -> np.ndarray:
    """Roda o GFPGAN num rosto já alinhado (512 x 512, BGR)."""
    x = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    out = runtime.session(models.model_path(MODEL)).run(None, {"input": x})[0][0]
    out = (np.clip(out, -1, 1) + 1) / 2 * 255.0
    return cv2.cvtColor(out.transpose(1, 2, 0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _paste_mask() -> np.ndarray:
    """Máscara do rosto alinhado com borda suave (evita emenda visível)."""
    mask = np.zeros((SIDE, SIDE), np.float32)
    pad = int(SIDE * 0.06)
    mask[pad:SIDE - pad, pad:SIDE - pad] = 1.0
    return cv2.GaussianBlur(mask, (0, 0), SIDE * 0.04)


def restore_faces(img: np.ndarray, strength: float = DEFAULT_STRENGTH) -> tuple[np.ndarray, int]:
    """Restaura todos os rostos da foto. Retorna (foto, quantos rostos foram restaurados)."""
    out = img.copy()
    gray_photo = is_gray(img)
    H, W = img.shape[:2]
    soft = _paste_mask()
    count = 0
    for (x, y, w, h), points in detect_faces_landmarks(img):
        if w < MIN_FACE_PX:
            continue
        matrix, _ = cv2.estimateAffinePartial2D(points, FFHQ_TEMPLATE, method=cv2.LMEDS)
        if matrix is None:
            continue
        aligned = cv2.warpAffine(out, matrix, (SIDE, SIDE), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
        restored = _restore_aligned(aligned)
        if gray_photo:                       # foto P&B: a cor fica por conta da colorização
            restored = cv2.cvtColor(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        inverse = cv2.invertAffineTransform(matrix)
        back = cv2.warpAffine(restored, inverse, (W, H), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(soft, inverse, (W, H), flags=cv2.INTER_LINEAR)[..., None] * strength
        out = np.clip(out * (1 - alpha) + back * alpha, 0, 255).astype(np.uint8)
        count += 1
    return out, count


def make_face_restore_stage(strength: float = DEFAULT_STRENGTH):
    """Etapa para ``restore(post_stages=...)``. O motivo fica em ``result.notes["rostos_ia"]``."""

    def face_restore(img: np.ndarray, result) -> np.ndarray:
        if not is_available():
            result.notes["rostos_ia"] = "pulada: modelo não baixado (python -m retrato_vivo modelos gfpgan)"
            return img
        out, count = restore_faces(img, strength)
        result.notes["rostos_ia"] = (f"{count} rosto(s) restaurado(s) pelo GFPGAN em {runtime.device_name()} "
                                     f"(intensidade {strength:.2f})" if count else "pulada: nenhum rosto encontrado")
        return out

    return face_restore
