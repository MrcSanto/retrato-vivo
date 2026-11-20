"""Colorização com DDColor (alternativa mais realista ao modelo de Zhang).

Modelo: DDColor (Kang et al., ICCV 2023, "Towards Photo-Realistic Image
Colorization via Dual Decoders"), licença Apache-2.0. Um decodificador entende
a cena e outro decide as cores por regiões; o resultado é bem mais natural que
o de Zhang (2016), que tende a manchas avermelhadas em fundos. Roda no ONNX
Runtime, na GPU quando houver (``runtime.py``).

Funciona como ``colorize.py``: a rede olha só a luminosidade (canal L do LAB)
e prevê a cor (canais a/b). Juntando o L original com o a/b previsto, a foto
fica colorida sem perder nenhum detalhe. A rede trabalha em 512 x 512; a cor é
ampliada de volta para o tamanho da foto.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import models, runtime

MODEL = "ddcolor"
SIDE = 512
DEFAULT_STRENGTH = 1.0     # a cor do DDColor já é contida; não precisa reduzir como a de Zhang


def is_available() -> bool:
    return runtime.is_available() and models.is_available(MODEL)


def predict_ab(img: np.ndarray) -> np.ndarray:
    """Canais a/b (LAB em ponto flutuante) previstos para a foto, no tamanho dela."""
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, :1]
    gray_rgb = cv2.cvtColor(np.concatenate([L, np.zeros_like(L), np.zeros_like(L)], axis=-1),
                            cv2.COLOR_LAB2RGB)                      # só a luminosidade, em RGB
    x = cv2.resize(gray_rgb, (SIDE, SIDE), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)[None]
    ab = runtime.session(models.model_path(MODEL)).run(None, {"input": x.astype(np.float32)})[0][0]
    return cv2.resize(ab.transpose(1, 2, 0), (w, h), interpolation=cv2.INTER_CUBIC)


def colorize_image(img: np.ndarray, strength: float = DEFAULT_STRENGTH) -> np.ndarray:
    """Foto colorida: L original + a/b previsto (multiplicado por ``strength``)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[:, :, :1]
    lab = np.concatenate([L, predict_ab(img) * strength], axis=-1)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(out * 255.0, 0, 255).round().astype(np.uint8)


def make_ddcolor_stage(strength: float = DEFAULT_STRENGTH):
    """Etapa para ``restore(post_stages=...)``, com o mesmo nome e regras da de Zhang:
    só colore foto P&B/sépia, e o motivo fica em ``result.notes["colorizacao"]``."""

    def colorize(img: np.ndarray, result) -> np.ndarray:
        if not result.diagnosis.monochrome:
            result.notes["colorizacao"] = "pulada: a foto já é colorida"
            return img
        if not is_available():
            result.notes["colorizacao"] = "pulada: modelo DDColor não baixado (python -m retrato_vivo modelos ddcolor)"
            return img
        result.notes["colorizacao"] = (f"foto P&B/sépia colorida pelo DDColor em {runtime.device_name()} "
                                       f"(intensidade {strength:.2f})")
        return colorize_image(img, strength)

    return colorize
