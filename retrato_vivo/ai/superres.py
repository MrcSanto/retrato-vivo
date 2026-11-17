"""Super-resolução: amplia fotos pequenas com Deep Learning, recriando detalhe.

Uma ampliação comum (bicúbica) só interpola os pixels vizinhos e a foto fica
borrada. Uma rede de super-resolução aprendeu, com milhares de pares de fotos
pequena/grande, como costumam ser as bordas e texturas, e recria parte desse
detalhe.

Dois modelos, ambos em TensorFlow (.pb), executados com ``cv2.dnn``:

- **ESPCN** (Shi et al., CVPR 2016), o padrão. Rede minúscula (100 KB),
  centésimos de segundo. Trabalha só no brilho (canal Y do YCrCb); a cor é
  ampliada pelo método bicúbico, porque o olho quase não percebe detalhe de cor.
- **EDSR** (Lim et al., CVPRW 2017), "qualidade máxima". Rede grande (38 MB)
  que trabalha nos três canais. É bem melhor e centenas de vezes mais lenta,
  por isso só roda em fotos de até ``EDSR_MAX_PIXELS``; acima disso usa o ESPCN.

Medido com fotos reduzidas e ampliadas de volta (PSNR contra a original):
ESPCN ganha de 0,2 a 1,1 dB do bicúbico; EDSR, cerca de 1,3 dB no 2x.
O ganho é real, mas modesto: não inventa detalhe que a foto nunca teve.

A etapa só amplia fotos menores que o tamanho alvo, escolhendo 2x ou 4x, e
nunca passa do alvo.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import models
from .tf_fix import fix_depth_to_space

MODELS = ("espcn", "edsr")
FACTORS = (2, 4)
DEFAULT_MODEL = "espcn"
DEFAULT_TARGET = 2048           # lado maior desejado, em pixels (igual ao max_side do pipeline)
EDSR_MAX_PIXELS = 250_000       # ~500x500: o EDSR 2x leva cerca de 1 min nesse tamanho
EDSR_MEAN_BGR = np.array([103.1545, 111.5287, 115.3436], np.float32)  # média do DIV2K, usada no treino

_NETS: dict[tuple[str, int], cv2.dnn.Net] = {}


def model_name(model: str, factor: int) -> str:
    if model not in MODELS or factor not in FACTORS:
        raise ValueError(f"modelo {model!r} / fator {factor} inválido: use {MODELS} e {FACTORS}")
    return f"{model}_x{factor}"


def is_available(model: str = DEFAULT_MODEL, factor: int = 2) -> bool:
    return models.is_available(model_name(model, factor))


def _load_net(model: str, factor: int) -> cv2.dnn.Net:
    key = (model, factor)
    if key not in _NETS:
        name = model_name(model, factor)
        if not models.is_available(name):
            raise FileNotFoundError(f"modelo {name} não baixado: rode  python -m retrato_vivo modelos")
        graph, _ = fix_depth_to_space(models.model_path(name).read_bytes())
        _NETS[key] = cv2.dnn.readNetFromTensorflow(np.frombuffer(graph, np.uint8))
    return _NETS[key]


def upscale_image(img: np.ndarray, factor: int = 2, model: str = DEFAULT_MODEL) -> np.ndarray:
    """Amplia ``img`` (BGR uint8) exatamente ``factor`` vezes com a rede escolhida."""
    net = _load_net(model, factor)
    h, w = img.shape[:2]
    if model == "espcn":
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        net.setInput(cv2.dnn.blobFromImage(ycc[:, :, 0].astype(np.float32) / 255.0))
        y = net.forward()[0, 0]
        out = cv2.resize(ycc, (w * factor, h * factor), interpolation=cv2.INTER_CUBIC)
        out[:, :, 0] = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)
    net.setInput(cv2.dnn.blobFromImage(img.astype(np.float32) - EDSR_MEAN_BGR))
    out = net.forward()[0].transpose(1, 2, 0) + EDSR_MEAN_BGR
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def choose_factor(long_side: int, target: int) -> int | None:
    """Menor fator (2 ou 4) que alcança o alvo; ``None`` se a foto já é grande o bastante."""
    if long_side * 1.25 >= target:      # ganho menor que 25% não compensa
        return None
    return 2 if long_side * 2 >= target else 4


def make_upscale_stage(model: str = DEFAULT_MODEL, target: int = DEFAULT_TARGET):
    """Cria a etapa de super-resolução para ``restore(..., post_stages=[...])``.

    Exemplo::

        restore(foto, post_stages=[("colorizacao", colorize),
                                   ("super_resolucao", make_upscale_stage("edsr"))])

    Coloque depois da colorização: colorir a foto menor é mais rápido e dá o mesmo resultado.
    """
    model_name(model, 2)  # valida o nome já na criação

    def upscale(img: np.ndarray, result) -> np.ndarray:
        h, w = img.shape[:2]
        factor = choose_factor(max(h, w), target)
        if factor is None:
            result.notes["super_resolucao"] = f"pulada: a foto já tem {max(h, w)} px (alvo {target} px)"
            return img

        chosen, reason = model, ""
        if chosen == "edsr" and h * w > EDSR_MAX_PIXELS:
            chosen, reason = "espcn", " (EDSR seria lento demais neste tamanho)"
        if not is_available(chosen, factor) and chosen == "edsr" and is_available("espcn", factor):
            chosen, reason = "espcn", " (EDSR não baixado)"
        if not is_available(chosen, factor):
            result.notes["super_resolucao"] = ("pulada: modelo não baixado "
                                               "(python -m retrato_vivo modelos)")
            return img

        out = upscale_image(img, factor, chosen)
        if max(out.shape[:2]) > target:                    # 4x pode passar do alvo: reduz o excesso
            k = target / max(out.shape[:2])
            out = cv2.resize(out, (round(out.shape[1] * k), round(out.shape[0] * k)),
                             interpolation=cv2.INTER_AREA)
        result.notes["super_resolucao"] = (f"ampliada {factor}x com {chosen.upper()}{reason}: "
                                           f"{w}x{h} -> {out.shape[1]}x{out.shape[0]} px")
        return out

    return upscale


upscale = make_upscale_stage()
"""Etapa pronta (ESPCN, alvo 2048 px). Uso: ``post_stages=[("super_resolucao", upscale)]``."""
