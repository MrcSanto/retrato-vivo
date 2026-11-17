"""Colorização automática de fotos P&B com Deep Learning.

Modelo: "Colorful Image Colorization", Zhang, Isola e Efros (ECCV 2016),
uma CNN treinada em 1,3 milhão de fotos do ImageNet, executada com ``cv2.dnn``.

Como funciona, em uma frase: a foto vai para o espaço LAB, onde o canal L é
a luminosidade (a foto P&B em si) e os canais a/b são a cor; a rede olha só
o L e prevê o a/b. Juntando o L original com o a/b previsto, temos a foto
colorida sem perder nenhum detalhe da original.

Detalhes que importam:

- A rede trabalha em 224x224 e devolve a cor em 56x56. A cor é ampliada de
  volta para o tamanho da foto; como o olho humano é muito menos sensível a
  detalhes de cor do que de luminosidade, a baixa resolução não aparece.
- A rede não prevê uma cor direto: ela dá uma probabilidade para cada uma de
  313 cores possíveis (``pts_in_hull.npy``) e a resposta é a média ponderada.
  O fator 2,606 "esquenta" essa média (*annealed mean* do artigo), para a cor
  não ficar desbotada.
- ``intensidade`` multiplica o a/b: 1 é a cor da rede, 0 é a foto cinza.
  Abaixo de 1 a cor fica menos "berrante".
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import models

NET_SIDE = 224                 # tamanho de entrada da rede
ANNEALED_MEAN_T = 2.606        # constante do artigo (temperatura da média das 313 cores)
DEFAULT_STRENGTH = 0.8         # um pouco abaixo da cor "pura" da rede
REQUIRED_MODELS = ("colorizacao_rede", "colorizacao_pesos", "colorizacao_cores")

_NET = None


def is_available() -> bool:
    """Os três arquivos do modelo foram baixados?"""
    return all(models.is_available(name) for name in REQUIRED_MODELS)


def _load_net() -> cv2.dnn.Net:
    global _NET
    if _NET is None:
        if not is_available():
            raise FileNotFoundError("modelo de colorização não baixado: "
                                    "rode  python -m retrato_vivo modelos")
        net = cv2.dnn.readNetFromCaffe(str(models.model_path("colorizacao_rede")),
                                       str(models.model_path("colorizacao_pesos")))
        # As duas camadas finais não têm pesos no .caffemodel: são preenchidas aqui.
        hull = np.load(models.model_path("colorizacao_cores"))          # (313, 2): as cores a/b
        net.getLayer(net.getLayerId("class8_ab")).blobs = [
            hull.T.reshape(2, 313, 1, 1).astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [
            np.full((1, 313), ANNEALED_MEAN_T, np.float32)]
        _NET = net
    return _NET


def predict_ab(img: np.ndarray) -> np.ndarray:
    """Cor prevista pela rede: canais a/b (H, W, 2) float32, no tamanho de ``img``."""
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)  # L em 0..100
    L_small = cv2.resize(lab[:, :, 0], (NET_SIDE, NET_SIDE), interpolation=cv2.INTER_AREA)

    net = _load_net()
    net.setInput(cv2.dnn.blobFromImage(L_small - 50.0))   # a rede espera L centrado em zero
    ab_small = net.forward()[0].transpose(1, 2, 0)        # (56, 56, 2)

    h, w = img.shape[:2]
    return cv2.resize(ab_small, (w, h), interpolation=cv2.INTER_CUBIC)


def colorize_image(img: np.ndarray, strength: float = DEFAULT_STRENGTH) -> np.ndarray:
    """Colore uma foto (BGR uint8). Mantém o L original e troca só a cor.

    Lança ``FileNotFoundError`` se o modelo não foi baixado.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    ab = predict_ab(img) * strength
    out = cv2.merge([lab[:, :, 0], ab[:, :, 0], ab[:, :, 1]])
    bgr = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0 + 0.5, 0, 255).astype(np.uint8)


def make_colorize_stage(strength: float = DEFAULT_STRENGTH):
    """Cria a etapa de colorização para ``restore(..., post_stages=[...])``.

    Exemplo::

        restore(foto, post_stages=[("colorizacao", make_colorize_stage(0.7))])

    Só age em foto P&B ou sépia (``result.diagnosis.monochrome``); foto colorida
    passa direto. O motivo da decisão fica em ``result.notes["colorizacao"]``.
    """

    def colorize(img: np.ndarray, result) -> np.ndarray:
        if not result.diagnosis.monochrome:
            result.notes["colorizacao"] = "pulada: a foto já é colorida"
            return img
        if not is_available():
            result.notes["colorizacao"] = ("pulada: modelo não baixado "
                                           "(python -m retrato_vivo modelos)")
            return img
        result.notes["colorizacao"] = f"foto P&B/sépia colorida pela rede (intensidade {strength:.2f})"
        return colorize_image(img, strength)

    return colorize


colorize = make_colorize_stage()
"""Etapa pronta, com a intensidade padrão. Uso: ``post_stages=[("colorizacao", colorize)]``."""
