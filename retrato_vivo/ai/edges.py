"""Bordas: Canny (PI clássico) × HED (Deep Learning), antes e depois da restauração.

É uma **visualização**, como ``visualize.py``: não altera a foto restaurada.
Serve para mostrar duas coisas na apresentação:

1. **O ganho da restauração.** Riscos, vincos e poeira viram "bordas" falsas na
   foto danificada; depois da restauração elas somem e ficam só os contornos reais.
2. **A diferença entre PI clássico e Deep Learning.**

   - **Canny** (1986) olha só o gradiente local: marca qualquer variação forte de
     brilho, inclusive textura, grão e sujeira. Os limiares são escolhidos
     automaticamente pela mediana do brilho da foto.
   - **HED** (Xie & Tu, ICCV 2015, *Holistically-Nested Edge Detection*) é uma
     rede VGG treinada com contornos desenhados por pessoas (BSDS500). Ela junta
     as bordas de cinco escalas e aprendeu a marcar contornos de objetos,
     ignorando boa parte da textura.

O HED roda em Caffe via ``cv2.dnn``. O OpenCV 4.14 até carrega a camada
``Crop`` da rede sem ajuda, mas recorta a partir do canto e as bordas saem
deslocadas (~35 px). Por isso registramos uma ``Crop`` própria, que recorta pelo
centro, como no exemplo oficial do OpenCV (``samples/dnn/edge_detection.py``).
"""

from __future__ import annotations

from contextlib import contextmanager

import cv2
import numpy as np

from .. import models

HED_MAX_SIDE = 512              # o HED na CPU leva ~1 s nesse tamanho; a saída volta ao tamanho original
HED_MEAN_BGR = (104.00698793, 116.66876762, 122.67891434)   # média usada no treino
CELL_MAX_HEIGHT = 480           # altura máxima de cada quadro do painel
LABEL_HEIGHT = 30

_NET: cv2.dnn.Net | None = None


def is_available() -> bool:
    return models.is_available("hed_rede") and models.is_available("hed_pesos")


class _CropLayer:
    """Camada ``Crop`` do Caffe recortando pelo CENTRO (a do OpenCV 4.14 recorta do canto).

    Recebe dois blobs: o que será recortado e um de referência, com o tamanho desejado.
    """

    def __init__(self, params, blobs):
        self.y0 = self.x0 = self.y1 = self.x1 = 0

    def getMemoryShapes(self, inputs):
        (n, c, h, w), (_, _, th, tw) = inputs[0], inputs[1]
        self.y0, self.x0 = (h - th) // 2, (w - tw) // 2
        self.y1, self.x1 = self.y0 + th, self.x0 + tw
        return [[n, c, th, tw]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.y0:self.y1, self.x0:self.x1]]


@contextmanager
def _center_crop():
    """Liga a nossa ``Crop`` só enquanto o HED é carregado e executado.

    O OpenCV cria as camadas na primeira execução (não no carregamento), então
    registrar só no ``readNet`` não basta. Removida no fim, para não mexer em
    outras redes Caffe (colorização).
    """
    cv2.dnn_registerLayer("Crop", _CropLayer)
    try:
        yield
    finally:
        cv2.dnn_unregisterLayer("Crop")


def _load_net() -> cv2.dnn.Net:
    global _NET
    if _NET is None:
        if not is_available():
            raise FileNotFoundError("modelo HED não baixado: rode  python -m retrato_vivo modelos")
        with _center_crop():
            _NET = cv2.dnn.readNetFromCaffe(str(models.model_path("hed_rede")),
                                            str(models.model_path("hed_pesos")))
    return _NET


def bordas_canny(img: np.ndarray) -> np.ndarray:
    """Bordas de Canny (uint8, 0 ou 255, H×W), com limiares automáticos pela mediana."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))
    return cv2.Canny(gray, low, max(high, low + 1))


def bordas_hed(img: np.ndarray) -> np.ndarray:
    """Probabilidade de borda segundo o HED (uint8, 0 a 255, H×W)."""
    net = _load_net()
    h, w = img.shape[:2]
    scale = min(1.0, HED_MAX_SIDE / max(h, w))
    small = img if scale == 1.0 else cv2.resize(img, (round(w * scale), round(h * scale)),
                                                interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    net.setInput(cv2.dnn.blobFromImage(small, 1.0, (sw, sh), HED_MEAN_BGR, swapRB=False, crop=False))
    with _center_crop():
        edges = net.forward()[0, 0]
    if (sh, sw) != (h, w):
        edges = cv2.resize(edges, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(edges * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _cell(edges: np.ndarray | None, height: int, width: int, label: str) -> np.ndarray:
    """Um quadro do painel: bordas escuras sobre fundo branco, com o nome em cima."""
    cell = np.full((height + LABEL_HEIGHT, width, 3), 30, np.uint8)
    if edges is None:
        cell[LABEL_HEIGHT:] = 235
        cv2.putText(cell, "modelo HED nao baixado", (10, LABEL_HEIGHT + height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 160), 1, cv2.LINE_AA)
    else:
        cell[LABEL_HEIGHT:] = cv2.cvtColor(255 - edges, cv2.COLOR_GRAY2BGR)
    cv2.putText(cell, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return cell


def painel_bordas(antes: np.ndarray, depois: np.ndarray, gap: int = 8) -> np.ndarray:
    """Painel 2×2: linhas Canny / HED, colunas antes / depois (BGR uint8).

    As duas fotos podem ter tamanhos diferentes (a super-resolução amplia a
    restaurada): as duas são reduzidas à mesma altura ANTES de detectar as bordas,
    senão a foto maior ganharia traços mais finos e densos e a comparação ficaria
    injusta. Sem o modelo HED, a
    linha do HED mostra um aviso e o painel sai assim mesmo. Os rótulos não têm
    acento porque ``cv2.putText`` só desenha ASCII.
    """
    height = min(CELL_MAX_HEIGHT, antes.shape[0], depois.shape[0])
    fotos = [cv2.resize(img, (round(img.shape[1] * height / img.shape[0]), height),
                        interpolation=cv2.INTER_AREA) for img in (antes, depois)]
    hed = is_available()
    rows = []
    for method, detect in (("Canny", bordas_canny), ("HED", bordas_hed if hed else None)):
        cells = []
        for img, when in zip(fotos, ("antes", "depois")):
            cells.append(_cell(detect(img) if detect else None, height, img.shape[1], f"{method} - {when}"))
        spacer = np.full((cells[0].shape[0], gap, 3), 255, np.uint8)
        rows.append(np.hstack([cells[0], spacer, cells[1]]))
    spacer = np.full((gap, rows[0].shape[1], 3), 255, np.uint8)
    return np.vstack([rows[0], spacer, rows[1]])
