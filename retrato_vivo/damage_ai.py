"""Detector de riscos com Deep Learning.

Rede: U-Net de detecção de riscos do "Bringing Old Photos Back to Life"
(Wan et al., CVPR 2020, Microsoft, licença MIT), treinada pelos autores em
fotos antigas reais anotadas. Os pesos oficiais (PyTorch) foram convertidos
para ONNX por ``scripts/exportar_detector_riscos.py`` e rodam no ONNX Runtime,
na GPU quando houver (``runtime.py``).

Por que existe, se ``damage.py`` já detecta riscos: as regras de lá (limiares
ajustados à mão) perdem riscos longos que se partem em pedaços, riscos sobre
bordas da cena e fotos cobertas de arranhões. A rede aprendeu com exemplos
reais e pega esses casos. As duas máscaras são SOMADAS em ``detect_damage``:
a rede para riscos e rachaduras; as regras para manchas, vincos e a proteção
de rostos.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import models, runtime

MODEL = "detector_riscos"
SHORT_SIDE = 256        # a resolução de trabalho dos autores; mais que isso gera falsos alarmes
MIN_AREA_FRAC = 2e-5    # pontinhos isolados menores que isso (fração da foto) são ignorados


def is_available() -> bool:
    return runtime.is_available() and models.is_available(MODEL)


def threshold_for(sensitivity: float) -> float:
    """Probabilidade mínima para marcar um pixel: 0,6 (só o certo) a 0,25 (pega tudo).
    Na sensibilidade padrão (0,5) fica perto do 0,4 que os autores usam."""
    return 0.6 - 0.35 * float(np.clip(sensitivity, 0.0, 1.0))


def scratch_probability(img: np.ndarray) -> np.ndarray:
    """Probabilidade (0 a 1) de cada pixel ser risco, no tamanho da foto."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    k = SHORT_SIDE / min(h, w)
    nh, nw = max(16, round(h * k / 16) * 16), max(16, round(w * k / 16) * 16)   # múltiplos de 16
    small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    x = small.astype(np.float32)[None, None] / 127.5 - 1.0                       # [-1, 1], como no treino
    prob = runtime.session(models.model_path(MODEL)).run(None, {"gray": x})[0][0, 0]
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def detect_scratches_ai(img: np.ndarray, sensitivity: float = 0.5) -> tuple[np.ndarray, dict[str, int]]:
    """Máscara booleana dos riscos achados pela rede e um relatório no formato de ``damage.py``."""
    mask = scratch_probability(img) >= threshold_for(sensitivity)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= MIN_AREA_FRAC * mask.size
    keep[0] = False
    report = {"candidatos": int(n - 1), "pequeno demais": int((~keep[1:]).sum()),
              "aceitos": int(keep.sum())}
    return keep[labels], report
