"""Preenchimento de danos (inpainting): apaga o que a máscara marcou,
reconstruindo a região a partir da vizinhança.

Três métodos:

- ``telea``: avança da borda para dentro (Fast Marching, Telea 2004).
  Rápido e ótimo para riscos finos e poeira.
- ``ns``: propaga as linhas de mesmo brilho (Navier-Stokes, Bertalmio 2001).
- ``ia``: **híbrido com Deep Learning**. Riscos finos continuam no Telea; buracos
  largos (rasgos, pedaços faltando, áreas pintadas com o pincel) vão para o
  **LaMa** (Suvorov et al., WACV 2022), uma rede treinada para inventar textura
  plausível em áreas grandes, onde os métodos clássicos só borram. Roda na GPU
  quando houver (``runtime.py``).

Sem o modelo baixado, ``ia`` cai para o Telea e avisa em ``last_note``.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import models, runtime

METHODS = ("telea", "ns", "ia")
_CLASSIC = {"telea": cv2.INPAINT_TELEA, "ns": cv2.INPAINT_NS}

LAMA_MODEL = "lama"
LAMA_SIDE = 512            # o LaMa trabalha em 512 x 512
WIDE_HOLE_PX = 10.0        # buraco mais largo que isto (escala 1000 px) vai para o LaMa; o Telea vai bem até aqui
CONTEXT = 1.0              # margem de contexto ao redor do buraco, em proporção do tamanho dele
MIN_CROP_FRAC = 0.25       # recorte mínimo: 25% do maior lado da foto (dá contexto à rede)

last_note = ""             # o que o último inpaint("ia") fez, para o pipeline mostrar


def lama_available() -> bool:
    return runtime.is_available() and models.is_available(LAMA_MODEL)


def _as_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _scale(img: np.ndarray) -> float:
    return max(img.shape[:2]) / 1000.0


def inpaint_classic(img: np.ndarray, mask: np.ndarray, method: str = "telea",
                    radius: float = 3.0) -> np.ndarray:
    """Telea ou Navier-Stokes, com o raio ajustado à resolução."""
    mask = _as_mask(mask)
    if not mask.any():
        return img.copy()
    return cv2.inpaint(img, mask, max(1.0, radius * _scale(img)), _CLASSIC[method])


# ===================================================================== LaMa

def lama_patch(patch: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Roda o LaMa num recorte qualquer (redimensiona para 512 e volta)."""
    h, w = patch.shape[:2]
    x = cv2.resize(patch, (LAMA_SIDE, LAMA_SIDE), interpolation=cv2.INTER_AREA
                   if max(h, w) > LAMA_SIDE else cv2.INTER_CUBIC)
    m = cv2.resize(hole.astype(np.uint8), (LAMA_SIDE, LAMA_SIDE), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))        # cobre a auréola do redimensionamento
    image = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    sess = runtime.session(models.model_path(LAMA_MODEL))
    out = sess.run(None, {"image": image, "mask": (m > 0).astype(np.float32)[None, None]})[0][0]
    out = cv2.cvtColor(np.clip(out.transpose(1, 2, 0), 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA
                      if max(h, w) < LAMA_SIDE else cv2.INTER_CUBIC)


def _crop_boxes(wide: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Recortes quadrados com contexto em volta de cada grupo de buracos largos.

    Buracos próximos caem no mesmo recorte (as caixas que se sobrepõem são
    unidas), para a rede ver o conjunto e não emendar dois preenchimentos.
    """
    H, W = wide.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(wide.astype(np.uint8), connectivity=8)
    boxes = []
    for x, y, w, h, _area in stats[1:]:
        side = max(w, h) * (1 + 2 * CONTEXT)
        side = int(min(max(side, MIN_CROP_FRAC * max(H, W)), max(H, W)))
        cx, cy = x + w / 2, y + h / 2
        x0 = int(np.clip(cx - side / 2, 0, max(0, W - side)))
        y0 = int(np.clip(cy - side / 2, 0, max(0, H - side)))
        boxes.append([x0, y0, min(W, x0 + side), min(H, y0 + side)])
    merged = True
    while merged:                       # une caixas que se sobrepõem até não sobrar nenhuma
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [tuple(b) for b in boxes]


def inpaint_ai(img: np.ndarray, mask: np.ndarray, radius: float = 3.0) -> np.ndarray:
    """Híbrido: Telea nos buracos finos, LaMa nos largos (ver o topo do arquivo)."""
    global last_note
    hole = np.asarray(mask) > 0
    if not hole.any():
        last_note = "nada a preencher"
        return img.copy()
    if not lama_available():
        last_note = "modelo LaMa não baixado: usando o Telea (python -m retrato_vivo modelos lama)"
        return inpaint_classic(img, hole, "telea", radius)

    # Separa os buracos largos: os que têm algum ponto a mais de meia "largura de risco"
    # da borda. A transformada de distância mede isso igual em qualquer direção
    # (uma erosão pequena deixaria passar linhas diagonais finas).
    half_width = max(1.5, WIDE_HOLE_PX * _scale(img) / 2)
    padded = cv2.copyMakeBorder(hole.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    core = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1] > half_width
    n, labels = cv2.connectedComponents(hole.astype(np.uint8), connectivity=8)
    wide_ids = np.unique(labels[core])
    wide = np.isin(labels, wide_ids[wide_ids > 0])
    thin = hole & ~wide

    out = inpaint_classic(img, thin, "telea", radius) if thin.any() else img.copy()
    boxes = _crop_boxes(wide) if wide.any() else []
    for x0, y0, x1, y1 in boxes:
        patch_hole = wide[y0:y1, x0:x1]
        filled = lama_patch(out[y0:y1, x0:x1], patch_hole)
        # Cola só dentro do buraco, com uma borda suave de alguns pixels.
        alpha = cv2.GaussianBlur(cv2.dilate(patch_hole.astype(np.float32), np.ones((3, 3), np.uint8)),
                                 (0, 0), 1.0)[..., None]
        region = out[y0:y1, x0:x1].astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(np.rint(region * (1 - alpha) + filled * alpha), 0, 255).astype(np.uint8)
    last_note = (f"IA (LaMa, {runtime.device_name()}) em {len(boxes)} área(s) grande(s); "
                 f"Telea nos riscos finos" if boxes else "só riscos finos: Telea")
    return out


def inpaint(img: np.ndarray, mask: np.ndarray, method: str = "telea",
            radius: float = 3.0) -> np.ndarray:
    """Preenche as regiões brancas de ``mask`` com ``method`` (``telea``, ``ns`` ou ``ia``).

    ``radius`` é o tamanho da vizinhança consultada pelos métodos clássicos,
    na escala de 1000 px (ajustado automaticamente à resolução da foto).
    """
    if method not in METHODS:
        raise ValueError(f"método desconhecido: {method!r}. Opções: {', '.join(METHODS)}")
    if method == "ia":
        return inpaint_ai(img, mask, radius)
    return inpaint_classic(img, mask, method, radius)
