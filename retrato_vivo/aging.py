"""Envelhecimento sintético: estraga uma foto boa de propósito.

Serve para MEDIR a restauração. ``age_photo`` envelhece uma foto limpa e
devolve também a máscara verdadeira dos danos; depois de restaurar, dá para
comparar com a original (comando ``avaliar`` e ``metrics.py``).

Cada função aplica UM efeito de foto antiga. Elas também são as peças de
montar da **Máquina do tempo** (bônus do projeto): uma época é só uma
sequência destes efeitos. Exemplo de "foto de 1920"::

    rng = np.random.default_rng()
    img = black_and_white(foto)
    img = soften(img, 0.9)
    img = sepia(img, 0.55)
    img = fade(img, 0.25)
    img = vignette(img, 0.6)
    img = grain(img, 9, rng)
    img = add_scratches(img, 10, rng)
    img = add_spots(img, 50, rng)
    img = add_border(img, 0.04)

Efeitos disponíveis: tom e cor (``black_and_white``, ``sepia``, ``fade``,
``contrast``, ``saturation``, ``color_cast``), lente e filme (``soften``,
``vignette``, ``grain``), danos (``add_scratches``, ``add_spots``) e
acabamento (``crop_square``, ``add_border``, ``polaroid_frame``, ``date_stamp``).

Todas recebem e devolvem imagem BGR ``uint8``. Tamanhos são relativos à
resolução, então o efeito fica igual numa foto pequena ou grande.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _scale(img: np.ndarray) -> float:
    return max(img.shape[:2]) / 1000.0


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- tom e cor

def black_and_white(img: np.ndarray) -> np.ndarray:
    """Preto e branco (mantém 3 canais, para encadear com os outros efeitos)."""
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def sepia(img: np.ndarray, amount: float = 1.0) -> np.ndarray:
    """Tom marrom-amarelado de papel fotográfico antigo. ``amount`` de 0 a 1."""
    matrix = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]], dtype=np.float32)
    toned = cv2.transform(img.astype(np.float32), matrix)
    return _u8(img * (1 - amount) + toned * amount)


def fade(img: np.ndarray, amount: float = 0.35,
         paper: tuple[int, int, int] = (175, 200, 215)) -> np.ndarray:
    """Desbotamento: mistura a foto com a cor do papel (perde preto e contraste)."""
    return _u8(img * (1 - amount) + np.array(paper, np.float32) * amount)


def contrast(img: np.ndarray, gain: float = 1.2) -> np.ndarray:
    """Aumenta (>1) ou reduz (<1) o contraste em torno do cinza médio."""
    return _u8((img.astype(np.float32) - 128.0) * gain + 128.0)


def saturation(img: np.ndarray, factor: float = 0.7) -> np.ndarray:
    """Cores mais fracas (<1) ou mais fortes (>1). 0 deixa a foto cinza."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def color_cast(img: np.ndarray, bgr_gain: tuple[float, float, float],
               lift: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Dominante de cor, como numa cópia colorida que desbotou: multiplica
    cada canal (B, G, R) por ``bgr_gain`` e soma ``lift`` (clareia as
    sombras com aquela cor). Ex.: anos 70 = ``(0.82, 0.97, 1.06)``, ``(8, 18, 30)``."""
    return _u8(img.astype(np.float32) * np.array(bgr_gain, np.float32)
               + np.array(lift, np.float32))


# ------------------------------------------------------------ lente e filme

def soften(img: np.ndarray, sigma: float = 0.8) -> np.ndarray:
    """Desfoque leve de lente antiga. ``sigma`` na escala de 1000 px."""
    return cv2.GaussianBlur(img, (0, 0), max(0.3, sigma * _scale(img)))


def vignette(img: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """Cantos escurecidos."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    r = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    falloff = 1 - amount * np.clip(r - 0.4, 0, None) ** 2
    return _u8(img * falloff[..., None])


def grain(img: np.ndarray, sigma: float = 8.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Granulação de filme (ruído gaussiano igual nos três canais)."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, sigma, img.shape[:2])
    return _u8(img + noise[..., None])


# ------------------------------------------------------------------- danos

def add_scratches(img: np.ndarray, n: int, rng: np.random.Generator,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """Desenha ``n`` riscos finos (a maioria claros e verticais, como em filme).
    Se ``mask`` for dada, marca nela onde cada risco foi desenhado."""
    out = img.copy()
    h, w = img.shape[:2]
    s = _scale(img)
    for _ in range(n):
        vertical = rng.random() < 0.6
        length = rng.uniform(0.15, 0.8) * (h if vertical else w)
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        angle = (np.pi / 2 if vertical else 0.0) + rng.normal(0, 0.25)
        t = np.linspace(0, 1, 8)
        bend = rng.normal(0, 0.03) * length * np.sin(np.pi * t)  # leve curvatura
        xs = x0 + np.cos(angle) * length * t - np.sin(angle) * bend
        ys = y0 + np.sin(angle) * length * t + np.cos(angle) * bend
        pts = np.stack([xs, ys], 1).round().astype(np.int32).reshape(-1, 1, 2)
        thickness = max(1, int(round(rng.uniform(1, 2.5) * s)))
        value = int(rng.uniform(215, 255)) if rng.random() < 0.75 else int(rng.uniform(0, 40))
        cv2.polylines(out, [pts], False, (value, value, value), thickness, cv2.LINE_AA)
        if mask is not None:
            cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_8)
    return out


def add_spots(img: np.ndarray, n: int, rng: np.random.Generator,
              mask: np.ndarray | None = None) -> np.ndarray:
    """Desenha ``n`` manchas pequenas de poeira (escuras ou claras)."""
    out = img.copy()
    h, w = img.shape[:2]
    s = _scale(img)
    for _ in range(n):
        center = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        axes = (max(1, int(rng.uniform(1, 5) * s)), max(1, int(rng.uniform(1, 5) * s)))
        angle = float(rng.uniform(0, 180))
        value = int(rng.uniform(0, 40)) if rng.random() < 0.6 else int(rng.uniform(220, 255))
        cv2.ellipse(out, center, axes, angle, 0, 360, (value, value, value), -1, cv2.LINE_AA)
        if mask is not None:
            cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1, cv2.LINE_8)
    return out


# --------------------------------------------------------------- acabamento

def crop_square(img: np.ndarray) -> np.ndarray:
    """Recorta o quadrado central da foto (formato Polaroid/Instax)."""
    h, w = img.shape[:2]
    side = min(h, w)
    y0, x0 = (h - side) // 2, (w - side) // 2
    return img[y0:y0 + side, x0:x0 + side].copy()


def add_border(img: np.ndarray, frac: float = 0.04,
               color: tuple[int, int, int] = (235, 240, 242)) -> np.ndarray:
    """Borda de papel ao redor da foto (``frac`` do maior lado). A imagem cresce."""
    b = max(2, int(round(frac * max(img.shape[:2]))))
    return cv2.copyMakeBorder(img, b, b, b, b, cv2.BORDER_CONSTANT, value=color)


def polaroid_frame(img: np.ndarray, color: tuple[int, int, int] = (238, 243, 245)) -> np.ndarray:
    """Moldura de Polaroid: fina em cima e dos lados, larga embaixo.
    Use depois de ``crop_square``. A imagem cresce."""
    side = min(img.shape[:2])
    thin, thick = int(side * 0.06), int(side * 0.24)
    return cv2.copyMakeBorder(img, thin, thick, thin, thin, cv2.BORDER_CONSTANT, value=color)


def date_stamp(img: np.ndarray, text: str, margin_frac: float = 0.04) -> np.ndarray:
    """Data laranja no canto inferior direito, como nas câmeras dos anos 90.
    Só caracteres ASCII (limitação das fontes do OpenCV), ex.: ``"12 08 '95"``."""
    out = img.copy()
    h, w = img.shape[:2]
    font_scale = max(0.4, min(h, w) / 700.0)
    thickness = max(1, int(round(font_scale * 2)))
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    m = int(margin_frac * min(h, w))
    org = (max(0, w - tw - m), h - m)
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_DUPLEX, font_scale, (20, 60, 120),
                thickness + 2, cv2.LINE_AA)   # halo escuro
    cv2.putText(out, text, org, cv2.FONT_HERSHEY_DUPLEX, font_scale, (40, 150, 255),
                thickness, cv2.LINE_AA)       # laranja
    return out


# --------------------------------------------- envelhecimento para avaliação

@dataclass
class AgingConfig:
    """Quanto de cada efeito aplicar em ``age_photo`` (0 desliga o efeito)."""

    n_scratches: int = 12
    n_spots: int = 60
    sepia: float = 0.8
    fade: float = 0.35
    grain: float = 8.0
    vignette: float = 0.5
    blur: float = 0.8
    seed: int | None = None

    @classmethod
    def only_damage(cls, seed: int | None = None) -> "AgingConfig":
        """Só riscos e manchas, sem mexer no tom: permite comparar a foto
        restaurada com a original pixel a pixel."""
        return cls(sepia=0, fade=0, grain=0, vignette=0, blur=0, seed=seed)


def age_photo(img: np.ndarray, cfg: AgingConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Envelhece a foto. Retorna ``(foto_envelhecida, mascara_verdadeira_uint8)``.

    A máscara marca só os danos que ficaram visíveis: um risco branco sobre
    um céu branco não muda nada e não deve ser cobrado do detector.
    """
    cfg = cfg or AgingConfig()
    rng = np.random.default_rng(cfg.seed)

    toned = img
    if cfg.blur > 0:
        toned = soften(toned, cfg.blur)
    if cfg.sepia > 0:
        toned = sepia(toned, cfg.sepia)
    if cfg.fade > 0:
        toned = fade(toned, cfg.fade)
    if cfg.vignette > 0:
        toned = vignette(toned, cfg.vignette)
    if cfg.grain > 0:
        toned = grain(toned, cfg.grain, rng)

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    damaged = add_scratches(toned, cfg.n_scratches, rng, mask)
    damaged = add_spots(damaged, cfg.n_spots, rng, mask)

    visible = cv2.absdiff(damaged, toned).max(axis=2) > 12
    mask[~visible] = 0
    return damaged, mask
