"""Envelhecimento sintético: estraga uma foto boa de propósito.

Serve para MEDIR a restauração. ``age_photo`` envelhece uma foto limpa e
devolve também a máscara verdadeira dos danos; depois de restaurar, dá para
comparar com a original (comando ``avaliar`` e ``metrics.py``).

Cada função aplica UM efeito de foto antiga. A "Máquina do tempo" (bônus do
projeto) pode reaproveitar estas mesmas peças.

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
