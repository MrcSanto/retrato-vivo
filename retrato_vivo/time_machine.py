"""Máquina do tempo (bônus): transforma uma foto atual numa foto de outra época.

É o inverso da restauração. Cada época é uma função curta que lista, em
ordem, os efeitos de ``aging.py`` que formam o "visual" daquela década.
Para criar uma época nova, escreva outra função e registre em ``ERAS``.

Uso::

    from retrato_vivo.time_machine import travel, ERAS
    antiga = travel(foto, "1920")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import aging as fx


@dataclass(frozen=True)
class Era:
    key: str            # usado na linha de comando e na GUI
    name: str           # nome para mostrar ao usuário
    description: str
    apply: Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _era_1900(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Retrato de estúdio em papel albuminado: sépia forte, lente suave,
    cantos escuros e muito desgaste."""
    img = fx.black_and_white(img)
    img = fx.soften(img, 1.6)
    img = fx.sepia(img, 1.0)
    img = fx.fade(img, 0.30, paper=(150, 185, 205))
    img = fx.vignette(img, 0.9)
    img = fx.grain(img, 10, rng)
    img = fx.add_scratches(img, 14, rng)
    img = fx.add_spots(img, 90, rng)
    return fx.add_border(img, 0.05, color=(170, 200, 215))


def _era_1920(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Foto de família dos anos 20: preto e branco amarelado, granulado, riscado."""
    img = fx.black_and_white(img)
    img = fx.soften(img, 0.9)
    img = fx.sepia(img, 0.55)
    img = fx.fade(img, 0.25)
    img = fx.vignette(img, 0.6)
    img = fx.grain(img, 9, rng)
    img = fx.add_scratches(img, 10, rng)
    img = fx.add_spots(img, 50, rng)
    return fx.add_border(img, 0.04)


def _era_1950(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Preto e branco de fotojornalismo: contraste alto, grão fino, borda branca."""
    img = fx.black_and_white(img)
    img = fx.contrast(img, 1.25)
    img = fx.vignette(img, 0.3)
    img = fx.grain(img, 6, rng)
    img = fx.add_spots(img, 12, rng)
    return fx.add_border(img, 0.04, color=(240, 242, 242))


def _era_1970(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cópia colorida dos anos 70 que desbotou: tudo puxado para o
    amarelo-avermelhado, pretos lavados, cores fracas."""
    img = fx.soften(img, 0.6)
    img = fx.saturation(img, 0.75)
    img = fx.color_cast(img, bgr_gain=(0.82, 0.97, 1.06), lift=(8, 18, 30))
    img = fx.contrast(img, 0.88)
    img = fx.vignette(img, 0.35)
    img = fx.grain(img, 5, rng)
    img = fx.add_spots(img, 8, rng)
    return fx.add_border(img, 0.035, color=(225, 238, 242))


def _era_1990(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Polaroid dos anos 90: quadrada, contraste suave, sombras esverdeadas,
    data laranja no canto e moldura branca de base larga."""
    img = fx.contrast(img, 0.92)
    img = fx.saturation(img, 0.9)
    img = fx.color_cast(img, bgr_gain=(0.96, 1.0, 1.02), lift=(10, 14, 4))
    img = fx.soften(img, 0.5)
    img = fx.grain(img, 3, rng)
    img = fx.crop_square(img)
    day, month = int(rng.integers(1, 29)), int(rng.integers(1, 13))
    img = fx.date_stamp(img, f"{day:02d} {month:02d} '95")
    return fx.polaroid_frame(img)


ERAS: dict[str, Era] = {e.key: e for e in (
    Era("1900", "1900 - Retrato de estúdio", "Sépia forte, cantos escuros, muito desgaste", _era_1900),
    Era("1920", "1920 - Álbum de família", "Preto e branco amarelado, riscos e poeira", _era_1920),
    Era("1950", "1950 - Fotojornalismo", "Preto e branco contrastado com borda branca", _era_1950),
    Era("1970", "1970 - Cópia colorida desbotada", "Cores lavadas puxadas para o amarelo", _era_1970),
    Era("1990", "1990 - Polaroid", "Quadrada, moldura branca e data laranja", _era_1990),
)}


def travel(img: np.ndarray, era: str, seed: int | None = None) -> np.ndarray:
    """Leva a foto para a época ``era`` (uma das chaves de ``ERAS``).

    ``seed`` fixa a posição dos riscos e manchas (mesmo resultado sempre).
    A imagem de saída pode ser maior que a de entrada por causa da moldura.
    """
    if era not in ERAS:
        raise ValueError(f"época desconhecida: {era!r}. Opções: {', '.join(ERAS)}")
    return ERAS[era].apply(img, np.random.default_rng(seed))
