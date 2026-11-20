"""Retrato Vivo: transforma a foto restaurada num vídeo curto com efeito 3D.

É o diferencial do projeto (o "algo a mais" pedido pelo professor). Uma câmera
virtual "passeia" diante da foto: o que está perto se mexe mais que o que está
longe (paralaxe), e a pessoa se destaca do fundo como numa cena de verdade.

Etapas:

1. **Profundidade (Deep Learning):** Depth Anything V2 Small (Yang et al., NeurIPS
   2024, Apache-2.0) estima, para cada pixel, o quão perto da câmera ele está.
2. **Camadas:** a profundidade é dividida em primeiro plano (a pessoa) e fundo
   (limiar de Otsu sobre a profundidade + morfologia + borda suave).
3. **Fundo escondido (Deep Learning):** ao mover a câmera, aparece o que estava
   atrás da pessoa. O LaMa (``inpaint.py``) inventa esse pedaço de fundo antes.
4. **Câmera virtual:** em cada quadro, cada camada é deslocada com
   ``cv2.remap`` proporcionalmente à profundidade de cada pixel, com um leve zoom,
   e o primeiro plano é composto sobre o fundo.

As redes rodam na GPU quando houver (``runtime.py``); o resto é OpenCV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import models, runtime
from .inpaint import inpaint
from .io_utils import resize_max_side

DEPTH_MODEL = "profundidade"
DEPTH_SIDE = 518                 # resolução de trabalho do Depth Anything (múltiplo de 14)
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # normalização ImageNet, como no treino
_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class LivingPortraitConfig:
    seconds: float = 4.0         # duração de uma volta da câmera
    fps: int = 25
    max_side: int = 1024         # resolução do vídeo (maior lado)
    motion: float = 0.025        # amplitude do movimento da câmera (fração do maior lado)
    zoom: float = 0.05           # aproximação máxima da câmera durante a volta
    fill_background: bool = True # usa o LaMa para inventar o fundo atrás da pessoa


@dataclass
class LivingPortrait:
    frames: list[np.ndarray]                  # quadros BGR uint8
    depth: np.ndarray                         # 0 = longe, 1 = perto
    foreground: np.ndarray                    # alfa do primeiro plano (0 a 1)
    background: np.ndarray                    # fundo com a pessoa removida
    fps: int
    timings_ms: dict[str, float] = field(default_factory=dict)
    device: str = "CPU"


def is_available() -> bool:
    return runtime.is_available() and models.is_available(DEPTH_MODEL)


# ============================================================ 1. profundidade

def estimate_depth(img: np.ndarray) -> np.ndarray:
    """Profundidade relativa de cada pixel, de 0 (mais longe) a 1 (mais perto)."""
    h, w = img.shape[:2]
    k = DEPTH_SIDE / max(h, w)
    nh, nw = max(14, round(h * k / 14) * 14), max(14, round(w * k / 14) * 14)
    x = cv2.cvtColor(cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    x = ((x.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
    raw = runtime.session(models.model_path(DEPTH_MODEL)).run(None, {"pixel_values": x})[0][0]
    lo, hi = np.percentile(raw, (2, 98))
    depth = np.clip((raw - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_CUBIC)
    # Suaviza sem borrar o contorno da pessoa (filtro bilateral).
    return np.clip(cv2.bilateralFilter(depth, 9, 0.1, 9), 0, 1)


def depth_colormap(depth: np.ndarray) -> np.ndarray:
    """Profundidade em cores para mostrar na interface (quente = perto)."""
    return cv2.applyColorMap((depth * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


# =================================================================== 2. camadas

def split_layers(depth: np.ndarray) -> np.ndarray:
    """Alfa do primeiro plano (0 a 1): o que está bem mais perto que o resto da cena."""
    d8 = (depth * 255).astype(np.uint8)
    t, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    near = (d8 > t).astype(np.uint8)
    s = max(depth.shape) / 1000.0
    k = max(3, int(9 * s) | 1)
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    # Fica só com as regiões grandes (a pessoa), sem ilhas soltas.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(near)
    keep = stats[:, cv2.CC_STAT_AREA] >= 0.02 * near.size
    keep[0] = False
    near = keep[labels].astype(np.float32)
    return np.clip(cv2.GaussianBlur(near, (0, 0), max(1.5, 3 * s)), 0, 1)


def hidden_background(img: np.ndarray, alpha: np.ndarray, use_ai: bool = True) -> np.ndarray:
    """O fundo com a pessoa apagada: o que a câmera revela quando se move."""
    s = max(img.shape[:2]) / 1000.0
    hole = cv2.dilate((alpha > 0.05).astype(np.uint8) * 255,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(25 * s) | 1),) * 2))
    return inpaint(img, hole, "ia" if use_ai else "telea")


# ============================================================ 3. câmera virtual

def _warp(img: np.ndarray, grid: tuple[np.ndarray, np.ndarray], shift_x: np.ndarray,
          shift_y: np.ndarray, zoom: float) -> np.ndarray:
    """Desloca cada pixel por (shift_x, shift_y) e aproxima a câmera por ``zoom``.

    ``grid`` são as coordenadas (x, y) de cada pixel relativas ao centro, calculadas
    uma vez só para todos os quadros."""
    gx, gy = grid
    h, w = img.shape[:2]
    map_x = gx * np.float32(1.0 / zoom) + np.float32(w / 2.0) - shift_x     # o remap exige float32
    map_y = gy * np.float32(1.0 / zoom) + np.float32(h / 2.0) - shift_y
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def render_frames(img: np.ndarray, depth: np.ndarray, alpha: np.ndarray, background: np.ndarray,
                  cfg: LivingPortraitConfig) -> list[np.ndarray]:
    """Uma volta completa da câmera (o vídeo pode repetir sem emenda)."""
    h, w = img.shape[:2]
    n = max(2, int(round(cfg.seconds * cfg.fps)))
    amplitude = cfg.motion * max(h, w)
    base_zoom = 1.0 + 2.5 * cfg.motion            # margem para a borda da foto nunca aparecer
    # Fundo: usa a profundidade do próprio fundo (a da pessoa é trocada pela média do fundo).
    bg_depth = depth.copy()
    bg_depth[alpha > 0.05] = float(depth[alpha <= 0.05].mean()) if (alpha <= 0.05).any() else 0.0
    bg_depth = cv2.GaussianBlur(bg_depth, (0, 0), 5)
    fg_depth = cv2.GaussianBlur(depth, (0, 0), 3)
    fg_layer = np.dstack([img, (alpha * 255).astype(np.uint8)])            # BGRA: foto + alfa
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    grid = (xs - w / 2.0, ys - h / 2.0)
    bg_depth, fg_depth = bg_depth.astype(np.float32), fg_depth.astype(np.float32)
    frames = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        dx = np.float32(np.cos(angle) * amplitude)                         # elipse: mais lateral
        dy = np.float32(0.5 * np.sin(angle) * amplitude)
        zoom = base_zoom * (1.0 + cfg.zoom * 0.5 * (1 - np.cos(angle)))
        bg = _warp(background, grid, dx * bg_depth, dy * bg_depth, zoom)
        fg = _warp(fg_layer, grid, dx * fg_depth, dy * fg_depth, zoom)
        a = fg[..., 3:4].astype(np.float32) / 255.0
        frames.append((fg[..., :3] * a + bg * (1 - a)).astype(np.uint8))
    return frames


def make_living_portrait(img: np.ndarray, cfg: LivingPortraitConfig | None = None) -> LivingPortrait:
    """Cria o Retrato Vivo a partir de uma foto (de preferência já restaurada)."""
    cfg = cfg or LivingPortraitConfig()
    if not is_available():
        raise RuntimeError("modelo de profundidade não baixado (python -m retrato_vivo modelos profundidade)")
    timings: dict[str, float] = {}
    img, _ = resize_max_side(img, cfg.max_side)
    h, w = img.shape[:2]
    img = img[: h - h % 2, : w - w % 2]           # vídeo MP4 exige lados pares

    t0 = time.perf_counter()
    depth = estimate_depth(img)
    timings["profundidade"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    alpha = split_layers(depth)
    background = hidden_background(img, alpha, cfg.fill_background)
    timings["camadas_e_fundo"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    frames = render_frames(img, depth, alpha, background, cfg)
    timings["quadros"] = (time.perf_counter() - t0) * 1000
    return LivingPortrait(frames=frames, depth=depth, foreground=alpha, background=background,
                          fps=cfg.fps, timings_ms=timings, device=runtime.device_name())


# ================================================================== 4. salvar

def save_video(portrait: LivingPortrait, path: str | Path, loops: int = 2) -> Path:
    """Salva em MP4 (``loops`` voltas da câmera)."""
    path = Path(path).with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = portrait.frames[0].shape[:2]
    # O VideoWriter não aceita acentos no caminho no Windows: grava num nome simples e renomeia.
    tmp = path.with_name("retrato_vivo_tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), portrait.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("não foi possível criar o vídeo MP4")
    for _ in range(loops):
        for frame in portrait.frames:
            writer.write(frame)
    writer.release()
    tmp.replace(path)
    return path


def save_gif(portrait: LivingPortrait, path: str | Path, max_side: int = 480) -> Path:
    """Salva em GIF animado (menor, para mandar por mensagem)."""
    from PIL import Image

    path = Path(path).with_suffix(".gif")
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in portrait.frames[::2]:              # metade dos quadros: arquivo bem menor
        small, _ = resize_max_side(frame, max_side)
        images.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
    images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                   duration=int(2000 / portrait.fps), optimize=True)
    return path
