"""Testes das etapas de Deep Learning (retrato_vivo/ai).

Os testes que rodam a rede são pulados se o modelo não foi baixado
(``python -m retrato_vivo modelos``); os demais rodam sempre.
"""

import importlib

import cv2
import numpy as np
import pytest

from retrato_vivo import preprocess
from retrato_vivo.ai import colorize, colorize_image, make_colorize_stage
from retrato_vivo.cli import build_parser
from retrato_vivo.pipeline import STEP_INFO, restore

# ``retrato_vivo.ai.colorize`` é a função (exportada no __init__); o módulo vem daqui.
colorize_module = importlib.import_module("retrato_vivo.ai.colorize")

needs_colorization_model = pytest.mark.skipif(
    not colorize_module.is_available(),
    reason="modelo de colorização não baixado (python -m retrato_vivo modelos)")


def make_color_scene(h: int = 240, w: int = 320, seed: int = 0) -> np.ndarray:
    """Cena colorida simples: céu azul, grama verde e círculos de cores variadas."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (200, 150, 90)       # céu (BGR)
    img[h // 2:] = (60, 140, 50)         # grama
    for _ in range(5):
        c = (int(rng.uniform(0, w)), int(rng.uniform(0, h)))
        color = tuple(int(v) for v in rng.uniform(30, 230, 3))
        cv2.circle(img, c, int(rng.uniform(15, 50)), color, -1)
    return cv2.GaussianBlur(img, (0, 0), 1.5)


def gray_scene() -> np.ndarray:
    return preprocess.to_neutral_gray(make_color_scene())


def chroma(img: np.ndarray) -> float:
    """Saturação média no LAB (0 = cinza)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    return float(np.hypot(lab[..., 1] - 128, lab[..., 2] - 128).mean())


def lightness(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)


# ============================================================ colorize.py

@needs_colorization_model
def test_colorize_adds_color_and_keeps_the_details():
    gray = gray_scene()
    out = colorize_image(gray, strength=1.0)
    assert out.shape == gray.shape and out.dtype == np.uint8
    assert chroma(gray) < 1.0 and chroma(out) > 3.0
    # a luminosidade (os detalhes da foto) é a da original
    assert np.abs(lightness(out) - lightness(gray)).mean() < 2.0


@needs_colorization_model
def test_colorize_strength_controls_how_much_color():
    gray = gray_scene()
    none, half, full = (colorize_image(gray, s) for s in (0.0, 0.5, 1.0))
    assert chroma(none) < 1.0
    assert chroma(none) < chroma(half) < chroma(full)


@needs_colorization_model
@pytest.mark.parametrize("shape", [(37, 501), (480, 64), (300, 300)])
def test_colorize_any_size(shape):
    gray = cv2.resize(gray_scene(), shape[::-1])
    assert colorize_image(gray).shape == gray.shape


@needs_colorization_model
def test_pipeline_colorizes_black_and_white_photo():
    res = restore(gray_scene(), post_stages=[("colorizacao", colorize)])
    assert res.diagnosis.monochrome
    assert "colorizacao" in res.steps and "colorizacao" in res.timings_ms
    assert res.notes["colorizacao"].startswith("foto P&B")
    assert chroma(res.image) > 3.0


def test_color_photo_passes_through_untouched():
    """Não precisa do modelo: a etapa decide antes de carregar a rede."""
    photo = make_color_scene()
    res = restore(photo, post_stages=[("colorizacao", colorize)])
    assert not res.diagnosis.monochrome
    assert res.notes["colorizacao"].startswith("pulada")
    assert np.array_equal(res.steps["colorizacao"], res.steps.get("nitidez", res.image))


def test_missing_model_is_skipped_with_explanation(monkeypatch):
    monkeypatch.setattr(colorize_module, "is_available", lambda: False)
    gray = gray_scene()
    res = restore(gray, post_stages=[("colorizacao", make_colorize_stage())])
    assert "modelo não baixado" in res.notes["colorizacao"]
    assert chroma(res.image) < 1.0


def test_colorization_step_has_a_description():
    assert "colorizacao" in STEP_INFO


def test_cli_accepts_colorization_options():
    args = build_parser().parse_args(["restaurar", "foto.jpg", "--colorir", "--intensidade-cor", "0.6"])
    assert args.colorir and args.intensidade_cor == 0.6
